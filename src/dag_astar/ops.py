"""Operator catalog for DAG A*.

Each operator:
- Knows its input arity and valid input shape/type combinations
- Produces tensors with statically-inferable output shape
- Has an estimated cost contribution (params, memory, MACs)
- Emits itself as an ONNX node + initializers

Operators come in two flavors:
- Static: all behavior determined by hyperparameters (e.g., Transpose, Gather
  with given indices, Slice with given starts/ends). Fully defined at enum time.
- Parameterized: have learnable weights (e.g., Conv, MatMul). Weight values
  are optimized AFTER the topology is fixed via gradient descent.

Each operator type registers a builder that takes:
  (inputs: list[TensorRef], hyperparams: dict) -> (output_ref, onnx_nodes,
                                                    initializers, est_cost)

Search enumerates all (op_type, hyperparam) combinations given a list of
available input tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from onnx import TensorProto, helper

from .tensor import C, H, W, TensorRef


@dataclass
class OpCost:
    """Cost contribution of a single operator."""
    params: int = 0
    memory: int = 0  # intermediate tensors held in the model graph
    macs: int = 0

    @property
    def total(self) -> int:
        return self.params + self.memory + self.macs


@dataclass
class Emitted:
    """What an operator produces when added to the graph."""
    output: TensorRef
    nodes: list          # list of onnx.NodeProto
    initializers: list   # list of onnx.TensorProto
    cost: OpCost
    label: str           # human-readable for logging


# ----- Operator registry: (name -> builder_fn) -----
OP_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def dec(fn):
        OP_REGISTRY[name] = fn
        return fn
    return dec


# -------------------------------------------------------------------------
# Static operators (no learnable weights)
# -------------------------------------------------------------------------

@register("identity")
def op_identity(inputs, hp, name_gen):
    """output = input (copy via Identity node)."""
    x = inputs[0]
    out_name = name_gen("id")
    node = helper.make_node("Identity", [x.name], [out_name])
    out = TensorRef(out_name, x.shape, x.dtype, "identity")
    # Identity is free but occupies memory for its output
    cost = OpCost(params=0, memory=x.bytes_fp32, macs=0)
    return Emitted(out, [node], [], cost, "identity")


@register("transpose")
def op_transpose(inputs, hp, name_gen):
    """output = Transpose(input, perm).
    hp: {'perm': list[int] of len = input rank}
    """
    x = inputs[0]
    perm = hp["perm"]
    if sorted(perm) != list(range(x.rank)):
        raise ValueError(f"bad perm {perm} for rank {x.rank}")
    new_shape = tuple(x.shape[p] for p in perm)
    out_name = name_gen("tp")
    node = helper.make_node("Transpose", [x.name], [out_name], perm=perm)
    out = TensorRef(out_name, new_shape, x.dtype, "transpose")
    cost = OpCost(memory=x.bytes_fp32)
    return Emitted(out, [node], [], cost, f"transpose({perm})")


@register("reshape")
def op_reshape(inputs, hp, name_gen):
    """output = Reshape(input, target_shape)."""
    x = inputs[0]
    target = tuple(hp["target_shape"])
    # Verify element count
    if np.prod(target) != x.num_elements:
        raise ValueError(f"Reshape from {x.shape} to {target} not element-preserving")
    shape_name = name_gen("rs_shape")
    shape_init = helper.make_tensor(shape_name, TensorProto.INT64, [len(target)],
                                     list(target))
    out_name = name_gen("rs")
    node = helper.make_node("Reshape", [x.name, shape_name], [out_name])
    out = TensorRef(out_name, target, x.dtype, "reshape")
    cost = OpCost(memory=x.bytes_fp32 + 8 * len(target))
    return Emitted(out, [node], [shape_init], cost, f"reshape({target})")


@register("gather")
def op_gather(inputs, hp, name_gen):
    """output = Gather(input, indices, axis).
    hp: {'indices': np.ndarray int64, 'axis': int}
    """
    x = inputs[0]
    indices = np.asarray(hp["indices"], dtype=np.int64)
    axis = hp["axis"]
    new_shape = list(x.shape)
    # Gather replaces the single axis with the indices' shape
    new_shape[axis:axis+1] = list(indices.shape)
    new_shape = tuple(new_shape)

    idx_name = name_gen("gi")
    idx_init = helper.make_tensor(idx_name, TensorProto.INT64,
                                   list(indices.shape), indices.flatten().tolist())
    out_name = name_gen("gather")
    node = helper.make_node("Gather", [x.name, idx_name], [out_name], axis=axis)
    out = TensorRef(out_name, new_shape, x.dtype, "gather")
    cost = OpCost(
        memory=indices.size * 8 + x.bytes_fp32,
        macs=0,
    )
    import hashlib
    idx_hash = hashlib.sha256(indices.tobytes()).hexdigest()[:8]
    return Emitted(out, [node], [idx_init], cost, f"gather(axis={axis},{idx_hash})")


@register("slice")
def op_slice(inputs, hp, name_gen):
    """output = Slice(input, starts, ends, axes) using opset-10 attribute form."""
    x = inputs[0]
    starts = hp["starts"]
    ends = hp["ends"]
    axes = hp["axes"]
    new_shape = list(x.shape)
    for ax, s, e in zip(axes, starts, ends):
        e = min(e, x.shape[ax])
        new_shape[ax] = max(0, e - s)
    out_name = name_gen("slice")
    node = helper.make_node(
        "Slice", [x.name], [out_name],
        starts=starts, ends=ends, axes=axes,
    )
    out = TensorRef(out_name, tuple(new_shape), x.dtype, "slice")
    cost = OpCost(memory=sum(new_shape) * 4)
    return Emitted(out, [node], [], cost, f"slice({axes}, {starts}→{ends})")


@register("pad")
def op_pad(inputs, hp, name_gen):
    """output = Pad(input, pads) using opset-10 attribute form."""
    x = inputs[0]
    pads = hp["pads"]  # [begin_d0, begin_d1, ..., end_d0, end_d1, ...]
    value = hp.get("value", 0.0)
    n = x.rank
    new_shape = list(x.shape)
    for i in range(n):
        new_shape[i] += pads[i] + pads[i + n]
    out_name = name_gen("pad")
    node = helper.make_node(
        "Pad", [x.name], [out_name],
        mode="constant", pads=pads, value=value,
    )
    out = TensorRef(out_name, tuple(new_shape), x.dtype, "pad")
    bytes_out = 1
    for d in new_shape:
        bytes_out *= d
    bytes_out *= 4
    cost = OpCost(memory=bytes_out)
    return Emitted(out, [node], [], cost, f"pad({pads})")


@register("concat")
def op_concat(inputs, hp, name_gen):
    """output = Concat(inputs, axis)."""
    axis = hp["axis"]
    # Check shapes compatible except at concat axis
    ref = inputs[0]
    for other in inputs[1:]:
        for i, (a, b) in enumerate(zip(ref.shape, other.shape)):
            if i != axis and a != b:
                raise ValueError(f"concat shape mismatch at axis {i}")
    new_shape = list(ref.shape)
    new_shape[axis] = sum(t.shape[axis] for t in inputs)
    out_name = name_gen("concat")
    node = helper.make_node("Concat", [t.name for t in inputs], [out_name],
                             axis=axis)
    out = TensorRef(out_name, tuple(new_shape), ref.dtype, "concat")
    cost = OpCost(memory=sum(t.bytes_fp32 for t in inputs))
    return Emitted(out, [node], [], cost, f"concat(axis={axis})")


@register("mul_const")
def op_mul_const(inputs, hp, name_gen):
    """output = input * const_tensor (broadcast)."""
    x = inputs[0]
    const = np.asarray(hp["const"], dtype=np.float32)
    const_name = name_gen("mconst")
    const_init = helper.make_tensor(const_name, TensorProto.FLOAT,
                                     list(const.shape), const.flatten().tolist())
    out_name = name_gen("mul")
    node = helper.make_node("Mul", [x.name, const_name], [out_name])
    out = TensorRef(out_name, x.shape, x.dtype, "mul_const")
    cost = OpCost(params=const.size, memory=x.bytes_fp32,
                   macs=x.num_elements)
    return Emitted(out, [node], [const_init], cost, "mul_const")


@register("add_const")
def op_add_const(inputs, hp, name_gen):
    x = inputs[0]
    const = np.asarray(hp["const"], dtype=np.float32)
    const_name = name_gen("aconst")
    const_init = helper.make_tensor(const_name, TensorProto.FLOAT,
                                     list(const.shape), const.flatten().tolist())
    out_name = name_gen("add")
    node = helper.make_node("Add", [x.name, const_name], [out_name])
    out = TensorRef(out_name, x.shape, x.dtype, "add_const")
    cost = OpCost(params=const.size, memory=x.bytes_fp32,
                   macs=x.num_elements)
    return Emitted(out, [node], [const_init], cost, "add_const")


@register("relu")
def op_relu(inputs, hp, name_gen):
    x = inputs[0]
    out_name = name_gen("relu")
    node = helper.make_node("Relu", [x.name], [out_name])
    out = TensorRef(out_name, x.shape, x.dtype, "relu")
    cost = OpCost(memory=x.bytes_fp32, macs=x.num_elements)
    return Emitted(out, [node], [], cost, "relu")


@register("reduce_max")
def op_reduce_max(inputs, hp, name_gen):
    x = inputs[0]
    axes = hp["axes"]
    keepdims = hp.get("keepdims", 0)
    new_shape = list(x.shape)
    for a in sorted(axes, reverse=True):
        if keepdims:
            new_shape[a] = 1
        else:
            del new_shape[a]
    out_name = name_gen("rmax")
    node = helper.make_node("ReduceMax", [x.name], [out_name],
                             axes=axes, keepdims=keepdims)
    out = TensorRef(out_name, tuple(new_shape), x.dtype, "reduce_max")
    cost = OpCost(memory=x.bytes_fp32, macs=x.num_elements)
    return Emitted(out, [node], [], cost, f"reduce_max({axes})")


@register("argmax")
def op_argmax(inputs, hp, name_gen):
    x = inputs[0]
    axis = hp["axis"]
    keepdims = hp.get("keepdims", 0)
    new_shape = list(x.shape)
    if keepdims:
        new_shape[axis] = 1
    else:
        del new_shape[axis]
    out_name = name_gen("amax")
    node = helper.make_node("ArgMax", [x.name], [out_name],
                             axis=axis, keepdims=keepdims)
    out = TensorRef(out_name, tuple(new_shape), TensorProto.INT64, "argmax")
    cost = OpCost(memory=8 * max(1, int(np.prod(new_shape))),
                   macs=x.num_elements)
    return Emitted(out, [node], [], cost, f"argmax({axis})")


@register("squeeze")
def op_squeeze(inputs, hp, name_gen):
    x = inputs[0]
    axes = hp["axes"]
    new_shape = [d for i, d in enumerate(x.shape) if i not in axes]
    out_name = name_gen("sq")
    node = helper.make_node("Squeeze", [x.name], [out_name], axes=axes)
    out = TensorRef(out_name, tuple(new_shape), x.dtype, "squeeze")
    return Emitted(out, [node], [], OpCost(memory=0), f"squeeze({axes})")


@register("unsqueeze")
def op_unsqueeze(inputs, hp, name_gen):
    x = inputs[0]
    axes = hp["axes"]
    new_shape = list(x.shape)
    for a in sorted(axes):
        new_shape.insert(a, 1)
    out_name = name_gen("uns")
    node = helper.make_node("Unsqueeze", [x.name], [out_name], axes=axes)
    out = TensorRef(out_name, tuple(new_shape), x.dtype, "unsqueeze")
    return Emitted(out, [node], [], OpCost(memory=0), f"unsqueeze({axes})")


# -------------------------------------------------------------------------
# Parameterized operators (learnable weights)
# -------------------------------------------------------------------------

@register("conv")
def op_conv(inputs, hp, name_gen):
    """Conv2d with learnable weights.
    hp: {'weights': np.ndarray of shape (C_out, C_in, kH, kW)}
    """
    x = inputs[0]
    if x.rank != 4:
        raise ValueError(f"Conv expects rank-4 input, got {x.shape}")
    w = np.asarray(hp["weights"], dtype=np.float32)
    co, ci, kh, kw = w.shape
    if ci != x.shape[1]:
        raise ValueError(f"Conv input channels {x.shape[1]} != weight {ci}")
    w_name = name_gen("W")
    w_init = helper.make_tensor(w_name, TensorProto.FLOAT, list(w.shape),
                                 w.flatten().tolist())
    out_name = name_gen("conv")
    pads = [kh // 2, kw // 2, kh // 2, kw // 2]
    node = helper.make_node("Conv", [x.name, w_name], [out_name],
                             kernel_shape=[kh, kw], pads=pads)
    out_shape = (x.shape[0], co, x.shape[2], x.shape[3])
    out = TensorRef(out_name, out_shape, x.dtype, "conv")
    macs = co * ci * kh * kw * x.shape[2] * x.shape[3]
    bytes_out = 1
    for d in out_shape: bytes_out *= d
    bytes_out *= 4
    cost = OpCost(params=w.size, memory=bytes_out, macs=macs)
    return Emitted(out, [node], [w_init], cost, f"conv({co}x{ci}x{kh}x{kw})")
