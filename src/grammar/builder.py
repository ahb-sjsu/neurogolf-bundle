"""ONNX graph builder with typed shape tracking.

Usage:
    b = GraphBuilder()
    x = b.input()
    h = b.conv(x, weights_np)
    y = b.output(h)
    model = b.build()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper


# NeuroGolf canvas shape
CANVAS = (1, 10, 30, 30)
IR_VERSION = 10
OPSET = [helper.make_opsetid("", 10)]


@dataclass
class Tensor:
    """A tensor flowing through the graph."""
    name: str
    shape: tuple[int, ...]
    dtype: int = TensorProto.FLOAT  # ONNX type enum

    def __repr__(self):
        return f"Tensor({self.name}, {self.shape})"


@dataclass
class GraphBuilder:
    """Incremental ONNX graph builder.

    Tracks node list, initializers, value_info, and emits a valid
    ModelProto at build time.
    """
    nodes: list = field(default_factory=list)
    initializers: list = field(default_factory=list)
    value_info: list = field(default_factory=list)
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    _name_counter: int = 0

    def _fresh_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    # ---- Input / output ----

    def input(self, shape: tuple[int, ...] = CANVAS) -> Tensor:
        """Declare a named graph input."""
        t = Tensor("input", shape)
        self.inputs.append(helper.make_tensor_value_info(
            t.name, TensorProto.FLOAT, list(shape)))
        return t

    def output(self, tensor: Tensor, name: str = "output") -> None:
        """Mark a tensor as the graph output (re-names if needed)."""
        if tensor.name != name:
            # Add an Identity node to rename the last tensor to "output"
            self.nodes.append(helper.make_node(
                "Identity", [tensor.name], [name]))
        self.outputs.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, list(tensor.shape)))

    # ---- Initializers ----

    def const_tensor(self, arr: np.ndarray, name_prefix: str = "c") -> Tensor:
        """Add a constant initializer tensor."""
        name = self._fresh_name(name_prefix)
        dtype = TensorProto.FLOAT if arr.dtype == np.float32 else TensorProto.INT64
        init = helper.make_tensor(name, dtype, list(arr.shape), arr.flatten().tolist())
        self.initializers.append(init)
        return Tensor(name, tuple(arr.shape), dtype=dtype)

    # ---- Operators ----

    def conv(self, x: Tensor, weights: np.ndarray) -> Tensor:
        """Conv2d with given weights (shape: C_out, C_in, kH, kW)."""
        co, ci, kh, kw = weights.shape
        w_tensor = self.const_tensor(weights.astype(np.float32), "W")
        out_name = self._fresh_name("conv")
        pads = [kh // 2, kw // 2, kh // 2, kw // 2]
        self.nodes.append(helper.make_node(
            "Conv", [x.name, w_tensor.name], [out_name],
            kernel_shape=[kh, kw], pads=pads))
        out_shape = (x.shape[0], co, x.shape[2], x.shape[3])
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(out_shape)))
        return Tensor(out_name, out_shape)

    def relu(self, x: Tensor) -> Tensor:
        out_name = self._fresh_name("relu")
        self.nodes.append(helper.make_node("Relu", [x.name], [out_name]))
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(x.shape)))
        return Tensor(out_name, x.shape)

    def reshape(self, x: Tensor, target_shape: tuple[int, ...]) -> Tensor:
        shape_const = self.const_tensor(
            np.array(target_shape, dtype=np.int64), "shape")
        out_name = self._fresh_name("reshape")
        self.nodes.append(helper.make_node(
            "Reshape", [x.name, shape_const.name], [out_name]))
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(target_shape)))
        return Tensor(out_name, target_shape)

    def gather(self, x: Tensor, indices: np.ndarray, axis: int) -> Tensor:
        """Gather along an axis with integer indices."""
        idx_tensor = self.const_tensor(indices.astype(np.int64), "gi")
        out_name = self._fresh_name("gather")
        self.nodes.append(helper.make_node(
            "Gather", [x.name, idx_tensor.name], [out_name], axis=axis))
        # Infer output shape
        new_shape = list(x.shape)
        new_shape[axis] = len(indices)
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def mul(self, a: Tensor, b_or_arr) -> Tensor:
        """Element-wise multiply (b can be Tensor or numpy array constant)."""
        if isinstance(b_or_arr, np.ndarray):
            b = self.const_tensor(b_or_arr.astype(np.float32), "m")
        else:
            b = b_or_arr
        out_name = self._fresh_name("mul")
        self.nodes.append(helper.make_node(
            "Mul", [a.name, b.name], [out_name]))
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(a.shape)))
        return Tensor(out_name, a.shape)

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        out_name = self._fresh_name("add")
        self.nodes.append(helper.make_node(
            "Add", [a.name, b.name], [out_name]))
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(a.shape)))
        return Tensor(out_name, a.shape)

    def sub(self, a: Tensor, b: Tensor) -> Tensor:
        out_name = self._fresh_name("sub")
        self.nodes.append(helper.make_node(
            "Sub", [a.name, b.name], [out_name]))
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(a.shape)))
        return Tensor(out_name, a.shape)

    def transpose(self, x: Tensor, perm: list[int]) -> Tensor:
        out_name = self._fresh_name("transpose")
        self.nodes.append(helper.make_node(
            "Transpose", [x.name], [out_name], perm=perm))
        new_shape = tuple(x.shape[p] for p in perm)
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(new_shape)))
        return Tensor(out_name, new_shape)

    def slice(self, x: Tensor, starts: list[int], ends: list[int],
              axes: list[int]) -> Tensor:
        """Slice (opset 10 uses attributes)."""
        out_name = self._fresh_name("slice")
        self.nodes.append(helper.make_node(
            "Slice", [x.name], [out_name],
            starts=starts, ends=ends, axes=axes))
        # Shape inference
        new_shape = list(x.shape)
        for ax, s, e in zip(axes, starts, ends):
            e = min(e, x.shape[ax])
            new_shape[ax] = max(0, e - s)
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def pad(self, x: Tensor, pads: list[int], value: float = 0.0) -> Tensor:
        """Pad (opset 10 uses attributes)."""
        out_name = self._fresh_name("pad")
        self.nodes.append(helper.make_node(
            "Pad", [x.name], [out_name],
            mode="constant", pads=pads, value=value))
        # pads = [begin_d0, begin_d1, ..., end_d0, end_d1, ...]
        n = len(x.shape)
        new_shape = list(x.shape)
        for i in range(n):
            new_shape[i] += pads[i] + pads[i + n]
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def reduce_max(self, x: Tensor, axes: list[int], keepdims: int = 0) -> Tensor:
        out_name = self._fresh_name("rmax")
        self.nodes.append(helper.make_node(
            "ReduceMax", [x.name], [out_name],
            axes=axes, keepdims=keepdims))
        new_shape = list(x.shape)
        for a in sorted(axes, reverse=True):
            if keepdims:
                new_shape[a] = 1
            else:
                del new_shape[a]
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def argmax(self, x: Tensor, axis: int, keepdims: int = 0) -> Tensor:
        out_name = self._fresh_name("amax")
        self.nodes.append(helper.make_node(
            "ArgMax", [x.name], [out_name],
            axis=axis, keepdims=keepdims))
        new_shape = list(x.shape)
        if keepdims:
            new_shape[axis] = 1
        else:
            del new_shape[axis]
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.INT64, new_shape))
        return Tensor(out_name, tuple(new_shape), dtype=TensorProto.INT64)

    def concat(self, tensors: list[Tensor], axis: int) -> Tensor:
        out_name = self._fresh_name("concat")
        self.nodes.append(helper.make_node(
            "Concat", [t.name for t in tensors], [out_name], axis=axis))
        new_shape = list(tensors[0].shape)
        new_shape[axis] = sum(t.shape[axis] for t in tensors)
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def squeeze(self, x: Tensor, axes: list[int]) -> Tensor:
        out_name = self._fresh_name("squeeze")
        self.nodes.append(helper.make_node(
            "Squeeze", [x.name], [out_name], axes=axes))
        new_shape = list(x.shape)
        for a in sorted(axes, reverse=True):
            del new_shape[a]
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def unsqueeze(self, x: Tensor, axes: list[int]) -> Tensor:
        out_name = self._fresh_name("unsq")
        self.nodes.append(helper.make_node(
            "Unsqueeze", [x.name], [out_name], axes=axes))
        new_shape = list(x.shape)
        for a in sorted(axes):
            new_shape.insert(a, 1)
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, new_shape))
        return Tensor(out_name, tuple(new_shape))

    def clip(self, x: Tensor, min_val: float = None, max_val: float = None) -> Tensor:
        out_name = self._fresh_name("clip")
        attrs = {}
        if min_val is not None: attrs['min'] = min_val
        if max_val is not None: attrs['max'] = max_val
        self.nodes.append(helper.make_node(
            "Clip", [x.name], [out_name], **attrs))
        self.value_info.append(helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, list(x.shape)))
        return Tensor(out_name, x.shape)

    # ---- Build ----

    def build(self, graph_name: str = "net") -> onnx.ModelProto:
        """Materialize an ONNX ModelProto."""
        graph = helper.make_graph(
            self.nodes, graph_name, self.inputs, self.outputs,
            initializer=self.initializers, value_info=self.value_info,
        )
        model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET)
        return model
