"""Gradient-descent weight optimization for parameterized ops.

Given a DAGState (topology) with placeholder Conv weights, rebuild the
forward computation in PyTorch, train weights on task examples with
hinge-loss (threshold-aware), then extract the trained weights back into
the ONNX state.

This keeps topology search and weight optimization separate:
- DAG A* finds topologies
- WeightOptimizer trains weights per topology candidate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper

from .graph import DAGState
from .tensor import C, H, W, CANVAS_SHAPE


def grid_to_onehot_np(grid):
    t = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= H: break
        for c, color in enumerate(row):
            if c >= W: break
            if 0 <= color < C:
                t[0][color][r][c] = 1.0
    return t


def collect_tensors(task, device, subsets=('train', 'test', 'arc-gen')):
    inputs, outputs = [], []
    for sub in subsets:
        for ex in task.get(sub, []):
            inputs.append(grid_to_onehot_np(ex['input'])[0])
            outputs.append(grid_to_onehot_np(ex['output'])[0])
    if not inputs:
        return None, None
    X = torch.from_numpy(np.stack(inputs)).to(device)
    Y = torch.from_numpy(np.stack(outputs)).to(device)
    return X, Y


class TorchForward:
    """Run a DAGState's forward computation in PyTorch.

    For each op in the state's nodes, build a PyTorch layer that mirrors it.
    Conv weights become nn.Parameter (trainable).
    Static ops (Gather, Reshape, etc.) use their fixed attributes.
    """

    def __init__(self, state: DAGState, device='cpu'):
        self.device = device
        self.state = state
        # name → tensor for tracking intermediate values in forward()
        self._tensor_cache = {}
        # Parameters: name → nn.Parameter for Conv weights
        self.params = {}
        self.initializer_names = {init.name for init in state.initializers}
        self.init_values = {init.name: init for init in state.initializers}
        self._build()

    def _build(self):
        # Find Conv weights in initializers; make them learnable Parameters
        for init in self.state.initializers:
            if init.name.startswith("W_") or init.name.split("_")[0] == "W":
                # Heuristic: Conv weight names start with W
                np_arr = np.frombuffer(init.raw_data or b'', dtype=np.float32)
                if len(np_arr) == 0:
                    np_arr = np.array(init.float_data, dtype=np.float32)
                shape = tuple(init.dims)
                if not np_arr.size and shape:
                    np_arr = np.zeros(shape, dtype=np.float32)
                t = torch.from_numpy(np_arr.reshape(shape).astype(np.float32)).to(self.device)
                p = nn.Parameter(t)
                self.params[init.name] = p

    def parameters(self):
        return list(self.params.values())

    def forward(self, x):
        """Run the forward pass for input tensor x."""
        cache = {"input": x}
        for node in self.state.nodes:
            op = node.op_type
            ins = [cache[n] if n in cache else self._const(n) for n in node.input]
            out = self._apply(op, ins, node)
            cache[node.output[0]] = out
        # Final output: last node's first output
        if not self.state.nodes:
            return x
        return cache[self.state.nodes[-1].output[0]]

    def _const(self, name):
        """Fetch an initializer as a torch tensor (possibly trainable)."""
        if name in self.params:
            return self.params[name]
        init = self.init_values.get(name)
        if init is None:
            raise ValueError(f"No initializer {name}")
        dtype_map = {TensorProto.FLOAT: torch.float32,
                     TensorProto.INT64: torch.int64}
        np_dtype = {TensorProto.FLOAT: np.float32,
                    TensorProto.INT64: np.int64}[init.data_type]
        if init.raw_data:
            arr = np.frombuffer(init.raw_data, dtype=np_dtype)
        elif init.data_type == TensorProto.FLOAT:
            arr = np.array(init.float_data, dtype=np.float32)
        elif init.data_type == TensorProto.INT64:
            arr = np.array(init.int64_data, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported init dtype {init.data_type}")
        shape = tuple(init.dims) if init.dims else (arr.size,)
        t = torch.from_numpy(arr.reshape(shape).copy()).to(self.device)
        t = t.to(dtype_map.get(init.data_type, torch.float32))
        cache_key = name  # don't cache in instance to avoid mutability
        return t

    def _apply(self, op, ins, node):
        """Execute an ONNX node in PyTorch."""
        if op == "Identity":
            return ins[0]
        if op == "Conv":
            x, w = ins[0], ins[1]
            kh, kw = w.shape[2], w.shape[3]
            return F.conv2d(x, w, padding=(kh // 2, kw // 2))
        if op == "Relu":
            return F.relu(ins[0])
        if op == "Reshape":
            shape_arr = ins[1].detach().cpu().numpy().astype(int).tolist()
            # Substitute batch dim to match actual batch (ONNX spec uses 1 for single batch,
            # but training passes N examples). Use -1 for auto-inferred batch dim.
            if shape_arr and shape_arr[0] == 1 and ins[0].shape[0] != 1:
                shape_arr = [ins[0].shape[0]] + shape_arr[1:]
            return ins[0].reshape(shape_arr)
        if op == "Gather":
            axis = 0
            for a in node.attribute:
                if a.name == "axis": axis = a.i
            return torch.index_select(ins[0], dim=axis, index=ins[1].long())
        if op == "Transpose":
            perm = list(range(ins[0].dim()))
            for a in node.attribute:
                if a.name == "perm": perm = list(a.ints)
            return ins[0].permute(*perm)
        if op == "Slice":
            starts, ends, axes = None, None, None
            for a in node.attribute:
                if a.name == "starts": starts = list(a.ints)
                if a.name == "ends": ends = list(a.ints)
                if a.name == "axes": axes = list(a.ints)
            out = ins[0]
            for ax, s, e in zip(axes, starts, ends):
                out = out.narrow(ax, s, min(e, out.shape[ax]) - s)
            return out
        if op == "Pad":
            pads_attr = None
            value = 0.0
            for a in node.attribute:
                if a.name == "pads": pads_attr = list(a.ints)
                if a.name == "value": value = a.f
            # ONNX pad = [begin_d0, begin_d1, ..., end_d0, end_d1, ...]
            # PyTorch F.pad takes pads for dims in REVERSE order as pairs
            n = ins[0].dim()
            torch_pads = []
            for i in range(n - 1, -1, -1):
                torch_pads += [pads_attr[i], pads_attr[i + n]]
            return F.pad(ins[0], torch_pads, value=value)
        if op == "Concat":
            axis = 0
            for a in node.attribute:
                if a.name == "axis": axis = a.i
            return torch.cat(ins, dim=axis)
        if op == "Mul":
            return ins[0] * ins[1]
        if op == "Add":
            return ins[0] + ins[1]
        if op == "Sub":
            return ins[0] - ins[1]
        if op == "Squeeze":
            axes = []
            for a in node.attribute:
                if a.name == "axes": axes = list(a.ints)
            out = ins[0]
            for ax in sorted(axes, reverse=True):
                out = out.squeeze(ax)
            return out
        if op == "Unsqueeze":
            axes = []
            for a in node.attribute:
                if a.name == "axes": axes = list(a.ints)
            out = ins[0]
            for ax in sorted(axes):
                out = out.unsqueeze(ax)
            return out
        if op == "ReduceMax":
            axes, keepdims = None, 1
            for a in node.attribute:
                if a.name == "axes": axes = list(a.ints)
                if a.name == "keepdims": keepdims = a.i
            out = ins[0]
            for ax in sorted(axes, reverse=True):
                out = out.max(dim=ax, keepdim=bool(keepdims)).values
            return out
        if op == "ArgMax":
            axis, keepdims = 0, 0
            for a in node.attribute:
                if a.name == "axis": axis = a.i
                if a.name == "keepdims": keepdims = a.i
            return ins[0].argmax(dim=axis, keepdim=bool(keepdims)).to(torch.int64)
        if op == "Clip":
            min_v, max_v = None, None
            for a in node.attribute:
                if a.name == "min": min_v = a.f
                if a.name == "max": max_v = a.f
            return ins[0].clamp(min=min_v, max=max_v)
        raise ValueError(f"unsupported op {op}")


def _train_single(state: DAGState, X: torch.Tensor, Y: torch.Tensor,
                   device: str, steps: int, lr: float, margin: float,
                   seed: int, reinit: bool = True):
    """Single-seed training attempt. Returns (ok, forward) on converge.

    If reinit=False, keeps whatever initial weights the caller set up
    (useful for the first attempt so A*'s hardcoded init is tried first).
    """
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    forward = TorchForward(state, device=device)
    params = forward.parameters()

    # Re-init learnable params with this seed's draws, if requested
    if reinit:
        with torch.no_grad():
            for p in params:
                p.copy_(torch.randn_like(p) * 0.1)

    if not params:
        with torch.no_grad():
            pred = (forward.forward(X) > 0).float()
            return torch.all(pred == Y).item(), forward

    opt = torch.optim.Adam(params, lr=lr)
    best_loss = float("inf")
    for step in range(steps):
        opt.zero_grad()
        out = forward.forward(X)
        pos = torch.clamp(margin - out, min=0) * Y
        neg = torch.clamp(margin + out, min=0) * (1 - Y)
        loss = (pos.sum() + neg.sum()) / (X.numel() + 1e-9)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
        if step % 50 == 49:
            with torch.no_grad():
                pred = (forward.forward(X) > 0).float()
                if torch.all(pred == Y).item():
                    return True, forward

    with torch.no_grad():
        pred = (forward.forward(X) > 0).float()
        return torch.all(pred == Y).item(), forward


def train_weights(state: DAGState, X: torch.Tensor, Y: torch.Tensor,
                  device='cpu', steps=500, lr=0.05, margin=0.5,
                  num_seeds: int = 1):
    """Train weights in a DAGState to fit X → Y.

    Tries up to `num_seeds` random initializations; returns on first success.
    Returns (success, trained_state) where trained_state has updated
    ONNX initializers with the learned weights.
    """
    # Pass-through if no trainable params — verify only.
    forward0 = TorchForward(state, device=device)
    if not forward0.parameters():
        with torch.no_grad():
            pred = (forward0.forward(X) > 0).float()
            if torch.all(pred == Y).item():
                return True, state
        return False, state

    forward = None
    ok = False
    for seed_idx in range(num_seeds):
        # First attempt: use A*'s hardcoded initial weights (do not reinit).
        # Subsequent attempts: random new init per seed.
        ok, forward = _train_single(
            state, X, Y, device, steps, lr, margin,
            seed=1000 + seed_idx * 17,
            reinit=(seed_idx > 0))
        if ok:
            break

    if not ok:
        return False, state

    # Replace initializers in state with trained weights
    new_inits = []
    for init in state.initializers:
        if init.name in forward.params:
            trained = forward.params[init.name].detach().cpu().numpy().astype(np.float32)
            new_inits.append(helper.make_tensor(
                init.name, TensorProto.FLOAT, list(trained.shape),
                trained.flatten().tolist()))
        else:
            new_inits.append(init)

    new_state = DAGState(
        tensors=state.tensors,
        nodes=state.nodes,
        initializers=tuple(new_inits),
        total_cost=state.total_cost,
        op_labels=state.op_labels,
    )
    return True, new_state
