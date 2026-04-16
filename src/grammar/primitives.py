"""High-level primitives built on GraphBuilder.

Each returns an ONNX ModelProto for the given spec.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import onnx

from .builder import GraphBuilder, CANVAS

C, H, W = 10, 30, 30


# =============================================================================
# Color-only primitives
# =============================================================================

def identity_network() -> onnx.ModelProto:
    """output = input (via 1x1 Conv with identity weights)."""
    b = GraphBuilder()
    x = b.input()
    w = np.eye(C, dtype=np.float32).reshape(C, C, 1, 1)
    h = b.conv(x, w)
    b.output(h)
    return b.build()


def color_remap_network(color_map: dict[int, int],
                         margin: float = 0.5) -> onnx.ModelProto:
    """1x1 Conv that maps input colors to output colors.

    For each input color c in {0..9}, W[color_map[c], c, 0, 0] = 1
    All other weights = -margin (forces other channels below threshold).

    For colors not in color_map, default to identity (c -> c).
    """
    w = np.full((C, C, 1, 1), -margin, dtype=np.float32)
    for ic in range(C):
        oc = color_map.get(ic, ic)
        w[oc, ic, 0, 0] = 1.0

    b = GraphBuilder()
    x = b.input()
    h = b.conv(x, w)
    b.output(h)
    return b.build()


# =============================================================================
# Spatial primitives (static Gather indices)
# =============================================================================

def gather_network(gather_indices: np.ndarray) -> onnx.ModelProto:
    """Permute pixels via Gather on (1, C, H*W).

    gather_indices: shape (H*W,) int64, giving source index for each output pos.
    """
    b = GraphBuilder()
    x = b.input()
    # Reshape (1,C,H,W) -> (1,C,H*W)
    flat = b.reshape(x, (1, C, H * W))
    # Gather along axis 2
    gathered = b.gather(flat, gather_indices, axis=2)
    # Reshape back
    out = b.reshape(gathered, (1, C, H, W))
    b.output(out)
    return b.build()


def gather_with_mask_network(gather_indices: np.ndarray,
                              mask: np.ndarray) -> onnx.ModelProto:
    """Gather + multiply by mask to zero out specific positions.

    gather_indices: (H*W,) int64
    mask: (1, C, H, W) float32 — 1.0 for kept positions, 0.0 for zeroed
    """
    b = GraphBuilder()
    x = b.input()
    flat = b.reshape(x, (1, C, H * W))
    gathered = b.gather(flat, gather_indices, axis=2)
    back = b.reshape(gathered, (1, C, H, W))
    masked = b.mul(back, mask.astype(np.float32))
    b.output(masked)
    return b.build()


# =============================================================================
# Local (Conv) primitives
# =============================================================================

def single_conv_network(weights: np.ndarray) -> onnx.ModelProto:
    """Single Conv2d. weights: (C_out, C_in, k, k)."""
    b = GraphBuilder()
    x = b.input()
    h = b.conv(x, weights)
    b.output(h)
    return b.build()


def two_layer_conv_network(w1: np.ndarray, w2: np.ndarray) -> onnx.ModelProto:
    """Conv -> ReLU -> Conv."""
    b = GraphBuilder()
    x = b.input()
    h1 = b.conv(x, w1)
    h2 = b.relu(h1)
    h3 = b.conv(h2, w2)
    b.output(h3)
    return b.build()


# =============================================================================
# Learned affine transform (the key new primitive for Week 1)
# =============================================================================

def _affine_indices(A: np.ndarray, b: np.ndarray,
                    in_h: int = H, in_w: int = W) -> np.ndarray:
    """Compute gather indices for an affine transformation.

    For each output position (r, c), find source position:
      (r_src, c_src) = round(A @ (r, c) + b)
    If out of range, use canvas corner (H-1, W-1) as sentinel
    (guaranteed zero for sub-canvas grids).

    Returns indices of shape (H*W,) for use in Gather on flat (1, C, H*W).
    """
    # Sentinel: use a corner position guaranteed to be zero for sub-canvas grids.
    # For full-canvas transforms, all positions are valid so sentinel is irrelevant.
    sentinel = (H - 1) * W + (W - 1)
    indices = np.full(H * W, sentinel, dtype=np.int64)
    for r in range(H):
        for c in range(W):
            src_rc = A @ np.array([r, c], dtype=np.float64) + b
            r_src = int(round(float(src_rc[0])))
            c_src = int(round(float(src_rc[1])))
            if 0 <= r_src < in_h and 0 <= c_src < in_w:
                indices[r * W + c] = r_src * W + c_src
    return indices


def _affine_mask(A: np.ndarray, b: np.ndarray,
                 in_h: int = H, in_w: int = W) -> np.ndarray:
    """Mask for affine transform: 1.0 where source is in-range, 0.0 else."""
    mask = np.zeros((1, C, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            src_rc = A @ np.array([r, c], dtype=np.float64) + b
            r_src = int(round(float(src_rc[0])))
            c_src = int(round(float(src_rc[1])))
            if 0 <= r_src < in_h and 0 <= c_src < in_w:
                mask[0, :, r, c] = 1.0
    return mask


def affine_gather_network(A: np.ndarray, b: np.ndarray,
                           in_h: int = H, in_w: int = W,
                           apply_mask: bool = True) -> onnx.ModelProto:
    """Apply spatial transform (r',c') = A·(r,c) + b via Gather.

    A: (2,2) float — the rotation/scale/shear matrix
    b: (2,) float — the translation
    in_h, in_w: input grid size (for masking out-of-range source)
    apply_mask: if True, zero out output pixels where source is out of range

    Example: 90° rotation is A = [[0, 1], [-1, 0]], b = [W-1, 0]
    """
    indices = _affine_indices(A, b, in_h, in_w)
    if apply_mask:
        mask = _affine_mask(A, b, in_h, in_w)
        return gather_with_mask_network(indices, mask)
    else:
        return gather_network(indices)


# =============================================================================
# Utilities
# =============================================================================

def score_model(model: onnx.ModelProto) -> dict:
    """Run through onnx_tool to get params+memory+macs.

    Returns dict with {params, memory, macs, cost, score} or None on error.
    """
    import tempfile
    import os
    import onnx_tool
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
    try:
        onnx.save(model, tmp_path)
        m = onnx_tool.loadmodel(tmp_path, {'verbose': False})
        g = m.graph
        g.graph_reorder_nodes()
        g.shape_infer(None)
        g.profile()
        if not g.valid_profile:
            return None
        macs = int(sum(g.macs))
        memory = int(g.memory)
        params = int(g.params)
        cost = macs + memory + params
        score = max(1.0, 25.0 - math.log(cost))
        return {
            'params': params, 'memory': memory, 'macs': macs,
            'cost': cost, 'score': score,
        }
    finally:
        try: os.unlink(tmp_path)
        except: pass


def verify_model(model: onnx.ModelProto, task: dict,
                 subsets: tuple = ('train', 'test', 'arc-gen')) -> tuple[int, int]:
    """Verify on task examples. Returns (correct, total)."""
    import onnxruntime
    import io

    buf = io.BytesIO()
    onnx.save(model, buf)

    sess = onnxruntime.InferenceSession(buf.getvalue(),
        providers=['CPUExecutionProvider'])

    correct = 0
    total = 0
    for sub in subsets:
        for ex in task.get(sub, []):
            inp = _grid_to_onehot(ex['input'])
            expected = _grid_to_onehot(ex['output'])
            try:
                out = sess.run(['output'], {'input': inp})[0]
                pred = (out > 0.0).astype(np.float32)
                total += 1
                if np.array_equal(pred, expected):
                    correct += 1
            except Exception:
                total += 1
    return correct, total


def _grid_to_onehot(grid):
    t = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= H: break
        for c, color in enumerate(row):
            if c >= W: break
            if 0 <= color < C:
                t[0][color][r][c] = 1.0
    return t
