"""Shared ONNX building blocks for the DSL→ONNX compiler.

All functions append to (nodes, inits, vinfo) lists and return tensor
names as strings. This keeps maximum control over the graph while
avoiding the verbosity of raw onnx.helper calls in each compiler module.

Naming convention: each function uses a `prefix` parameter so multiple
calls don't collide on tensor names.
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

C, H, W = 10, 30, 30


# ── Helpers ──────────────────────────────────────────────────

def _int64(name: str, vals: list[int]):
    return helper.make_tensor(name, TensorProto.INT64, [len(vals)], vals)


def _float_tensor(name: str, arr: np.ndarray):
    return helper.make_tensor(name, TensorProto.FLOAT,
                               list(arr.shape), arr.flatten().tolist())


def _vi(name: str, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, list(shape))


# ── Grid detection ───────────────────────────────────────────

def detect_grid_extent(nodes, inits, vinfo, prefix="gd"):
    """Detect (h, w) of non-background content in the 30×30 canvas.

    Returns (h_name, w_name): INT64 scalar tensor names.

    Algorithm:
      1. Gather channels 1-9 (skip background channel 0)
      2. ReduceMax across channels + cols → row occupancy (H,)
      3. ReduceMax across channels + rows → col occupancy (W,)
      4. Reverse each, ArgMax → distance from end
      5. h = H - dist_rows, w = W - dist_cols
    """
    p = prefix

    # Non-background channels
    inits.append(_int64(f"{p}_ch", list(range(1, C))))
    nodes.append(helper.make_node("Gather", ["input", f"{p}_ch"], [f"{p}_nonbg"], axis=1))
    vinfo.append(_vi(f"{p}_nonbg", TensorProto.FLOAT, [1, C-1, H, W]))

    # Row occupancy: max over channels and cols
    nodes.append(helper.make_node("ReduceMax", [f"{p}_nonbg"], [f"{p}_rc"],
                                   axes=[1, 3], keepdims=0))
    vinfo.append(_vi(f"{p}_rc", TensorProto.FLOAT, [1, H]))
    nodes.append(helper.make_node("Squeeze", [f"{p}_rc"], [f"{p}_row"], axes=[0]))
    vinfo.append(_vi(f"{p}_row", TensorProto.FLOAT, [H]))

    # Col occupancy: max over channels and rows
    nodes.append(helper.make_node("ReduceMax", [f"{p}_nonbg"], [f"{p}_cc"],
                                   axes=[1, 2], keepdims=0))
    vinfo.append(_vi(f"{p}_cc", TensorProto.FLOAT, [1, W]))
    nodes.append(helper.make_node("Squeeze", [f"{p}_cc"], [f"{p}_col"], axes=[0]))
    vinfo.append(_vi(f"{p}_col", TensorProto.FLOAT, [W]))

    # Reverse + ArgMax for rows
    rev_r = np.arange(H - 1, -1, -1, dtype=np.int64)
    inits.append(_int64(f"{p}_rev_r", rev_r.tolist()))
    nodes.append(helper.make_node("Gather", [f"{p}_row", f"{p}_rev_r"],
                                   [f"{p}_row_rev"], axis=0))
    vinfo.append(_vi(f"{p}_row_rev", TensorProto.FLOAT, [H]))
    nodes.append(helper.make_node("ArgMax", [f"{p}_row_rev"], [f"{p}_dr"],
                                   axis=0, keepdims=0))
    vinfo.append(_vi(f"{p}_dr", TensorProto.INT64, []))

    # Reverse + ArgMax for cols
    rev_c = np.arange(W - 1, -1, -1, dtype=np.int64)
    inits.append(_int64(f"{p}_rev_c", rev_c.tolist()))
    nodes.append(helper.make_node("Gather", [f"{p}_col", f"{p}_rev_c"],
                                   [f"{p}_col_rev"], axis=0))
    vinfo.append(_vi(f"{p}_col_rev", TensorProto.FLOAT, [W]))
    nodes.append(helper.make_node("ArgMax", [f"{p}_col_rev"], [f"{p}_dc"],
                                   axis=0, keepdims=0))
    vinfo.append(_vi(f"{p}_dc", TensorProto.INT64, []))

    # h = H - dist_rows, w = W - dist_cols
    inits.append(_int64(f"{p}_H", [H]))
    inits.append(_int64(f"{p}_W", [W]))
    nodes.append(helper.make_node("Sub", [f"{p}_H", f"{p}_dr"], [f"{p}_h"]))
    vinfo.append(_vi(f"{p}_h", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", [f"{p}_W", f"{p}_dc"], [f"{p}_w"]))
    vinfo.append(_vi(f"{p}_w", TensorProto.INT64, [1]))

    return f"{p}_h", f"{p}_w"


def detect_min_position(nodes, inits, vinfo, prefix="mp"):
    """Detect (min_row, min_col) of first non-background pixel.

    Returns (min_row_name, min_col_name): INT64 scalar tensor names.
    """
    p = prefix

    # Reuse non-bg detection
    inits.append(_int64(f"{p}_ch", list(range(1, C))))
    nodes.append(helper.make_node("Gather", ["input", f"{p}_ch"], [f"{p}_nonbg"], axis=1))
    vinfo.append(_vi(f"{p}_nonbg", TensorProto.FLOAT, [1, C-1, H, W]))

    # Row occupancy
    nodes.append(helper.make_node("ReduceMax", [f"{p}_nonbg"], [f"{p}_rc"],
                                   axes=[1, 3], keepdims=0))
    vinfo.append(_vi(f"{p}_rc", TensorProto.FLOAT, [1, H]))
    nodes.append(helper.make_node("Squeeze", [f"{p}_rc"], [f"{p}_row"], axes=[0]))
    vinfo.append(_vi(f"{p}_row", TensorProto.FLOAT, [H]))
    nodes.append(helper.make_node("ArgMax", [f"{p}_row"], [f"{p}_min_r"],
                                   axis=0, keepdims=0))
    vinfo.append(_vi(f"{p}_min_r", TensorProto.INT64, []))

    # Col occupancy
    nodes.append(helper.make_node("ReduceMax", [f"{p}_nonbg"], [f"{p}_cc"],
                                   axes=[1, 2], keepdims=0))
    vinfo.append(_vi(f"{p}_cc", TensorProto.FLOAT, [1, W]))
    nodes.append(helper.make_node("Squeeze", [f"{p}_cc"], [f"{p}_col"], axes=[0]))
    vinfo.append(_vi(f"{p}_col", TensorProto.FLOAT, [W]))
    nodes.append(helper.make_node("ArgMax", [f"{p}_col"], [f"{p}_min_c"],
                                   axis=0, keepdims=0))
    vinfo.append(_vi(f"{p}_min_c", TensorProto.INT64, []))

    return f"{p}_min_r", f"{p}_min_c"


# ── Index grids ──────────────────────────────────────────────

def build_row_col_grids(inits, prefix="rc"):
    """Create constant INT64 grids of shape (H*W,).

    row_grid = [0,0,...,0, 1,1,...,1, ..., 29,...,29]
    col_grid = [0,1,...,29, 0,1,...,29, ...]

    Returns (row_name, col_name).
    """
    row = np.repeat(np.arange(H, dtype=np.int64), W)
    col = np.tile(np.arange(W, dtype=np.int64), H)
    inits.append(_int64(f"{prefix}_row", row.tolist()))
    inits.append(_int64(f"{prefix}_col", col.tolist()))
    return f"{prefix}_row", f"{prefix}_col"


# ── INT64 clamping ───────────────────────────────────────────

def clamp_int(nodes, inits, vinfo, x_name, min_val, max_val, prefix="cl"):
    """Clamp INT64 tensor to [min_val, max_val] via Cast→Clip→Cast."""
    p = prefix
    nodes.append(helper.make_node("Cast", [x_name], [f"{p}_f"], to=TensorProto.FLOAT))
    vinfo.append(_vi(f"{p}_f", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Clip", [f"{p}_f"], [f"{p}_cf"],
                                   min=float(min_val), max=float(max_val)))
    vinfo.append(_vi(f"{p}_cf", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Cast", [f"{p}_cf"], [f"{p}_i"], to=TensorProto.INT64))
    vinfo.append(_vi(f"{p}_i", TensorProto.INT64, [H * W]))
    return f"{p}_i"


# ── Dynamic masking ──────────────────────────────────────────

def build_mask_from_diff(nodes, inits, vinfo, diff_int_name, prefix="mk"):
    """Build float mask: 1.0 where diff >= 0, 0.0 where diff < 0.

    diff is an INT64 tensor. We use: Clip(Cast(diff) + 1, 0, 1).
    When diff >= 0: cast+1 >= 1, clip to 1.0.
    When diff < 0: cast+1 <= 0, clip to 0.0.
    When diff = -1: cast+1 = 0, clip to 0.0. ← edge case handled correctly.
    """
    p = prefix
    nodes.append(helper.make_node("Cast", [diff_int_name], [f"{p}_f"],
                                   to=TensorProto.FLOAT))
    vinfo.append(_vi(f"{p}_f", TensorProto.FLOAT, [H * W]))

    ones = np.ones(H * W, dtype=np.float32)
    inits.append(helper.make_tensor(f"{p}_ones", TensorProto.FLOAT, [H * W], ones.tolist()))
    nodes.append(helper.make_node("Add", [f"{p}_f", f"{p}_ones"], [f"{p}_shifted"]))
    vinfo.append(_vi(f"{p}_shifted", TensorProto.FLOAT, [H * W]))

    nodes.append(helper.make_node("Clip", [f"{p}_shifted"], [f"{p}_mask"],
                                   min=0.0, max=1.0))
    vinfo.append(_vi(f"{p}_mask", TensorProto.FLOAT, [H * W]))
    return f"{p}_mask"


# ── Flatten → Gather → Reshape ───────────────────────────────

def flatten_gather_reshape(nodes, inits, vinfo, indices_name, prefix="fgr"):
    """Flatten input to (1,C,H*W), Gather with indices, reshape to (1,C,H,W)."""
    p = prefix
    inits.append(_int64(f"{p}_s1", [1, C, H * W]))
    nodes.append(helper.make_node("Reshape", ["input", f"{p}_s1"], [f"{p}_flat"]))
    vinfo.append(_vi(f"{p}_flat", TensorProto.FLOAT, [1, C, H * W]))

    nodes.append(helper.make_node("Gather", [f"{p}_flat", indices_name],
                                   [f"{p}_gathered"], axis=2))
    vinfo.append(_vi(f"{p}_gathered", TensorProto.FLOAT, [1, C, H * W]))

    inits.append(_int64(f"{p}_s2", [1, C, H, W]))
    nodes.append(helper.make_node("Reshape", [f"{p}_gathered", f"{p}_s2"],
                                   [f"{p}_out"]))
    vinfo.append(_vi(f"{p}_out", TensorProto.FLOAT, [1, C, H, W]))
    return f"{p}_out"


# ── Apply spatial mask ───────────────────────────────────────

def apply_mask(nodes, inits, vinfo, data_name, mask_flat_name, prefix="am"):
    """Multiply data (1,C,H,W) by mask (H*W,) broadcast over channels."""
    p = prefix
    inits.append(_int64(f"{p}_ms", [1, 1, H, W]))
    nodes.append(helper.make_node("Reshape", [mask_flat_name, f"{p}_ms"],
                                   [f"{p}_m4d"]))
    vinfo.append(_vi(f"{p}_m4d", TensorProto.FLOAT, [1, 1, H, W]))
    nodes.append(helper.make_node("Mul", [data_name, f"{p}_m4d"], [f"{p}_out"]))
    vinfo.append(_vi(f"{p}_out", TensorProto.FLOAT, [1, C, H, W]))
    return f"{p}_out"


# ── Model assembly ───────────────────────────────────────────

def make_model(nodes, inits, vinfo, output_name="output"):
    """Wrap nodes/inits/vinfo into a complete ModelProto."""
    # Add Identity to rename final tensor to "output" if needed
    if output_name != "output":
        nodes.append(helper.make_node("Identity", [output_name], ["output"]))

    in_info = _vi("input", TensorProto.FLOAT, [1, C, H, W])
    out_info = _vi("output", TensorProto.FLOAT, [1, C, H, W])
    graph = helper.make_graph(nodes, "compiled", [in_info], [out_info],
                               initializer=inits, value_info=vinfo)
    return helper.make_model(graph, ir_version=10,
                              opset_imports=[helper.make_opsetid("", 10)])
