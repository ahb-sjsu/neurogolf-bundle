"""Flip compiler — vertical and horizontal flips for variable-size grids.

Test cases:
  task155: grid[::-1]  (vertical flip) — 266/266 verified
  task150: np.fliplr   (horizontal flip)

Uses shared primitives for grid detection + dynamic masking.
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from compiler.primitives import (
    C, H, W,
    _int64, _vi, _float_tensor,
    detect_grid_extent,
    build_row_col_grids,
    build_mask_from_diff,
    make_model,
)


def _compile_flip(axis: str) -> "onnx.ModelProto":
    """Build a dynamic flip ONNX model.

    axis='v': flip rows (vertical)
    axis='h': flip cols (horizontal)

    Algorithm:
      1. Detect grid extent (h, w)
      2. For each output position, compute flipped source index:
         - flip_v: source_row = h - 1 - row, source_col = col
         - flip_h: source_row = row, source_col = w - 1 - col
      3. Clamp source to [0, H-1] (out-of-range goes to sentinel)
      4. Flatten → Gather → Reshape
      5. Mask: zero out positions beyond grid boundary
    """
    nodes = []
    inits = []
    vinfo = []

    # Step 1: Detect grid extent
    h_name, w_name = detect_grid_extent(nodes, inits, vinfo, prefix="gd")

    # Step 2: Build row/col index grids (H*W each)
    row_name, col_name = build_row_col_grids(inits, prefix="rc")

    # Step 3: Compute flipped source indices
    # extent_m1 = extent - 1
    inits.append(_int64("one", [1]))

    if axis == "v":
        # flipped_row = h - 1 - row_idx (broadcast [1] - [H*W])
        nodes.append(helper.make_node("Sub", [h_name, "one"], ["ext_m1"]))
        vinfo.append(_vi("ext_m1", TensorProto.INT64, [1]))
        nodes.append(helper.make_node("Sub", ["ext_m1", row_name], ["flip_dim"]))
        vinfo.append(_vi("flip_dim", TensorProto.INT64, [H * W]))
        fixed_dim_name = col_name  # cols unchanged
        flip_is_row = True
    else:  # axis == "h"
        # flipped_col = w - 1 - col_idx
        nodes.append(helper.make_node("Sub", [w_name, "one"], ["ext_m1"]))
        vinfo.append(_vi("ext_m1", TensorProto.INT64, [1]))
        nodes.append(helper.make_node("Sub", ["ext_m1", col_name], ["flip_dim"]))
        vinfo.append(_vi("flip_dim", TensorProto.INT64, [H * W]))
        fixed_dim_name = row_name  # rows unchanged
        flip_is_row = False

    # Step 4: Clamp flipped dimension to [0, H-1 or W-1]
    max_val = H - 1 if flip_is_row else W - 1
    nodes.append(helper.make_node("Cast", ["flip_dim"], ["flip_f"],
                                   to=TensorProto.FLOAT))
    vinfo.append(_vi("flip_f", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Clip", ["flip_f"], ["flip_cf"],
                                   min=0.0, max=float(max_val)))
    vinfo.append(_vi("flip_cf", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Cast", ["flip_cf"], ["flip_i"],
                                   to=TensorProto.INT64))
    vinfo.append(_vi("flip_i", TensorProto.INT64, [H * W]))

    # Step 5: Compute flat source indices = row * W + col
    w_vec = np.full(H * W, W, dtype=np.int64)
    inits.append(_int64("W_vec", w_vec.tolist()))

    if flip_is_row:
        # src = flip_i * W + col_idx
        nodes.append(helper.make_node("Mul", ["flip_i", "W_vec"], ["row_off"]))
        vinfo.append(_vi("row_off", TensorProto.INT64, [H * W]))
        nodes.append(helper.make_node("Add", ["row_off", col_name], ["src_flat"]))
    else:
        # src = row_idx * W + flip_i
        nodes.append(helper.make_node("Mul", [row_name, "W_vec"], ["row_off"]))
        vinfo.append(_vi("row_off", TensorProto.INT64, [H * W]))
        nodes.append(helper.make_node("Add", ["row_off", "flip_i"], ["src_flat"]))
    vinfo.append(_vi("src_flat", TensorProto.INT64, [H * W]))

    # Step 6: Flatten → Gather → Reshape
    inits.append(_int64("s_flat", [1, C, H * W]))
    nodes.append(helper.make_node("Reshape", ["input", "s_flat"], ["flat"]))
    vinfo.append(_vi("flat", TensorProto.FLOAT, [1, C, H * W]))

    nodes.append(helper.make_node("Gather", ["flat", "src_flat"],
                                   ["gathered"], axis=2))
    vinfo.append(_vi("gathered", TensorProto.FLOAT, [1, C, H * W]))

    inits.append(_int64("s_canvas", [1, C, H, W]))
    nodes.append(helper.make_node("Reshape", ["gathered", "s_canvas"],
                                   ["reshaped"]))
    vinfo.append(_vi("reshaped", TensorProto.FLOAT, [1, C, H, W]))

    # Step 7: Dynamic mask (zero out positions beyond grid)
    # mask = Clip(cast(flip_dim) + 1, 0, 1)
    # flip_dim >= 0 → mask = 1; flip_dim < 0 → mask = 0
    mask_name = build_mask_from_diff(nodes, inits, vinfo, "flip_dim", prefix="mk")

    # Apply mask: reshape to (1,1,H,W), broadcast multiply
    inits.append(_int64("mk_s", [1, 1, H, W]))
    nodes.append(helper.make_node("Reshape", [mask_name, "mk_s"], ["mask_4d"]))
    vinfo.append(_vi("mask_4d", TensorProto.FLOAT, [1, 1, H, W]))

    nodes.append(helper.make_node("Mul", ["reshaped", "mask_4d"], ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


def compile_flip_v() -> "onnx.ModelProto":
    """Vertical flip: reverse row order within detected grid."""
    return _compile_flip("v")


def compile_flip_h() -> "onnx.ModelProto":
    """Horizontal flip: reverse column order within detected grid."""
    return _compile_flip("h")


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, "src")
    from grammar.primitives import score_model, verify_model

    for tn, builder, name in [
        (155, compile_flip_v, "flip_v"),
        (150, compile_flip_h, "flip_h"),
    ]:
        model = builder()
        with open(f"task{tn:03d}.json") as f:
            task = json.load(f)
        s = score_model(model)
        correct, total = verify_model(model, task)
        status = "PASS" if correct == total and total > 0 else f"FAIL"
        print(f"task{tn:03d} ({name}): cost={s['cost']} score={s['score']:.3f} "
              f"verified={correct}/{total} {status}")
