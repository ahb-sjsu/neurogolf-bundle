"""Rotate compiler — 90/180/270 degree rotations.

Test cases:
  task380: rotate 90 degrees CCW (already solved by conv, but cheaper here)

Technique (rotate 90 CCW = transpose + flip_v):
  1. Detect grid extent (h, w)
  2. Compute rotated indices: output[r,c] = input[c, h-1-r]
     source_row = col, source_col = h - 1 - row
  3. Clamp, Gather, Mask

Rotate 180 = flip_v + flip_h (two flips)
Rotate 270 CCW = transpose + flip_h
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from compiler.primitives import (
    C, H, W,
    _int64, _vi,
    detect_grid_extent,
    build_row_col_grids,
    build_mask_from_diff,
    make_model,
)


def compile_rotate90() -> "onnx.ModelProto":
    """Rotate 90 degrees counter-clockwise.

    For a grid of size (h, w), output is size (w, h).
    output[r, c] = input[c, h-1-r]
    """
    nodes = []
    inits = []
    vinfo = []

    # Step 1: Detect grid extent
    h_name, w_name = detect_grid_extent(nodes, inits, vinfo, prefix="gd")

    # Step 2: Build row/col grids
    row_name, col_name = build_row_col_grids(inits, prefix="rc")

    # Step 3: Compute rotated source indices
    # source_row = col (output col maps to source row)
    # source_col = h - 1 - row (output row maps to source col, reversed)
    inits.append(_int64("one", [1]))

    # h - 1 - row
    nodes.append(helper.make_node("Sub", [h_name, "one"], ["h_m1"]))
    vinfo.append(_vi("h_m1", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", ["h_m1", row_name], ["src_col_raw"]))
    vinfo.append(_vi("src_col_raw", TensorProto.INT64, [H * W]))

    # Clamp source_col to [0, W-1]
    nodes.append(helper.make_node("Cast", ["src_col_raw"], ["sc_f"],
                                   to=TensorProto.FLOAT))
    vinfo.append(_vi("sc_f", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Clip", ["sc_f"], ["sc_cf"],
                                   min=0.0, max=float(W - 1)))
    vinfo.append(_vi("sc_cf", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Cast", ["sc_cf"], ["src_col"],
                                   to=TensorProto.INT64))
    vinfo.append(_vi("src_col", TensorProto.INT64, [H * W]))

    # flat_src = col * W + src_col (col is the source row)
    w_vec = np.full(H * W, W, dtype=np.int64)
    inits.append(_int64("W_vec", w_vec.tolist()))
    nodes.append(helper.make_node("Mul", [col_name, "W_vec"], ["sr_off"]))
    vinfo.append(_vi("sr_off", TensorProto.INT64, [H * W]))
    nodes.append(helper.make_node("Add", ["sr_off", "src_col"], ["src_flat"]))
    vinfo.append(_vi("src_flat", TensorProto.INT64, [H * W]))

    # Step 4: Flatten → Gather → Reshape
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

    # Step 5: Dynamic mask
    # Output valid where: row < w AND col < h (rotated dims)
    # row_mask: w - 1 - row >= 0
    nodes.append(helper.make_node("Sub", [w_name, "one"], ["w_m1"]))
    vinfo.append(_vi("w_m1", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", ["w_m1", row_name], ["rdiff"]))
    vinfo.append(_vi("rdiff", TensorProto.INT64, [H * W]))
    row_mask = build_mask_from_diff(nodes, inits, vinfo, "rdiff", prefix="rmk")

    # col_mask: h - 1 - col >= 0
    nodes.append(helper.make_node("Sub", ["h_m1", col_name], ["cdiff"]))
    vinfo.append(_vi("cdiff", TensorProto.INT64, [H * W]))
    col_mask = build_mask_from_diff(nodes, inits, vinfo, "cdiff", prefix="cmk")

    # Combined mask
    nodes.append(helper.make_node("Mul", [row_mask, col_mask], ["mask_flat"]))
    vinfo.append(_vi("mask_flat", TensorProto.FLOAT, [H * W]))

    inits.append(_int64("mk_s", [1, 1, H, W]))
    nodes.append(helper.make_node("Reshape", ["mask_flat", "mk_s"], ["mask_4d"]))
    vinfo.append(_vi("mask_4d", TensorProto.FLOAT, [1, 1, H, W]))

    nodes.append(helper.make_node("Mul", ["reshaped", "mask_4d"], ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


def compile_rotate180() -> "onnx.ModelProto":
    """Rotate 180 degrees = flip both axes.

    output[r, c] = input[h-1-r, w-1-c]
    """
    nodes = []
    inits = []
    vinfo = []

    h_name, w_name = detect_grid_extent(nodes, inits, vinfo, prefix="gd")
    row_name, col_name = build_row_col_grids(inits, prefix="rc")

    inits.append(_int64("one", [1]))

    # src_row = h - 1 - row
    nodes.append(helper.make_node("Sub", [h_name, "one"], ["h_m1"]))
    vinfo.append(_vi("h_m1", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", ["h_m1", row_name], ["src_row"]))
    vinfo.append(_vi("src_row", TensorProto.INT64, [H * W]))

    # src_col = w - 1 - col
    nodes.append(helper.make_node("Sub", [w_name, "one"], ["w_m1"]))
    vinfo.append(_vi("w_m1", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", ["w_m1", col_name], ["src_col"]))
    vinfo.append(_vi("src_col", TensorProto.INT64, [H * W]))

    # Clamp both
    for name, dim_name, max_v, suf in [("src_row", "sr", H-1, "r"), ("src_col", "sc", W-1, "c")]:
        nodes.append(helper.make_node("Cast", [name], [f"{suf}_f"],
                                       to=TensorProto.FLOAT))
        vinfo.append(_vi(f"{suf}_f", TensorProto.FLOAT, [H * W]))
        nodes.append(helper.make_node("Clip", [f"{suf}_f"], [f"{suf}_cf"],
                                       min=0.0, max=float(max_v)))
        vinfo.append(_vi(f"{suf}_cf", TensorProto.FLOAT, [H * W]))
        nodes.append(helper.make_node("Cast", [f"{suf}_cf"], [f"{suf}_i"],
                                       to=TensorProto.INT64))
        vinfo.append(_vi(f"{suf}_i", TensorProto.INT64, [H * W]))

    # flat = sr_i * W + sc_i
    w_vec = np.full(H * W, W, dtype=np.int64)
    inits.append(_int64("W_vec", w_vec.tolist()))
    nodes.append(helper.make_node("Mul", ["sr_i", "W_vec"], ["row_off"]))
    vinfo.append(_vi("row_off", TensorProto.INT64, [H * W]))
    nodes.append(helper.make_node("Add", ["row_off", "sc_i"], ["src_flat"]))
    vinfo.append(_vi("src_flat", TensorProto.INT64, [H * W]))

    # Flatten → Gather → Reshape
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

    # Mask: row < h AND col < w
    nodes.append(helper.make_node("Sub", ["h_m1", row_name], ["rdiff"]))
    vinfo.append(_vi("rdiff", TensorProto.INT64, [H * W]))
    row_mask = build_mask_from_diff(nodes, inits, vinfo, "rdiff", prefix="rmk")

    nodes.append(helper.make_node("Sub", ["w_m1", col_name], ["cdiff"]))
    vinfo.append(_vi("cdiff", TensorProto.INT64, [H * W]))
    col_mask = build_mask_from_diff(nodes, inits, vinfo, "cdiff", prefix="cmk")

    nodes.append(helper.make_node("Mul", [row_mask, col_mask], ["mask_flat"]))
    vinfo.append(_vi("mask_flat", TensorProto.FLOAT, [H * W]))

    inits.append(_int64("mk_s", [1, 1, H, W]))
    nodes.append(helper.make_node("Reshape", ["mask_flat", "mk_s"], ["mask_4d"]))
    vinfo.append(_vi("mask_4d", TensorProto.FLOAT, [1, 1, H, W]))

    nodes.append(helper.make_node("Mul", ["reshaped", "mask_4d"], ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, "src")
    from grammar.primitives import score_model, verify_model

    for tn in range(1, 401):
        try:
            with open(f"task{tn:03d}.json") as f:
                task = json.load(f)
            for builder, name in [
                (compile_rotate90, "rot90"),
                (compile_rotate180, "rot180"),
            ]:
                model = builder()
                correct, total = verify_model(model, task)
                if correct == total and total > 0:
                    s = score_model(model)
                    print(f"task{tn:03d} ({name}): cost={s['cost']} "
                          f"verified={correct}/{total} PASS")
        except Exception:
            pass
