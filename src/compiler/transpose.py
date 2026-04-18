"""Transpose compiler — swap rows and columns.

Test cases:
  task179: grid.T (transpose)
  task241: grid.T (transpose)

Technique:
  1. Detect grid extent (h, w)
  2. Compute transposed indices: src[r,c] = input[c,r]
     flat_idx = col * W + row  (swap row↔col in addressing)
  3. Clamp to valid range
  4. Flatten → Gather → Reshape
  5. Dynamic mask: zero out positions beyond (w, h) — note swapped dims
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


def compile_transpose() -> "onnx.ModelProto":
    """Transpose: swap rows and columns within detected grid.

    For a grid of size (h, w), output is size (w, h) — rows become cols.
    Since canvas is always 30×30, this means the content region changes shape.
    """
    nodes = []
    inits = []
    vinfo = []

    # Step 1: Detect grid extent
    h_name, w_name = detect_grid_extent(nodes, inits, vinfo, prefix="gd")

    # Step 2: Build row/col grids (output coordinates)
    row_name, col_name = build_row_col_grids(inits, prefix="rc")

    # Step 3: Compute transposed source indices
    # For output[r, c], we read from input[c, r]
    # flat_src = col * W + row  (col is the source row, row is the source col)
    # But we need to handle the canvas: only valid within (w, h) output region
    # (output row < w AND output col < h)

    w_vec = np.full(H * W, W, dtype=np.int64)
    inits.append(_int64("W_vec", w_vec.tolist()))

    # src_row = col (reading from input row = output col)
    # src_col = row (reading from input col = output row)
    # flat_src = src_row * W + src_col = col * W + row
    nodes.append(helper.make_node("Mul", [col_name, "W_vec"], ["src_row_off"]))
    vinfo.append(_vi("src_row_off", TensorProto.INT64, [H * W]))
    nodes.append(helper.make_node("Add", ["src_row_off", row_name], ["src_flat_raw"]))
    vinfo.append(_vi("src_flat_raw", TensorProto.INT64, [H * W]))

    # Clamp to valid range [0, H*W-1]
    max_idx = H * W - 1
    nodes.append(helper.make_node("Cast", ["src_flat_raw"], ["src_f"],
                                   to=TensorProto.FLOAT))
    vinfo.append(_vi("src_f", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Clip", ["src_f"], ["src_cf"],
                                   min=0.0, max=float(max_idx)))
    vinfo.append(_vi("src_cf", TensorProto.FLOAT, [H * W]))
    nodes.append(helper.make_node("Cast", ["src_cf"], ["src_flat"],
                                   to=TensorProto.INT64))
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
    # Output is valid where: row < w AND col < h (transposed dimensions)
    # row_mask: w - 1 - row >= 0
    # col_mask: h - 1 - col >= 0
    inits.append(_int64("one", [1]))

    # Row mask (output row must be < w, the original width)
    nodes.append(helper.make_node("Sub", [w_name, "one"], ["w_m1"]))
    vinfo.append(_vi("w_m1", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", ["w_m1", row_name], ["rdiff"]))
    vinfo.append(_vi("rdiff", TensorProto.INT64, [H * W]))

    row_mask = build_mask_from_diff(nodes, inits, vinfo, "rdiff", prefix="rmk")

    # Col mask (output col must be < h, the original height)
    nodes.append(helper.make_node("Sub", [h_name, "one"], ["h_m1"]))
    vinfo.append(_vi("h_m1", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", ["h_m1", col_name], ["cdiff"]))
    vinfo.append(_vi("cdiff", TensorProto.INT64, [H * W]))

    col_mask = build_mask_from_diff(nodes, inits, vinfo, "cdiff", prefix="cmk")

    # Combined mask
    nodes.append(helper.make_node("Mul", [row_mask, col_mask], ["mask_flat"]))
    vinfo.append(_vi("mask_flat", TensorProto.FLOAT, [H * W]))

    # Apply mask
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

    model = compile_transpose()

    for tn in [179, 241]:
        try:
            with open(f"task{tn:03d}.json") as f:
                task = json.load(f)
            s = score_model(model)
            correct, total = verify_model(model, task)
            status = "PASS" if correct == total and total > 0 else "FAIL"
            print(f"task{tn:03d}: cost={s['cost']} score={s['score']:.3f} "
                  f"verified={correct}/{total} {status}")
        except Exception as e:
            print(f"task{tn:03d}: ERROR {e}")
