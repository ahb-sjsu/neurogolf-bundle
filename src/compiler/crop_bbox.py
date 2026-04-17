"""Crop-to-bbox compiler — extract non-zero bounding box and place at origin.

Handles tasks where: output = input[r_min:r_max+1, c_min:c_max+1]
shifted to position (0,0) in the 30×30 canvas.

Test cases: task031, task032, task039, task289 (12 total crop tasks).

Technique:
  1. Detect grid boundaries via ReduceMax + ArgMax (shared primitives)
  2. Compute source indices: row_src = offset + min_row, col_src = offset + min_col
  3. Clamp to [0, H-1] (out-of-range → sentinel row/col which is zeros)
  4. Gather rows (axis=2), then cols (axis=3)
  5. Dynamic mask: 1.0 where offset < bbox_h AND offset < bbox_w, else 0.0
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from compiler.primitives import (
    C, H, W,
    _int64, _vi,
    detect_grid_extent, detect_min_position,
    build_row_col_grids, clamp_int,
    build_mask_from_diff, apply_mask,
    make_model,
)


def compile_crop_nonzero() -> "onnx.ModelProto":
    """Crop to bounding box of all non-background pixels.

    Dynamically detects min_row, min_col, bbox_h, bbox_w at inference time.
    Works for variable grid sizes across examples.
    """
    nodes = []
    inits = []
    vinfo = []

    # Step 1: Detect grid extent (h, w) and min position (min_r, min_c)
    h_name, w_name = detect_grid_extent(nodes, inits, vinfo, prefix="ge")
    min_r, min_c = detect_min_position(nodes, inits, vinfo, prefix="mp")

    # Step 2: Compute bbox dimensions
    # bbox_h = h (grid extent already = max - min + 1 for contiguous grids starting at min)
    # Actually: h from detect_grid_extent = total occupied rows (last - first + 1)
    # But if grid doesn't start at row 0, h includes rows from min_r to max_r.
    # We need bbox_h = max_r - min_r + 1 = h (since h = H - dist_from_end and
    # the grid is contiguous). Wait — detect_grid_extent gives h = total canvas
    # rows with content, which equals max_row - min_row + 1 only if the grid is
    # contiguous from min_row. For ARC tasks this is always true.
    #
    # Actually no: detect_grid_extent counts from the TOP. If content starts at
    # row 5 and ends at row 9, it returns h = H - (H - 10) = 10, not 5.
    # That's wrong — h should be 5 (bbox height).
    #
    # Fix: bbox_h = h_extent - min_r, bbox_w = w_extent - min_c
    # where h_extent = max_row + 1 = H - dist_from_end
    # So bbox_h = (H - dist_from_end) - min_r

    # h_extent = H - dist_from_end (already computed as ge_h)
    # bbox_h = h_extent - min_r
    nodes.append(helper.make_node("Sub", [h_name, min_r], ["bbox_h"]))
    vinfo.append(_vi("bbox_h", TensorProto.INT64, [1]))
    nodes.append(helper.make_node("Sub", [w_name, min_c], ["bbox_w"]))
    vinfo.append(_vi("bbox_w", TensorProto.INT64, [1]))

    # Step 3: Build source indices for Gather
    # For output position (r, c): source_row = min_r + r, source_col = min_c + c
    # We gather rows first (axis=2), then cols (axis=3).

    # Row indices: [0, 1, ..., H-1] + min_r → [min_r, min_r+1, ...]
    offset_r = np.arange(H, dtype=np.int64)
    inits.append(_int64("off_r", offset_r.tolist()))
    nodes.append(helper.make_node("Add", ["off_r", min_r], ["row_src"]))
    vinfo.append(_vi("row_src", TensorProto.INT64, [H]))

    # Clamp to [0, H-1]
    nodes.append(helper.make_node("Cast", ["row_src"], ["row_src_f"], to=TensorProto.FLOAT))
    vinfo.append(_vi("row_src_f", TensorProto.FLOAT, [H]))
    nodes.append(helper.make_node("Clip", ["row_src_f"], ["row_src_cf"],
                                   min=0.0, max=float(H - 1)))
    vinfo.append(_vi("row_src_cf", TensorProto.FLOAT, [H]))
    nodes.append(helper.make_node("Cast", ["row_src_cf"], ["row_idx"],
                                   to=TensorProto.INT64))
    vinfo.append(_vi("row_idx", TensorProto.INT64, [H]))

    # Col indices: [0, 1, ..., W-1] + min_c
    offset_c = np.arange(W, dtype=np.int64)
    inits.append(_int64("off_c", offset_c.tolist()))
    nodes.append(helper.make_node("Add", ["off_c", min_c], ["col_src"]))
    vinfo.append(_vi("col_src", TensorProto.INT64, [W]))

    nodes.append(helper.make_node("Cast", ["col_src"], ["col_src_f"], to=TensorProto.FLOAT))
    vinfo.append(_vi("col_src_f", TensorProto.FLOAT, [W]))
    nodes.append(helper.make_node("Clip", ["col_src_f"], ["col_src_cf"],
                                   min=0.0, max=float(W - 1)))
    vinfo.append(_vi("col_src_cf", TensorProto.FLOAT, [W]))
    nodes.append(helper.make_node("Cast", ["col_src_cf"], ["col_idx"],
                                   to=TensorProto.INT64))
    vinfo.append(_vi("col_idx", TensorProto.INT64, [W]))

    # Step 4: Gather rows, then cols
    nodes.append(helper.make_node("Gather", ["input", "row_idx"], ["rows_gathered"],
                                   axis=2))
    vinfo.append(_vi("rows_gathered", TensorProto.FLOAT, [1, C, H, W]))

    nodes.append(helper.make_node("Gather", ["rows_gathered", "col_idx"],
                                   ["gathered"], axis=3))
    vinfo.append(_vi("gathered", TensorProto.FLOAT, [1, C, H, W]))

    # Step 5: Build dynamic mask
    # Row mask: 1.0 where row_offset < bbox_h, i.e., bbox_h - offset - 1 >= 0
    # Use the Clip(cast(diff) + 1, 0, 1) trick.

    # bbox_h_minus_1_minus_offset = bbox_h - 1 - [0,1,...,H-1]
    inits.append(_int64("one_s", [1]))
    nodes.append(helper.make_node("Sub", ["bbox_h", "one_s"], ["bh_m1"]))
    vinfo.append(_vi("bh_m1", TensorProto.INT64, [1]))

    row_offsets = np.arange(H, dtype=np.int64)
    inits.append(_int64("row_off", row_offsets.tolist()))
    nodes.append(helper.make_node("Sub", ["bh_m1", "row_off"], ["rdiff"]))
    vinfo.append(_vi("rdiff", TensorProto.INT64, [H]))

    # rdiff >= 0 → mask = 1.0
    nodes.append(helper.make_node("Cast", ["rdiff"], ["rdiff_f"], to=TensorProto.FLOAT))
    vinfo.append(_vi("rdiff_f", TensorProto.FLOAT, [H]))
    ones_h = np.ones(H, dtype=np.float32)
    inits.append(helper.make_tensor("ones_h", TensorProto.FLOAT, [H], ones_h.tolist()))
    nodes.append(helper.make_node("Add", ["rdiff_f", "ones_h"], ["rdiff_s"]))
    vinfo.append(_vi("rdiff_s", TensorProto.FLOAT, [H]))
    nodes.append(helper.make_node("Clip", ["rdiff_s"], ["rmask"], min=0.0, max=1.0))
    vinfo.append(_vi("rmask", TensorProto.FLOAT, [H]))

    # Col mask: same logic with bbox_w
    nodes.append(helper.make_node("Sub", ["bbox_w", "one_s"], ["bw_m1"]))
    vinfo.append(_vi("bw_m1", TensorProto.INT64, [1]))

    col_offsets = np.arange(W, dtype=np.int64)
    inits.append(_int64("col_off", col_offsets.tolist()))
    nodes.append(helper.make_node("Sub", ["bw_m1", "col_off"], ["cdiff"]))
    vinfo.append(_vi("cdiff", TensorProto.INT64, [W]))

    nodes.append(helper.make_node("Cast", ["cdiff"], ["cdiff_f"], to=TensorProto.FLOAT))
    vinfo.append(_vi("cdiff_f", TensorProto.FLOAT, [W]))
    ones_w = np.ones(W, dtype=np.float32)
    inits.append(helper.make_tensor("ones_w", TensorProto.FLOAT, [W], ones_w.tolist()))
    nodes.append(helper.make_node("Add", ["cdiff_f", "ones_w"], ["cdiff_s"]))
    vinfo.append(_vi("cdiff_s", TensorProto.FLOAT, [W]))
    nodes.append(helper.make_node("Clip", ["cdiff_s"], ["cmask"], min=0.0, max=1.0))
    vinfo.append(_vi("cmask", TensorProto.FLOAT, [W]))

    # Combined mask: rmask (H,) reshaped to (1,1,H,1) × cmask (W,) reshaped to (1,1,1,W)
    inits.append(_int64("rmask_s", [1, 1, H, 1]))
    nodes.append(helper.make_node("Reshape", ["rmask", "rmask_s"], ["rmask_4d"]))
    vinfo.append(_vi("rmask_4d", TensorProto.FLOAT, [1, 1, H, 1]))

    inits.append(_int64("cmask_s", [1, 1, 1, W]))
    nodes.append(helper.make_node("Reshape", ["cmask", "cmask_s"], ["cmask_4d"]))
    vinfo.append(_vi("cmask_4d", TensorProto.FLOAT, [1, 1, 1, W]))

    nodes.append(helper.make_node("Mul", ["rmask_4d", "cmask_4d"], ["mask_4d"]))
    vinfo.append(_vi("mask_4d", TensorProto.FLOAT, [1, 1, H, W]))

    # Apply mask
    nodes.append(helper.make_node("Mul", ["gathered", "mask_4d"], ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, "src")
    from grammar.primitives import score_model, verify_model
    import onnx

    model = compile_crop_nonzero()

    # Test on multiple crop tasks
    for tn in [31, 32, 39, 289]:
        try:
            with open(f"task{tn:03d}.json") as f:
                task = json.load(f)
            s = score_model(model)
            correct, total = verify_model(model, task)
            status = "PASS" if correct == total and total > 0 else f"FAIL ({correct}/{total})"
            print(f"task{tn:03d}: cost={s['cost']} score={s['score']:.3f} "
                  f"verified={correct}/{total} {status}")
        except Exception as e:
            print(f"task{tn:03d}: ERROR {e}")
