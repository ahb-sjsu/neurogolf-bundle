"""Tile compiler — replicate grid content via Concat.

Test cases:
  task256: horizontal double (hstack)
  task001: Kronecker self-tile (tile where non-zero)

For fixed grid sizes: Slice to extract grid → Concat N copies → Pad to 30×30.
For variable sizes: detect grid extent first, then Slice dynamically.

Opset 10: Concat is straightforward. Slice with dynamic starts/ends
requires initializer inputs (not attributes like opset 1-9).
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from compiler.primitives import (
    C, H, W,
    _int64, _vi,
    make_model,
)


def compile_tile_fixed(grid_h: int, grid_w: int,
                        tile_rows: int, tile_cols: int) -> "onnx.ModelProto":
    """Tile a fixed-size grid region tile_rows × tile_cols times.

    Steps:
      1. Slice input to [1, C, grid_h, grid_w]
      2. Concat tile_rows copies along axis 2 → [1, C, grid_h*tile_rows, grid_w]
      3. Concat tile_cols copies along axis 3 → [1, C, grid_h*tile_rows, grid_w*tile_cols]
      4. Pad to [1, C, 30, 30]

    Constraint: grid_h * tile_rows <= 30 and grid_w * tile_cols <= 30.
    """
    out_h = grid_h * tile_rows
    out_w = grid_w * tile_cols
    if out_h > H or out_w > W:
        return None  # can't fit in canvas

    nodes = []
    inits = []
    vinfo = []

    # Step 1: Slice to grid region (opset 10: starts/ends/axes as inputs)
    inits.append(_int64("sl_starts", [0, 0, 0, 0]))
    inits.append(_int64("sl_ends", [1, C, grid_h, grid_w]))
    inits.append(_int64("sl_axes", [0, 1, 2, 3]))
    nodes.append(helper.make_node("Slice",
        ["input", "sl_starts", "sl_ends", "sl_axes"], ["crop"]))
    vinfo.append(_vi("crop", TensorProto.FLOAT, [1, C, grid_h, grid_w]))

    # Step 2: Concat along H (axis 2) — tile_rows copies
    cur = "crop"
    if tile_rows > 1:
        h_inputs = [cur] * tile_rows
        # Need unique names for Concat inputs — but they're all the same tensor
        # ONNX Concat allows repeated input names
        nodes.append(helper.make_node("Concat", h_inputs, ["tiled_h"], axis=2))
        vinfo.append(_vi("tiled_h", TensorProto.FLOAT,
                          [1, C, out_h, grid_w]))
        cur = "tiled_h"

    # Step 3: Concat along W (axis 3) — tile_cols copies
    if tile_cols > 1:
        w_inputs = [cur] * tile_cols
        nodes.append(helper.make_node("Concat", w_inputs, ["tiled_hw"], axis=3))
        vinfo.append(_vi("tiled_hw", TensorProto.FLOAT, [1, C, out_h, out_w]))
        cur = "tiled_hw"

    # Step 4: Pad to 30×30
    if out_h < H or out_w < W:
        pad_h = H - out_h
        pad_w = W - out_w
        nodes.append(helper.make_node("Pad", [cur], ["output"],
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w],
            value=0.0))
    else:
        nodes.append(helper.make_node("Identity", [cur], ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


def detect_tile_params(task: dict) -> tuple[int, int, int, int] | None:
    """Analyze a task to detect if it's a tiling pattern.

    Returns (grid_h, grid_w, tile_rows, tile_cols) or None.

    Detection: check if output is an integer multiple of input size,
    and if the output is the input repeated.
    """
    import numpy as _np

    examples = task.get("train", [])
    if not examples:
        return None

    params = None
    for ex in examples:
        inp = _np.array(ex["input"])
        out = _np.array(ex["output"])
        ih, iw = inp.shape
        oh, ow = out.shape

        if oh == 0 or ow == 0 or ih == 0 or iw == 0:
            return None
        if oh % ih != 0 or ow % iw != 0:
            return None

        tr = oh // ih
        tc = ow // iw

        # Verify: output is input tiled tr × tc
        tiled = _np.tile(inp, (tr, tc))
        if not _np.array_equal(tiled, out):
            return None

        if params is None:
            params = (ih, iw, tr, tc)
        elif params != (ih, iw, tr, tc):
            return None  # inconsistent

    return params


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, "src")
    from grammar.primitives import score_model, verify_model
    import onnx

    # Test on tasks that might be tiling
    for tn in [256, 1, 188, 289, 297]:
        try:
            with open(f"task{tn:03d}.json") as f:
                task = json.load(f)
            params = detect_tile_params(task)
            if params:
                gh, gw, tr, tc = params
                print(f"task{tn:03d}: detected tile {gh}x{gw} × {tr}x{tc}")
                model = compile_tile_fixed(gh, gw, tr, tc)
                if model:
                    s = score_model(model)
                    correct, total = verify_model(model, task)
                    status = "PASS" if correct == total and total > 0 else "FAIL"
                    print(f"  cost={s['cost']} score={s['score']:.3f} "
                          f"verified={correct}/{total} {status}")
                else:
                    print(f"  tile too large for canvas")
            else:
                print(f"task{tn:03d}: not a simple tile pattern")
        except Exception as e:
            print(f"task{tn:03d}: error {e}")
