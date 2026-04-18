# ONNX Compiler Curriculum

Study these patterns to write new compiler modules.
Each module: nodes/inits/vinfo lists -> make_model().

## color_remap.py
"""Color remap compiler — static color substitution via 1×1 Conv.

Handles tasks where: output = input with colors relabeled according
to a fixed mapping {src_color: dst_color}.


def compile_color_remap(mapping: dict[int, int]) -> "onnx.ModelProto":
    """Build ONNX model that remaps colors via 1×1 Conv.

    mapping: {src_color: dst_color} for colors 0-9.
    Colors not in mapping are left unchanged (identity).
    """
    nodes = []
    inits = []
    vinfo = []

    # Build 10×10 weight matrix
    W_data = np.zeros((C, C, 1, 1), dtype=np.float32)
    for src in range(C):
        dst = mapping.get(src, src)  # default: identity
        if 0 <= dst < C:
            W_data[dst, src, 0, 0] = 1.0

    inits.append(helper.make_tensor("W", TensorProto.FLOAT,
                                     [C, C, 1, 1], W_data.flatten().tolist()))

    nodes.append(helper.make_node("Conv", ["input", "W"], ["output"],
                                   kernel_shape=[1, 1]))

    return make_model(nodes, inits, vinfo, output_name="output")


def detect_static_color_remap(task: dict) -> dict[int, int] | None:
    """Detect if a task is a static color remap.

    Returns the mapping {src: dst} or None if not consistent.
    Requirements:
      - Same shape input/output for all examples
      - Consistent per-color mapping across all examples
      - At least one color changes
    """
    examples = task.get("train", [])
    if not examples:
        return None

    mapping = {}

---

## crop_bbox.py
"""Crop-to-bbox compiler — extract non-zero bounding box and place at origin.

Handles tasks where: output = input[r_min:r_max+1, c_min:c_max+1]
shifted to position (0,0) in the 30×30 canvas.


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

---

## flip.py
"""Flip compiler — vertical and horizontal flips for variable-size grids.

Test cases:
  task155: grid[::-1]  (vertical flip) — 266/266 verified
  task150: np.fliplr   (horizontal flip)

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


---

## rotate.py
"""Rotate compiler — 90/180/270 degree rotations.

Test cases:
  task380: rotate 90 degrees CCW (already solved by conv, but cheaper here)


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

---

## tile.py
"""Tile compiler — replicate grid content via Concat.

Test cases:
  task256: horizontal double (hstack)
  task001: Kronecker self-tile (tile where non-zero)

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


---

## transpose.py
"""Transpose compiler — swap rows and columns.

Test cases:
  task179: grid.T (transpose)
  task241: grid.T (transpose)

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

---

