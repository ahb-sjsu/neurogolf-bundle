"""Trace compiler — observe a Python transform and build ONNX from the mapping.

Given a verified Python transform function, run it on all training examples
and try to learn the mapping pattern.

Strategies:
  1. Fixed mapping: if output is always the same regardless of input, just
     store it as a constant (Identity-like).
  2. Per-pixel color remap: if each output pixel is a function of only
     its corresponding input pixel's color, build a 1×1 Conv.
  3. Spatial remap: if each output pixel copies from a fixed source position
     (regardless of color), build a Gather with static indices.
  4. Conv approximation: train a small conv network to match the transform.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def trace_transform(transform_fn, task: dict) -> dict:
    """Run transform on all examples and collect input/output pairs."""
    pairs = []
    for split in ("train", "test"):
        for ex in task.get(split, []):
            inp = ex["input"]
            try:
                out = transform_fn(inp)
                if isinstance(out, np.ndarray):
                    out = out.tolist()
                pairs.append((inp, out))
            except Exception:
                pass
    return pairs


def try_constant_output(pairs: list) -> "onnx.ModelProto | None":
    """Check if output is always the same (constant function)."""
    if not pairs:
        return None

    # Check if all outputs are identical
    first_out = pairs[0][1]
    for _, out in pairs[1:]:
        if out != first_out:
            return None

    # Build constant ONNX model
    from compiler.primitives import C, H, W, _vi, make_model, _int64
    from onnx import TensorProto, helper

    out_arr = np.zeros((1, C, H, W), dtype=np.float32)
    out_grid = np.array(first_out)
    oh, ow = out_grid.shape
    for r in range(min(oh, H)):
        for c in range(min(ow, W)):
            color = int(out_grid[r, c])
            if 0 <= color < C:
                out_arr[0, color, r, c] = 1.0

    nodes = []
    inits = []
    vinfo = []

    # Store constant output
    inits.append(helper.make_tensor("const_out", TensorProto.FLOAT,
                                     [1, C, H, W], out_arr.flatten().tolist()))
    nodes.append(helper.make_node("Identity", ["const_out"], ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


def try_pixel_remap(pairs: list) -> "onnx.ModelProto | None":
    """Check if output[r,c] depends only on input[r,c] (1×1 conv)."""
    if not pairs:
        return None

    # Build color mapping from all pairs
    mapping = {}  # src_color -> dst_color
    for inp, out in pairs:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None  # shape changes = not pixel remap

        for r in range(inp_arr.shape[0]):
            for c in range(inp_arr.shape[1]):
                src = int(inp_arr[r, c])
                dst = int(out_arr[r, c])
                if src in mapping:
                    if mapping[src] != dst:
                        return None  # inconsistent
                else:
                    mapping[src] = dst

    # Must have at least one non-identity mapping
    if all(mapping.get(i, i) == i for i in range(10)):
        return None

    from compiler.color_remap import compile_color_remap
    return compile_color_remap(mapping)


def try_spatial_remap(pairs: list) -> "onnx.ModelProto | None":
    """Check if output[r,c] = input[f(r,c)] for fixed source mapping f.

    This handles crops, shifts, rotations, flips, etc. where each output
    position always reads from the same input position.
    """
    if not pairs:
        return None

    from compiler.primitives import C, H, W, _int64, _vi, make_model
    from onnx import TensorProto, helper

    # For each output position, determine the source position
    # We need multiple examples with different colors to disambiguate

    # Convert to numpy
    np_pairs = [(np.array(inp), np.array(out)) for inp, out in pairs]

    # Check all have same input and output sizes
    shapes = set((inp.shape, out.shape) for inp, out in np_pairs)
    if len(shapes) != 1:
        return None  # variable sizes

    (ih, iw), (oh, ow) = shapes.pop()
    if oh > H or ow > W or ih > H or iw > W:
        return None

    # For each output position (r, c), find the source position (sr, sc)
    # We need the same source across all examples
    source_map = {}  # (r, c) -> (sr, sc) or None

    for r in range(oh):
        for c in range(ow):
            candidates = None
            for inp, out in np_pairs:
                out_color = int(out[r, c])
                if out_color == 0:
                    # Could come from any zero input position — skip for now
                    continue

                # Find all input positions with this color
                positions = set()
                for sr in range(ih):
                    for sc in range(iw):
                        if int(inp[sr, sc]) == out_color:
                            positions.add((sr, sc))

                if candidates is None:
                    candidates = positions
                else:
                    candidates = candidates & positions

                if not candidates:
                    break

            if candidates and len(candidates) == 1:
                source_map[(r, c)] = candidates.pop()
            elif candidates is None:
                # All outputs were 0 at this position — map to background
                source_map[(r, c)] = None
            else:
                # Ambiguous or no candidates
                source_map[(r, c)] = None

    # Check if we have enough resolved positions
    resolved = {k: v for k, v in source_map.items() if v is not None}
    if len(resolved) < max(1, oh * ow * 0.3):
        return None  # too few resolved

    # Build gather indices
    # For positions with resolved sources, use the source flat index
    # For unresolved, use a sentinel (position that's always 0)
    sentinel = (H - 1) * W + (W - 1)  # last position in canvas

    indices = np.full(H * W, sentinel, dtype=np.int64)
    for r in range(oh):
        for c in range(ow):
            src = source_map.get((r, c))
            if src is not None:
                sr, sc = src
                indices[r * W + c] = sr * W + sc

    # Build ONNX model with Gather
    nodes = []
    inits = []
    vinfo = []

    inits.append(_int64("src_idx", indices.tolist()))
    inits.append(_int64("s_flat", [1, C, H * W]))
    inits.append(_int64("s_canvas", [1, C, H, W]))

    nodes.append(helper.make_node("Reshape", ["input", "s_flat"], ["flat"]))
    vinfo.append(_vi("flat", TensorProto.FLOAT, [1, C, H * W]))

    nodes.append(helper.make_node("Gather", ["flat", "src_idx"],
                                   ["gathered"], axis=2))
    vinfo.append(_vi("gathered", TensorProto.FLOAT, [1, C, H * W]))

    nodes.append(helper.make_node("Reshape", ["gathered", "s_canvas"],
                                   ["output"]))

    return make_model(nodes, inits, vinfo, output_name="output")


def trace_compile(transform_fn, task: dict, task_num: int = 0,
                   device: str = "cpu") -> "onnx.ModelProto | None":
    """Try all trace-based compilation strategies."""
    pairs = trace_transform(transform_fn, task)
    if not pairs:
        return None

    from grammar.primitives import verify_model

    # Strategy 1: Constant output
    model = try_constant_output(pairs)
    if model:
        c, t = verify_model(model, task)
        if c == t and t > 0:
            return model

    # Strategy 2: Pixel remap
    model = try_pixel_remap(pairs)
    if model:
        c, t = verify_model(model, task)
        if c == t and t > 0:
            return model

    # Strategy 3: Spatial remap
    model = try_spatial_remap(pairs)
    if model:
        c, t = verify_model(model, task)
        if c == t and t > 0:
            return model

    # Strategy 4: Conv training
    try:
        from gpu_conv_trainer import solve_task_gpu
        r = solve_task_gpu(task, task_num, device, num_seeds=5, max_time_s=120)
        if r.get("status") == "solved":
            return r["model"]
    except Exception:
        pass

    return None


if __name__ == "__main__":
    import json
    sys.path.insert(0, "src")
    from grammar.primitives import score_model, verify_model

    # Test: load verified transforms and try to compile
    transforms_dir = ROOT / "solutions_vlm_transforms"
    if transforms_dir.exists():
        for f in sorted(transforms_dir.glob("task*.py")):
            tn = int(f.stem[4:])
            code = f.read_text()

            ns = {"np": np, "numpy": np}
            try:
                exec(code.strip(), ns)
            except Exception as e:
                print(f"task{tn:03d}: exec error {e}")
                continue

            transform_fn = ns.get("transform")
            if not transform_fn:
                print(f"task{tn:03d}: no transform function")
                continue

            task_file = ROOT / f"task{tn:03d}.json"
            if not task_file.exists():
                continue

            with open(task_file) as tf:
                task = json.load(tf)

            model = trace_compile(transform_fn, task, tn)
            if model:
                s = score_model(model)
                c, t = verify_model(model, task)
                print(f"task{tn:03d}: cost={s['cost']} verified={c}/{t} "
                      f"{'PASS' if c == t and t > 0 else 'FAIL'}")
            else:
                print(f"task{tn:03d}: no ONNX model compiled")
