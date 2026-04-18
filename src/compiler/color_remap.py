"""Color remap compiler — static color substitution via 1×1 Conv.

Handles tasks where: output = input with colors relabeled according
to a fixed mapping {src_color: dst_color}.

This is a 1×1 Conv with a 10×10 permutation-like weight matrix:
  W[dst_ch, src_ch, 1, 1] = 1.0 where mapping[src] = dst

Test cases: task016, task337 (already solved, but lower cost than conv sweep).
"""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from compiler.primitives import C, H, W, _vi, make_model


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
    for ex in examples:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])

        if inp.shape != out.shape:
            return None

        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                src = int(inp[r, c])
                dst = int(out[r, c])
                if src in mapping:
                    if mapping[src] != dst:
                        return None  # inconsistent
                else:
                    mapping[src] = dst

    # Check that at least one color changes
    if all(mapping.get(i, i) == i for i in range(C)):
        return None  # identity

    return mapping


def detect_and_compile(task: dict) -> "onnx.ModelProto | None":
    """Detect static color remap and compile if found."""
    mapping = detect_static_color_remap(task)
    if mapping is None:
        return None
    return compile_color_remap(mapping)


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, "src")
    from grammar.primitives import score_model, verify_model

    for tn in range(1, 401):
        try:
            with open(f"task{tn:03d}.json") as f:
                task = json.load(f)
            mapping = detect_static_color_remap(task)
            if mapping:
                model = compile_color_remap(mapping)
                s = score_model(model)
                correct, total = verify_model(model, task)
                status = "PASS" if correct == total and total > 0 else "FAIL"
                print(f"task{tn:03d}: {mapping} cost={s['cost']} "
                      f"verified={correct}/{total} {status}")
        except Exception as e:
            print(f"task{tn:03d}: ERROR {e}")
