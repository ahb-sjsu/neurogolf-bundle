"""Parallel compiler — try ALL compilation strategies per task on NRP.

For each unsolved task, spawn a pod that:
  1. Gets the LLM recipe (qwen3)
  2. Tries every compiler pattern (crop, flip, tile, A*, conv sweep)
  3. Tries LLM-generated ONNX code (multiple models × multiple prompts)
  4. Verifies each candidate
  5. Emits the first verified ONNX as base64 to stdout

The Security Radar principle: if a candidate fails on ANY training example,
discard immediately. Only emit 100%-verified models.

Results collected via atlas_collector.py.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def grid_to_onehot(grid):
    t = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= 30: break
        for c, color in enumerate(row):
            if c >= 30: break
            if 0 <= color < 10:
                t[0, color, r, c] = 1.0
    return t


def verify_model(model, task):
    """Verify on ALL examples. Returns (correct, total)."""
    import onnxruntime
    buf = io.BytesIO()
    import onnx
    onnx.save(model, buf)
    sess = onnxruntime.InferenceSession(buf.getvalue(),
        providers=["CPUExecutionProvider"])
    correct = total = 0
    for sub in ("train", "test", "arc-gen"):
        for ex in task.get(sub, []):
            total += 1
            try:
                inp = grid_to_onehot(ex["input"])
                expected = grid_to_onehot(ex["output"])
                out = sess.run(["output"], {"input": inp})[0]
                pred = (out > 0).astype(np.float32)
                if np.array_equal(pred, expected):
                    correct += 1
            except Exception:
                pass
    return correct, total


def try_crop(task, tn):
    """Try crop-to-nonzero-bbox."""
    try:
        from compiler.crop_bbox import compile_crop_nonzero
        model = compile_crop_nonzero()
        c, t = verify_model(model, task)
        if c == t and t > 0:
            return model, "crop_nonzero"
    except Exception:
        pass
    return None, None


def try_flip(task, tn):
    """Try both flip directions."""
    try:
        from compiler.flip import compile_flip_v, compile_flip_h
        for builder, name in [(compile_flip_v, "flip_v"), (compile_flip_h, "flip_h")]:
            model = builder()
            c, t = verify_model(model, task)
            if c == t and t > 0:
                return model, name
    except Exception:
        pass
    return None, None


def try_astar(task, tn, device="cpu"):
    """Try DAG A* search."""
    try:
        from dag_astar.search import astar_solve_task
        state, info = astar_solve_task(task, tn, device=device,
            time_budget_s=30, max_expansions=2000, max_depth=4, verbose=False)
        if state:
            return state.build_model(), "astar"
    except Exception:
        pass
    return None, None


def try_conv(task, tn, device="cpu"):
    """Try GPU conv sweep."""
    try:
        from gpu_conv_trainer import solve_task_gpu
        r = solve_task_gpu(task, tn, device, num_seeds=3, max_time_s=60)
        if r.get("status") == "solved":
            return r["model"], r.get("arch", "conv")
    except Exception:
        pass
    return None, None


def try_llm_onnx(task, tn):
    """Try LLM-generated ONNX code via qwen3."""
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        return None, None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")

        examples = ""
        for i, ex in enumerate(task.get("train", [])[:3]):
            examples += f"Example {i+1}:\n  input:  {ex['input']}\n  output: {ex['output']}\n\n"

        prompt = (
            "Write a Python function build_onnx() that returns an onnx.ModelProto.\n"
            "Input: 'input' shape [1,10,30,30] float32 one-hot.\n"
            "Output: 'output' shape [1,10,30,30] float32.\n"
            "Opset 10, ir_version 10.\n"
            f"\n{examples}\n"
            "Use only: Conv, Gather, Reshape, Concat, Pad, Slice, Mul, Add, Relu, "
            "ReduceMax, ArgMax, Squeeze, Unsqueeze, Transpose, Identity, Clip, Cast.\n"
            "Write ONLY the function. No explanation."
        )

        for model_name in ["gemma", "qwen3-small"]:
            try:
                r = client.chat.completions.create(
                    model=model_name, max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    if "qwen" in model_name else {},
                )
                code = r.choices[0].message.content or ""
                if "```" in code:
                    code = code.split("```")[1]
                    if code.startswith("python"):
                        code = code[6:]

                import onnx
                from onnx import helper, TensorProto
                ns = {"np": np, "onnx": onnx, "helper": helper,
                      "TensorProto": TensorProto}
                exec(code.strip(), ns)
                build = ns.get("build_onnx")
                if build:
                    model = build()
                    c, t = verify_model(model, task)
                    if c == t and t > 0:
                        return model, f"llm_{model_name}"
            except Exception:
                pass
    except Exception:
        pass
    return None, None


def solve_task(task, tn, device="cpu"):
    """Try all strategies, return first verified model."""
    strategies = [
        ("crop", try_crop),
        ("flip", try_flip),
        ("astar", lambda t, n: try_astar(t, n, device)),
        ("conv", lambda t, n: try_conv(t, n, device)),
        ("llm", try_llm_onnx),
    ]

    for name, fn in strategies:
        model, method = fn(task, tn)
        if model is not None:
            return model, method

    return None, None


def main():
    import onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--emit-stdout", action="store_true")
    ap.add_argument("--output-dir", default="solutions_parallel")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    task_nums = [int(t) for t in args.tasks.split(",") if t.strip()]

    for tn in task_nums:
        task_file = ROOT / f"task{tn:03d}.json"
        if not task_file.exists():
            for alt in [Path(f"/data/tasks/task{tn:03d}.json"),
                         Path(f"/data/task{tn:03d}.json")]:
                if alt.exists():
                    task_file = alt
                    break

        if not task_file.exists():
            if args.emit_stdout:
                print(json.dumps({"task": tn, "status": "missing"}), flush=True)
            continue

        with open(task_file) as f:
            task = json.load(f)

        t0 = time.time()
        model, method = solve_task(task, tn, device)
        elapsed = time.time() - t0

        if model is not None:
            # Score it
            from grammar.primitives import score_model
            s = score_model(model)
            cost = s["cost"] if s else 0
            score = s["score"] if s else 1.0

            # Save locally
            onnx.save(model, str(output_dir / f"task{tn:03d}.onnx"))

            if args.emit_stdout:
                buf = io.BytesIO()
                onnx.save(model, buf)
                rec = {
                    "task": tn, "status": "solved",
                    "method": method, "cost": cost,
                    "score": round(score, 3),
                    "elapsed": round(elapsed, 1),
                    "model_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                }
                print(json.dumps(rec), flush=True)
            else:
                print(f"task{tn:03d}: {method} cost={cost} score={score:.3f} "
                      f"t={elapsed:.1f}s", flush=True)
        else:
            if args.emit_stdout:
                print(json.dumps({"task": tn, "status": "unsolved",
                                   "elapsed": round(elapsed, 1)}), flush=True)
            else:
                print(f"task{tn:03d}: unsolved t={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
