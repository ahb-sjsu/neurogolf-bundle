"""VLM-guided ARC solver — uses vision models to analyze tasks and generate transforms.

Pipeline:
  1. Render task as image (input→output pairs)
  2. Send to VLM (Kimi-K2.5, Qwen3.5, Gemma-4) for analysis
  3. VLM returns Python transform function
  4. Verify transform on all examples
  5. If verified, try to compile to ONNX via:
     a. Direct ONNX code generation by LLM
     b. Conv training to approximate the transform
     c. DAG A* guided search

Uses NRP managed LLM API at https://ellm.nrp-nautilus.io/v1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ARC color palette (standard)
ARC_COLORS = [
    (0, 0, 0),       # 0: black
    (0, 116, 217),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (170, 170, 170),  # 5: gray
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: light blue
    (135, 12, 37),    # 9: maroon
]


def grid_to_image(grid: list[list[int]], cell_size: int = 20) -> Image.Image:
    """Render a grid as a colored image with grid lines."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    img = Image.new("RGB", (w * cell_size + 1, h * cell_size + 1), (0, 0, 0))
    pixels = img.load()
    for r in range(h):
        for c in range(w):
            color = ARC_COLORS[grid[r][c] % 10]
            for dy in range(cell_size):
                for dx in range(cell_size):
                    if dy == 0 or dx == 0:
                        pixels[c * cell_size + dx, r * cell_size + dy] = (64, 64, 64)
                    else:
                        pixels[c * cell_size + dx, r * cell_size + dy] = color
    return img


def task_to_image(task: dict, include_test: bool = False) -> Image.Image:
    """Render all training examples as a vertical stack of input→output pairs."""
    examples = task.get("train", [])
    if include_test:
        examples = examples + task.get("test", [])[:1]

    rows = []
    for ex in examples:
        inp_img = grid_to_image(ex["input"])
        out_img = grid_to_image(ex["output"])
        gap = 30
        combined_w = inp_img.width + gap + out_img.width
        combined_h = max(inp_img.height, out_img.height)
        combined = Image.new("RGB", (combined_w, combined_h), (32, 32, 32))
        combined.paste(inp_img, (0, 0))
        combined.paste(out_img, (inp_img.width + gap, 0))
        rows.append(combined)

    if not rows:
        return Image.new("RGB", (100, 100), (0, 0, 0))

    total_h = sum(r.height for r in rows) + 10 * (len(rows) - 1)
    max_w = max(r.width for r in rows)
    final = Image.new("RGB", (max_w, total_h), (32, 32, 32))
    y = 0
    for r in rows:
        final.paste(r, (0, y))
        y += r.height + 10
    return final


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ═══════════════════════════════════════════════════════════════
# VLM prompts
# ═══════════════════════════════════════════════════════════════

ANALYZE_PROMPT = """You are an expert at ARC-AGI puzzles. Each row shows an input grid (left) and its output grid (right).

Colors: black=0, blue=1, red=2, green=3, yellow=4, gray=5, magenta=6, orange=7, lightblue=8, maroon=9.

Analyze the transformation and write a Python function that implements it.

RULES:
- Function signature: def transform(grid: list[list[int]]) -> list[list[int]]
- Use only numpy operations
- The function must work for ANY valid input, not just the examples shown
- Focus on the GENERAL rule, not memorizing specific examples

Write ONLY the function. No explanation before or after. Wrap in ```python ... ```"""

ONNX_PROMPT = """You are an ONNX expert. Write a Python function build_onnx() that returns an onnx.ModelProto implementing this ARC transformation:

{description}

Constraints:
- Input: 'input' shape [1,10,30,30] float32 one-hot (channel c has 1.0 where color=c)
- Output: 'output' shape [1,10,30,30] float32 one-hot
- Opset 10, ir_version 10
- Grid content is embedded in top-left of 30x30 canvas, rest is zeros
- Allowed ops: Conv, Gather, Reshape, Concat, Pad, Slice, Mul, Add, Relu,
  ReduceMax, ArgMax, Squeeze, Unsqueeze, Transpose, Identity, Clip, Cast, Sub
- NO: Loop, Scan, If, NonZero, Unique, Where, Resize

Use onnx.helper to build the graph manually. Return the ModelProto.
Write ONLY the function. No explanation."""


# ═══════════════════════════════════════════════════════════════
# VLM calls
# ═══════════════════════════════════════════════════════════════

def get_client():
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        token_file = Path.home() / ".llmtoken"
        if token_file.exists():
            token = token_file.read_text().strip()
    from openai import OpenAI
    return OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")


def vlm_analyze(client, task: dict, model: str = "kimi") -> str | None:
    """Send task image to VLM and get Python transform code."""
    img = task_to_image(task)
    b64 = image_to_b64(img)

    extra = {}
    if model == "kimi":
        extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    elif "qwen" in model:
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    try:
        r = client.chat.completions.create(
            model=model, max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": ANALYZE_PROMPT},
                ]
            }],
            **extra
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  VLM error ({model}): {e}", flush=True)
        return None


def text_analyze(client, task: dict, model: str = "kimi") -> str | None:
    """Text-only analysis with grid data (fallback if vision fails)."""
    examples = ""
    for i, ex in enumerate(task.get("train", [])[:4]):
        examples += f"Example {i+1}:\n"
        examples += f"  input ({len(ex['input'])}x{len(ex['input'][0])}): {ex['input']}\n"
        examples += f"  output ({len(ex['output'])}x{len(ex['output'][0])}): {ex['output']}\n\n"

    prompt = (
        "You are an expert at ARC-AGI puzzles. Here are input/output pairs:\n\n"
        f"{examples}\n"
        "Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, "
        "6=magenta, 7=orange, 8=lightblue, 9=maroon.\n\n"
        "Write a Python function: def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "that implements the transformation using only numpy.\n"
        "Write ONLY the function. Wrap in ```python ... ```"
    )

    extra = {}
    if model == "kimi":
        extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    elif "qwen" in model:
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    try:
        r = client.chat.completions.create(
            model=model, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            **extra
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  text error ({model}): {e}", flush=True)
        return None


def extract_code(response: str) -> str | None:
    """Extract Python code from LLM response."""
    if "```" in response:
        parts = response.split("```")
        for p in parts[1:]:
            if p.startswith("python"):
                p = p[6:]
            code = p.strip()
            if "def transform" in code or "def build_onnx" in code:
                return code
    # Try the whole response
    if "def transform" in response:
        lines = response.split("\n")
        start = None
        for i, line in enumerate(lines):
            if "def transform" in line:
                start = i
                break
        if start is not None:
            return "\n".join(lines[start:])
    return None


# ═══════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════

def verify_transform(transform_fn, task: dict) -> tuple[int, int]:
    """Verify transform function on all examples."""
    correct = total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            total += 1
            try:
                result = transform_fn(ex["input"])
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                expected = ex["output"]
                if result == expected:
                    correct += 1
            except Exception:
                pass
    return correct, total


def try_compile_transform(transform_fn, task: dict, task_num: int,
                           client=None, device: str = "cpu"):
    """Try to compile a verified Python transform to ONNX.

    Strategy 1: Ask LLM to write ONNX directly
    Strategy 2: Conv training to approximate
    Strategy 3: DAG A* guided search
    """
    import onnx

    # Strategy 1: LLM-generated ONNX
    if client:
        # Get a text description of what the transform does
        import inspect
        source = inspect.getsource(transform_fn)
        for model in ["kimi", "qwen3"]:
            try:
                extra = {}
                if model == "kimi":
                    extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
                elif "qwen" in model:
                    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

                r = client.chat.completions.create(
                    model=model, max_tokens=3000,
                    messages=[{"role": "user",
                               "content": ONNX_PROMPT.format(description=source)}],
                    **extra
                )
                code = extract_code(r.choices[0].message.content or "")
                if code:
                    from onnx import helper, TensorProto
                    ns = {"np": np, "onnx": onnx, "helper": helper,
                          "TensorProto": TensorProto}
                    exec(code.strip(), ns)
                    build = ns.get("build_onnx")
                    if build:
                        model_proto = build()
                        from grammar.primitives import verify_model
                        c, t = verify_model(model_proto, task)
                        if c == t and t > 0:
                            return model_proto, f"llm_onnx_{model}"
            except Exception:
                pass

    # Strategy 2: Conv training
    try:
        from gpu_conv_trainer import solve_task_gpu
        r = solve_task_gpu(task, task_num, device, num_seeds=3, max_time_s=60)
        if r.get("status") == "solved":
            return r["model"], r.get("arch", "conv")
    except Exception:
        pass

    # Strategy 3: DAG A*
    try:
        from dag_astar.search import astar_solve_task
        state, info = astar_solve_task(task, task_num, device=device,
                                        time_budget_s=30, max_expansions=2000,
                                        max_depth=4, verbose=False)
        if state:
            return state.build_model(), "astar"
    except Exception:
        pass

    return None, None


# ═══════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════

def solve_task(task: dict, task_num: int, client=None,
               models: list[str] = None, device: str = "cpu",
               use_vision: bool = True) -> dict:
    """Full pipeline: VLM analysis → verify → compile → ONNX."""
    if models is None:
        models = ["kimi", "qwen3"]

    result = {"task": task_num, "status": "unsolved"}

    for model_name in models:
        # Step 1: Get transform from VLM
        if use_vision:
            response = vlm_analyze(client, task, model=model_name)
        else:
            response = text_analyze(client, task, model=model_name)

        if not response:
            continue

        code = extract_code(response)
        if not code:
            print(f"  {model_name}: no code extracted", flush=True)
            continue

        # Step 2: Execute and verify
        try:
            ns = {"np": np, "numpy": np}
            exec(code.strip(), ns)
            transform_fn = ns.get("transform")
            if not transform_fn:
                print(f"  {model_name}: no transform function", flush=True)
                continue

            correct, total = verify_transform(transform_fn, task)
            print(f"  {model_name}: verify={correct}/{total}", flush=True)

            if correct == total and total > 0:
                result["transform_verified"] = True
                result["model_used"] = model_name
                result["code"] = code

                # Step 3: Compile to ONNX
                onnx_model, method = try_compile_transform(
                    transform_fn, task, task_num, client, device)

                if onnx_model is not None:
                    from grammar.primitives import score_model
                    s = score_model(onnx_model)
                    result["status"] = "solved"
                    result["method"] = method
                    result["cost"] = s["cost"] if s else 0
                    result["onnx_model"] = onnx_model
                    return result
                else:
                    result["status"] = "transform_only"
                    print(f"  transform verified but ONNX compile failed", flush=True)
            else:
                # Partial — might still be useful
                if correct > 0:
                    result["partial"] = f"{correct}/{total}"
        except Exception as e:
            print(f"  {model_name}: exec error: {e}", flush=True)

    return result


def main():
    import onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=400)
    ap.add_argument("--models", default="kimi,qwen3")
    ap.add_argument("--output-dir", default="solutions_vlm")
    ap.add_argument("--skip-solved", action="store_true")
    ap.add_argument("--text-only", action="store_true",
                     help="Use text-only analysis (no vision)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-transforms", action="store_true",
                     help="Save verified Python transforms even without ONNX")
    args = ap.parse_args()

    client = get_client()
    models = [m.strip() for m in args.models.split(",")]

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = ROOT / "solutions_vlm_transforms"
    if args.save_transforms:
        transforms_dir.mkdir(parents=True, exist_ok=True)

    solved = set()
    if args.skip_solved:
        for d in ["solutions_safe", "solutions_merged_latest", "solutions_mshanawaz"]:
            p = ROOT / d
            if p.exists():
                solved.update(int(f.stem[4:]) for f in p.glob("task*.onnx"))

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    stats = {"analyzed": 0, "transform_verified": 0, "onnx_compiled": 0}

    for tn in range(args.start, args.end + 1):
        if tn in solved:
            continue
        task_file = ROOT / f"task{tn:03d}.json"
        if not task_file.exists():
            continue

        with open(task_file) as f:
            task = json.load(f)

        print(f"task{tn:03d}:", end=" ", flush=True)
        t0 = time.time()

        result = solve_task(task, tn, client, models, device,
                            use_vision=not args.text_only)
        elapsed = time.time() - t0
        stats["analyzed"] += 1

        if result.get("transform_verified"):
            stats["transform_verified"] += 1

        if result["status"] == "solved":
            stats["onnx_compiled"] += 1
            onnx_model = result.pop("onnx_model")
            onnx.save(onnx_model, str(output_dir / f"task{tn:03d}.onnx"))
            print(f"SOLVED method={result['method']} cost={result['cost']} "
                  f"t={elapsed:.1f}s", flush=True)
        elif result["status"] == "transform_only":
            if args.save_transforms and result.get("code"):
                with open(transforms_dir / f"task{tn:03d}.py", "w") as f:
                    f.write(result["code"])
            print(f"transform_only (no ONNX) t={elapsed:.1f}s", flush=True)
        else:
            partial = result.get("partial", "")
            print(f"unsolved {partial} t={elapsed:.1f}s", flush=True)

    print(f"\nStats: analyzed={stats['analyzed']} "
          f"transforms={stats['transform_verified']} "
          f"onnx={stats['onnx_compiled']}")


if __name__ == "__main__":
    main()
