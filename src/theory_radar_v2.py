"""Theory Radar v2 — Multi-hypothesis VLM solver with Security Radar filtering.

Theory Radar: generate MANY candidate hypotheses per task via:
  - Multiple VLM models (Kimi, Qwen3, Gemma-4)
  - Multiple prompt framings (vision, text, grid-focused, pattern-focused)
  - Multiple temperatures (0.0, 0.7, 1.0)

Security Radar: kill ANY candidate that fails on even ONE training example.
  - Verification is milliseconds; generation is seconds
  - Generate 10-20 candidates, verify all, keep survivors

Pipeline:
  1. Generate N Python transform candidates (Theory Radar)
  2. Verify each on ALL examples (Security Radar)
  3. For survivors: try ONNX compilation
  4. Verify ONNX on ALL examples (Security Radar again)
  5. Pick cheapest verified ONNX
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ═══════════════════════════════════════════════════════════════
# Prompt variants — Theory Radar generates diverse hypotheses
# ═══════════════════════════════════════════════════════════════

PROMPTS = {
    "direct": (
        "You are solving an ARC-AGI puzzle. Here are input/output pairs:\n\n"
        "{examples}\n"
        "Write a Python function: def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "that implements the transformation using numpy.\n"
        "Write ONLY the function. Wrap in ```python ... ```"
    ),
    "step_by_step": (
        "You are an ARC-AGI expert. Analyze these grid transformations step by step:\n\n"
        "{examples}\n"
        "Colors: 0=black,1=blue,2=red,3=green,4=yellow,5=gray,6=magenta,7=orange,8=lightblue,9=maroon\n\n"
        "1. What objects/patterns exist in the input?\n"
        "2. How do they change in the output?\n"
        "3. What is the general rule?\n\n"
        "Then write: def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "Wrap in ```python ... ```"
    ),
    "concise": (
        "ARC puzzle. Transform input grid to output grid.\n\n"
        "{examples}\n"
        "def transform(grid: list[list[int]]) -> list[list[int]]:\n"
        "    # Use numpy. Return list[list[int]].\n"
        "Write the complete function. ```python ... ```"
    ),
    "analogies": (
        "These are ARC-AGI grid transformation puzzles.\n\n"
        "{examples}\n"
        "Think about what geometric or color operation maps each input to its output. "
        "Common operations: crop, tile, flip, rotate, transpose, shift, recolor, "
        "fill, mask, scale, reflect, overlay, extract subgrid, sort, gravity.\n\n"
        "Write: def transform(grid: list[list[int]]) -> list[list[int]]\n"
        "Use only numpy. Wrap in ```python ... ```"
    ),
    "minimal": (
        "{examples}\n"
        "def transform(grid):\n"
        "    import numpy as np\n"
        "    # implement the pattern above\n"
        "Complete this function. ```python ... ```"
    ),
}


def format_examples(task: dict, max_examples: int = 4) -> str:
    """Format task examples as text."""
    parts = []
    for i, ex in enumerate(task.get("train", [])[:max_examples]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        parts.append(f"Example {i+1}:")
        parts.append(f"  input ({inp.shape[0]}x{inp.shape[1]}): {ex['input']}")
        parts.append(f"  output ({out.shape[0]}x{out.shape[1]}): {ex['output']}")
        parts.append("")
    return "\n".join(parts)


def extract_code(response: str) -> str | None:
    """Extract Python code from LLM response."""
    if not response:
        return None
    if "```" in response:
        parts = response.split("```")
        for p in parts[1:]:
            if p.startswith("python"):
                p = p[6:]
            code = p.strip()
            if "def transform" in code:
                return code
    if "def transform" in response:
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if "def transform" in line:
                return "\n".join(lines[i:])
    return None


# ═══════════════════════════════════════════════════════════════
# Security Radar — ruthless verification
# ═══════════════════════════════════════════════════════════════

def security_radar_python(transform_fn, task: dict) -> tuple[int, int]:
    """Verify transform on ALL examples. Any failure = kill."""
    correct = total = 0
    for split in ("train", "test", "arc-gen"):
        for ex in task.get(split, []):
            total += 1
            try:
                result = transform_fn(ex["input"])
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                if result == ex["output"]:
                    correct += 1
                else:
                    return correct, total  # Early exit on first failure
            except Exception:
                return correct, total
    return correct, total


def security_radar_onnx(model, task: dict) -> tuple[int, int]:
    """Verify ONNX model on ALL examples. Any failure = kill."""
    from grammar.primitives import verify_model
    return verify_model(model, task)


# ═══════════════════════════════════════════════════════════════
# Theory Radar — generate diverse hypotheses
# ═══════════════════════════════════════════════════════════════

def generate_candidates(client, task: dict,
                         models: list[str],
                         temps: list[float] = None) -> list[dict]:
    """Generate multiple transform candidates via different prompts/models/temps.

    Returns list of {code, model, prompt_key, temp} dicts.
    """
    if temps is None:
        temps = [0.0, 0.7]

    examples_text = format_examples(task)
    candidates = []

    for model_name in models:
        extra = {}
        if model_name == "kimi":
            extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
        elif "qwen" in model_name:
            extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

        for prompt_key, prompt_template in PROMPTS.items():
            for temp in temps:
                try:
                    prompt = prompt_template.format(examples=examples_text)
                    r = client.chat.completions.create(
                        model=model_name,
                        max_tokens=2000,
                        temperature=temp,
                        messages=[{"role": "user", "content": prompt}],
                        **extra,
                    )
                    response = r.choices[0].message.content or ""
                    code = extract_code(response)
                    if code:
                        candidates.append({
                            "code": code,
                            "model": model_name,
                            "prompt": prompt_key,
                            "temp": temp,
                        })
                except Exception:
                    pass

    return candidates


# ═══════════════════════════════════════════════════════════════
# ONNX compilation strategies
# ═══════════════════════════════════════════════════════════════

def try_compile_to_onnx(transform_fn, task: dict, task_num: int,
                         client=None, device: str = "cpu"):
    """Try multiple ONNX compilation strategies."""
    import onnx

    # Strategy 1: Trace-based compilation
    try:
        from compiler.trace_compile import trace_compile
        model = trace_compile(transform_fn, task, task_num, device)
        if model:
            c, t = security_radar_onnx(model, task)
            if c == t and t > 0:
                return model, "trace"
    except Exception:
        pass

    # Strategy 2: DAG A*
    try:
        from dag_astar.search import astar_solve_task
        state, info = astar_solve_task(task, task_num, device=device,
                                        time_budget_s=30, max_expansions=2000,
                                        max_depth=4, verbose=False)
        if state:
            model = state.build_model()
            c, t = security_radar_onnx(model, task)
            if c == t and t > 0:
                return model, "astar"
    except Exception:
        pass

    # Strategy 3: Conv training
    try:
        from gpu_conv_trainer import solve_task_gpu
        r = solve_task_gpu(task, task_num, device, num_seeds=5, max_time_s=120)
        if r.get("status") == "solved":
            model = r["model"]
            c, t = security_radar_onnx(model, task)
            if c == t and t > 0:
                return model, r.get("arch", "conv")
    except Exception:
        pass

    # Strategy 4: LLM writes ONNX directly
    if client:
        try:
            import inspect
            source = inspect.getsource(transform_fn)
            from vlm_solver import ONNX_PROMPT
            for model_name in ["kimi", "qwen3"]:
                try:
                    extra = {}
                    if model_name == "kimi":
                        extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
                    elif "qwen" in model_name:
                        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

                    r = client.chat.completions.create(
                        model=model_name, max_tokens=3000,
                        messages=[{"role": "user",
                                   "content": ONNX_PROMPT.format(description=source)}],
                        **extra,
                    )
                    code = extract_code(r.choices[0].message.content or "")
                    if code:
                        from onnx import helper, TensorProto
                        ns = {"np": np, "onnx": onnx, "helper": helper,
                              "TensorProto": TensorProto}
                        exec(code.strip(), ns)
                        build = ns.get("build_onnx")
                        if build:
                            model = build()
                            c, t = security_radar_onnx(model, task)
                            if c == t and t > 0:
                                return model, f"llm_onnx_{model_name}"
                except Exception:
                    pass
        except Exception:
            pass

    return None, None


# ═══════════════════════════════════════════════════════════════
# Main solver
# ═══════════════════════════════════════════════════════════════

def solve_task(task: dict, task_num: int, client,
               models: list[str], device: str = "cpu",
               max_candidates: int = 20) -> dict:
    """Full Theory Radar + Security Radar pipeline."""

    result = {
        "task": task_num,
        "status": "unsolved",
        "candidates_generated": 0,
        "candidates_verified": 0,
    }

    t0 = time.time()

    # Phase 1: Theory Radar — generate candidates
    candidates = generate_candidates(client, task, models)
    result["candidates_generated"] = len(candidates)

    if not candidates:
        return result

    # Phase 2: Security Radar — verify each candidate
    verified = []
    seen_codes = set()

    for cand in candidates[:max_candidates]:
        code = cand["code"]
        # Dedup by code hash
        code_hash = hash(code.strip())
        if code_hash in seen_codes:
            continue
        seen_codes.add(code_hash)

        try:
            ns = {"np": np, "numpy": np}
            exec(code.strip(), ns)
            transform_fn = ns.get("transform")
            if not transform_fn:
                continue

            correct, total = security_radar_python(transform_fn, task)
            if correct == total and total > 0:
                verified.append({
                    **cand,
                    "transform_fn": transform_fn,
                    "verified": f"{correct}/{total}",
                })
        except Exception:
            pass

    result["candidates_verified"] = len(verified)

    if not verified:
        elapsed = time.time() - t0
        result["elapsed"] = round(elapsed, 1)
        return result

    # Phase 3: Compile best verified transform to ONNX
    # Try the simplest-looking code first (shorter = likely simpler pattern)
    verified.sort(key=lambda x: len(x["code"]))

    for v in verified:
        model, method = try_compile_to_onnx(
            v["transform_fn"], task, task_num, client, device)

        if model is not None:
            from grammar.primitives import score_model
            s = score_model(model)
            result["status"] = "solved"
            result["method"] = method
            result["cost"] = s["cost"] if s else 0
            result["score"] = s["score"] if s else 1.0
            result["source_model"] = v["model"]
            result["source_prompt"] = v["prompt"]
            result["onnx_model"] = model
            result["code"] = v["code"]
            break

    if result["status"] != "solved" and verified:
        # Save transform even without ONNX
        result["status"] = "transform_only"
        result["code"] = verified[0]["code"]
        result["source_model"] = verified[0]["model"]

    elapsed = time.time() - t0
    result["elapsed"] = round(elapsed, 1)
    return result


def main():
    import onnx

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=400)
    ap.add_argument("--models", default="kimi,qwen3")
    ap.add_argument("--output-dir", default="solutions_radar")
    ap.add_argument("--skip-solved", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-transforms", action="store_true")
    ap.add_argument("--max-candidates", type=int, default=20)
    args = ap.parse_args()

    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        token_file = Path.home() / ".llmtoken"
        if token_file.exists():
            token = token_file.read_text().strip()

    from openai import OpenAI
    client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")

    models = [m.strip() for m in args.models.split(",")]

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = ROOT / "solutions_radar_transforms"
    if args.save_transforms:
        transforms_dir.mkdir(parents=True, exist_ok=True)

    solved = set()
    if args.skip_solved:
        for d in ["solutions_safe", "solutions_merged_latest"]:
            p = ROOT / d
            if p.exists():
                solved.update(int(f.stem[4:]) for f in p.glob("task*.onnx"))

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    stats = {"total": 0, "verified": 0, "compiled": 0}

    for tn in range(args.start, args.end + 1):
        if tn in solved:
            continue
        task_file = ROOT / f"task{tn:03d}.json"
        if not task_file.exists():
            continue

        with open(task_file) as f:
            task = json.load(f)

        stats["total"] += 1
        result = solve_task(task, tn, client, models, device,
                            args.max_candidates)

        gen = result["candidates_generated"]
        ver = result["candidates_verified"]

        if result["status"] == "solved":
            stats["compiled"] += 1
            stats["verified"] += 1
            onnx_model = result.pop("onnx_model", None)
            if onnx_model:
                onnx.save(onnx_model, str(output_dir / f"task{tn:03d}.onnx"))
            print(f"task{tn:03d}: SOLVED method={result['method']} "
                  f"cost={result['cost']} gen={gen} ver={ver} "
                  f"t={result['elapsed']}s", flush=True)

        elif result["status"] == "transform_only":
            stats["verified"] += 1
            if args.save_transforms and result.get("code"):
                with open(transforms_dir / f"task{tn:03d}.py", "w") as f:
                    f.write(result["code"])
            print(f"task{tn:03d}: TRANSFORM_ONLY gen={gen} ver={ver} "
                  f"t={result['elapsed']}s", flush=True)

        else:
            print(f"task{tn:03d}: unsolved gen={gen} ver={ver} "
                  f"t={result.get('elapsed', 0)}s", flush=True)

    print(f"\nTotal: {stats['total']} | Verified: {stats['verified']} | "
          f"Compiled: {stats['compiled']}")


if __name__ == "__main__":
    main()
