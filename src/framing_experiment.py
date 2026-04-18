"""ARC Prompt Framing Experiment — adapted from fun-hypothesis framework.

Tests which prompt framing gets the best Python transforms from VLMs.

Unlike the original fun-hypothesis (which uses judge panels), here we have
GROUND TRUTH: either the transform passes Security Radar (100% verify on
all examples) or it doesn't. No judges needed — math is the judge.

Framings to test:
  - direct: "Write a function that transforms..."
  - step_by_step: "First analyze, then write..."
  - analogies: "Common operations include crop, tile, flip..."
  - concise: Minimal prompt, maximum signal
  - expert: "As an ARC-AGI researcher..."
  - visual_thinker: "Imagine looking at colored grids..."
  - programmer: "Debug this: input->output mapping..."
  - mathematician: "Find the mathematical function..."

Metric: verify rate (% of tasks where transform passes ALL examples)
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ═══════════════════════════════════════════════════════════════
# Framing styles — each is a different way to ask the LLM
# ═══════════════════════════════════════════════════════════════

ARC_FRAMINGS = {
    "direct": {
        "name": "Direct",
        "prompt": (
            "Here are input/output grid pairs:\n\n{examples}\n"
            "Write a Python function: def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "that implements the transformation using numpy.\n"
            "Write ONLY the function. Wrap in ```python ... ```"
        ),
    },
    "step_by_step": {
        "name": "Step-by-Step Analysis",
        "prompt": (
            "Analyze these grid transformations step by step:\n\n{examples}\n"
            "Colors: 0=black,1=blue,2=red,3=green,4=yellow,5=gray,"
            "6=magenta,7=orange,8=lightblue,9=maroon\n\n"
            "Step 1: What objects/patterns exist in the input?\n"
            "Step 2: How do they change in the output?\n"
            "Step 3: What is the general rule?\n"
            "Step 4: Write the function.\n\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]:\n"
            "Wrap in ```python ... ```"
        ),
    },
    "analogies": {
        "name": "Operation Analogies",
        "prompt": (
            "These are ARC-AGI grid transformation puzzles.\n\n{examples}\n"
            "Common operations: crop, tile, flip, rotate, transpose, shift, recolor, "
            "fill, mask, scale, reflect, overlay, extract subgrid, sort, gravity, "
            "flood fill, border detection, pattern matching, symmetry completion.\n\n"
            "Which operation(s) describe this transformation? Then write:\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Wrap in ```python ... ```"
        ),
    },
    "concise": {
        "name": "Minimal/Concise",
        "prompt": (
            "{examples}\n"
            "```python\n"
            "import numpy as np\n"
            "def transform(grid):\n"
            "    grid = np.array(grid)\n"
            "    # YOUR CODE HERE\n"
            "    return result.tolist()\n"
            "```\n"
            "Complete the function."
        ),
    },
    "expert": {
        "name": "ARC Expert",
        "prompt": (
            "You are an ARC-AGI researcher who has solved hundreds of these puzzles. "
            "You know that ARC tasks test core knowledge priors: objectness, numerosity, "
            "basic geometry, and simple physics.\n\n{examples}\n"
            "Based on your expertise, identify the core prior being tested and "
            "write an efficient transform:\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Wrap in ```python ... ```"
        ),
    },
    "visual_thinker": {
        "name": "Visual Thinker",
        "prompt": (
            "Imagine you're looking at colored grids on a screen. "
            "Each cell contains a color (0-9).\n\n{examples}\n"
            "Visualize the input grid. Now visualize the output grid. "
            "What changed? What stayed the same? What moved? What grew? "
            "What shrank? What flipped?\n\n"
            "Now write the transformation:\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Wrap in ```python ... ```"
        ),
    },
    "programmer": {
        "name": "Debugging Programmer",
        "prompt": (
            "I have a function that maps input grids to output grids, but I lost "
            "the source code. Here are some input/output test cases:\n\n{examples}\n"
            "Can you reverse-engineer the function?\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Return list[list[int]]. Wrap in ```python ... ```"
        ),
    },
    "mathematician": {
        "name": "Mathematician",
        "prompt": (
            "Consider the following mapping f: Z^(m x n) -> Z^(p x q) "
            "where Z = {{0,...,9}}.\n\n{examples}\n"
            "Find f and express it as:\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Wrap in ```python ... ```"
        ),
    },
    "teacher": {
        "name": "Teaching Assistant",
        "prompt": (
            "A student is trying to understand pattern recognition. "
            "Show them how these grids transform:\n\n{examples}\n"
            "First explain the pattern simply, then write the code:\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Wrap in ```python ... ```"
        ),
    },
    "competition": {
        "name": "Competition Solver",
        "prompt": (
            "COMPETITION TASK. Solve for maximum points.\n\n{examples}\n"
            "Requirements:\n"
            "- Function must generalize to unseen inputs\n"
            "- Must handle variable grid sizes\n"
            "- Must be deterministic\n\n"
            "def transform(grid: list[list[int]]) -> list[list[int]]\n"
            "Use numpy. Wrap in ```python ... ```"
        ),
    },
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


def security_radar(transform_fn, task: dict) -> tuple[int, int]:
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
                    return correct, total
            except Exception:
                return correct, total
    return correct, total


@dataclass
class FramingTrial:
    framing_key: str
    task_num: int
    verified: bool
    correct: int
    total: int
    elapsed: float
    error: str = ""


@dataclass
class FramingResult:
    framing_key: str
    framing_name: str
    trials: int = 0
    verified: int = 0
    partial: int = 0  # got some right but not all
    failed: int = 0
    errors: int = 0
    verify_rate: float = 0.0
    avg_time: float = 0.0


def run_experiment(task_nums: list[int],
                    models: list[str] = None,
                    framing_keys: list[str] = None,
                    output_file: str = "framing_experiment_results.json"):
    """Run the framing experiment on a set of tasks."""

    if models is None:
        models = ["kimi"]
    if framing_keys is None:
        framing_keys = list(ARC_FRAMINGS.keys())

    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        token_file = Path.home() / ".llmtoken"
        if token_file.exists():
            token = token_file.read_text().strip()

    from openai import OpenAI
    client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")

    results = {}  # framing_key -> FramingResult

    for fk in framing_keys:
        results[fk] = FramingResult(
            framing_key=fk,
            framing_name=ARC_FRAMINGS[fk]["name"],
        )

    print(f"{'='*70}")
    print(f"ARC PROMPT FRAMING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Tasks: {len(task_nums)}")
    print(f"Framings: {len(framing_keys)}")
    print(f"Models: {models}")
    print(f"Total LLM calls: {len(task_nums) * len(framing_keys) * len(models)}")
    print(f"{'='*70}\n")

    all_trials = []

    for tn in task_nums:
        task_file = ROOT / f"task{tn:03d}.json"
        if not task_file.exists():
            continue

        with open(task_file) as f:
            task = json.load(f)

        examples_text = format_examples(task)
        print(f"task{tn:03d}:", flush=True)

        for fk in framing_keys:
            prompt_template = ARC_FRAMINGS[fk]["prompt"]
            prompt = prompt_template.format(examples=examples_text)

            for model_name in models:
                extra = {}
                if model_name == "kimi":
                    extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
                elif "qwen" in model_name:
                    extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

                t0 = time.time()
                trial = FramingTrial(
                    framing_key=fk, task_num=tn,
                    verified=False, correct=0, total=0, elapsed=0.0,
                )

                try:
                    r = client.chat.completions.create(
                        model=model_name, max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}],
                        **extra,
                    )
                    response = r.choices[0].message.content or ""
                    code = extract_code(response)

                    if code:
                        ns = {"np": np, "numpy": np}
                        exec(code.strip(), ns)
                        transform_fn = ns.get("transform")

                        if transform_fn:
                            correct, total = security_radar(transform_fn, task)
                            trial.correct = correct
                            trial.total = total
                            trial.verified = (correct == total and total > 0)
                    else:
                        trial.error = "no_code"
                except Exception as e:
                    trial.error = str(e)[:100]

                trial.elapsed = time.time() - t0
                all_trials.append(asdict(trial))

                # Update stats
                fr = results[fk]
                fr.trials += 1
                if trial.verified:
                    fr.verified += 1
                    tag = "PASS"
                elif trial.correct > 0:
                    fr.partial += 1
                    tag = f"partial({trial.correct}/{trial.total})"
                elif trial.error:
                    fr.errors += 1
                    tag = f"err({trial.error[:30]})"
                else:
                    fr.failed += 1
                    tag = "FAIL"

                print(f"  {fk:20s} {tag}", flush=True)

        print(flush=True)

    # Calculate final stats
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")

    print(f"{'Framing':<22} {'Rate':>6} {'Pass':>5} {'Partial':>8} "
          f"{'Fail':>5} {'Error':>6} {'Trials':>7}")
    print("-" * 65)

    rankings = []
    for fk in framing_keys:
        fr = results[fk]
        if fr.trials > 0:
            fr.verify_rate = fr.verified / fr.trials
        rankings.append((fk, fr))

    rankings.sort(key=lambda x: x[1].verify_rate, reverse=True)

    for fk, fr in rankings:
        print(f"{fr.framing_name:<22} {fr.verify_rate:>5.0%} {fr.verified:>5} "
              f"{fr.partial:>8} {fr.failed:>5} {fr.errors:>6} {fr.trials:>7}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "task_nums": task_nums,
        "models": models,
        "framing_keys": framing_keys,
        "results": {k: asdict(v) for k, v in results.items()},
        "rankings": [(k, v.verify_rate) for k, v in rankings],
        "trials": all_trials,
    }

    output_path = ROOT / output_file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if rankings:
        best_key, best_fr = rankings[0]
        print(f"\nBest framing: {best_fr.framing_name} ({best_fr.verify_rate:.0%} verify rate)")

    return output


def main():
    import argparse

    ap = argparse.ArgumentParser(description="ARC Prompt Framing Experiment")
    ap.add_argument("--tasks", type=str, default="1,3,5,7,8,9,10,11",
                     help="Comma-separated task numbers")
    ap.add_argument("--models", default="kimi",
                     help="Comma-separated model names")
    ap.add_argument("--framings", default=None,
                     help="Comma-separated framings (default: all)")
    ap.add_argument("--output", default="framing_experiment_results.json")
    args = ap.parse_args()

    task_nums = [int(t) for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",")]

    if args.framings:
        framing_keys = [f.strip() for f in args.framings.split(",")]
    else:
        framing_keys = list(ARC_FRAMINGS.keys())

    run_experiment(task_nums, models, framing_keys, args.output)


if __name__ == "__main__":
    main()
