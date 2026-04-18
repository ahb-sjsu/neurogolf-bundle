"""Coverage-guided prompt fuzzer for ARC task solving.

Applies AFL-style greybox fuzzing to prompt engineering:
  - Seed corpus: known prompt framings
  - Mutations: structural and lexical prompt transformations
  - Coverage: set of tasks solved (each task = a "branch")
  - Power schedule: prioritize prompts that solve rare tasks
  - Security Radar: binary fitness (100% verify or dead)

Mathematical foundation (from fuzzingbook.org):
  - Energy(seed) ~ 1/frequency(tasks_solved_by_seed)
  - Seeds that solve rare tasks get more mutation budget
  - Corpus distillation: keep only seeds that add new coverage
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ═══════════════════════════════════════════════════════════════
# Seed corpus — initial prompt templates
# ═══════════════════════════════════════════════════════════════

SEED_PROMPTS = [
    # Direct
    "Here are input/output grid pairs:\n\n{examples}\n"
    "Write: def transform(grid: list[list[int]]) -> list[list[int]]\n"
    "Use numpy. ```python ... ```",

    # Step-by-step
    "Analyze these grid transformations:\n\n{examples}\n"
    "Step 1: What patterns exist?\nStep 2: What changes?\n"
    "Step 3: Write the general rule as:\n"
    "def transform(grid: list[list[int]]) -> list[list[int]]\n"
    "Use numpy. ```python ... ```",

    # Analogies
    "ARC-AGI grid puzzles.\n\n{examples}\n"
    "Operations: crop, tile, flip, rotate, shift, recolor, fill, mask, "
    "scale, reflect, overlay, extract, sort, gravity, flood fill.\n"
    "Write: def transform(grid: list[list[int]]) -> list[list[int]]\n```python```",

    # Concise
    "{examples}\ndef transform(grid):\n    import numpy as np\n"
    "    # implement pattern\nComplete it. ```python```",

    # Expert
    "You are an ARC-AGI expert. Core priors: objectness, geometry, physics.\n\n"
    "{examples}\nIdentify the prior and write:\n"
    "def transform(grid: list[list[int]]) -> list[list[int]]```python```",

    # Reverse-engineer
    "I lost the source code. Here are test cases:\n\n{examples}\n"
    "Reverse-engineer: def transform(grid: list[list[int]]) -> list[list[int]]\n"
    "Use numpy. ```python```",

    # Mathematical
    "Find f: Z^(mxn) -> Z^(pxq), Z={{0..9}}.\n\n{examples}\n"
    "def transform(grid: list[list[int]]) -> list[list[int]]```python```",

    # Visual
    "Visualize colored grids. What moved? Flipped? Grew? Shrank?\n\n{examples}\n"
    "def transform(grid: list[list[int]]) -> list[list[int]]```python```",
]


# ═══════════════════════════════════════════════════════════════
# Mutation operators
# ═══════════════════════════════════════════════════════════════

def mutate_prompt(prompt: str) -> str:
    """Apply a random mutation to a prompt template."""
    mutations = [
        _mut_add_instruction,
        _mut_remove_sentence,
        _mut_swap_sentences,
        _mut_change_tone,
        _mut_add_constraint,
        _mut_simplify,
        _mut_add_color_legend,
        _mut_crossover,
    ]
    return random.choice(mutations)(prompt)


def _mut_add_instruction(p: str) -> str:
    """Add a guiding instruction."""
    additions = [
        "\nFocus on what stays the same between input and output.",
        "\nLook for symmetry, repetition, or translation patterns.",
        "\nConsider objects as connected components of same-colored cells.",
        "\nThe grid size may change. Handle variable dimensions.",
        "\nNon-zero cells form objects. Background is always 0 (black).",
        "\nThink about what a human would see at a glance.",
        "\nThe rule must generalize to any valid input grid.",
        "\nConsider: does each output cell depend on one input cell, or neighbors?",
        "\nCheck if the output is a subgrid, supergrid, or same size as input.",
        "\nColors 0-9 map to: black,blue,red,green,yellow,gray,magenta,orange,lightblue,maroon.",
    ]
    insertion = random.choice(additions)
    # Insert before the function signature line
    if "def transform" in p:
        idx = p.index("def transform")
        return p[:idx] + insertion + "\n" + p[idx:]
    return p + insertion


def _mut_remove_sentence(p: str) -> str:
    """Remove a random non-essential sentence."""
    lines = p.split("\n")
    if len(lines) <= 3:
        return p
    # Don't remove the examples placeholder or function signature
    removable = [i for i, l in enumerate(lines)
                 if l.strip() and "{examples}" not in l and "def transform" not in l
                 and "```" not in l]
    if removable:
        idx = random.choice(removable)
        lines.pop(idx)
    return "\n".join(lines)


def _mut_swap_sentences(p: str) -> str:
    """Swap two adjacent sentences."""
    lines = p.split("\n")
    if len(lines) <= 3:
        return p
    swappable = [i for i in range(len(lines) - 1)
                 if lines[i].strip() and lines[i+1].strip()
                 and "{examples}" not in lines[i] and "{examples}" not in lines[i+1]]
    if swappable:
        idx = random.choice(swappable)
        lines[idx], lines[idx+1] = lines[idx+1], lines[idx]
    return "\n".join(lines)


def _mut_change_tone(p: str) -> str:
    """Replace a phrase with a different tone."""
    replacements = [
        ("Write:", "Implement:"),
        ("Write:", "Code:"),
        ("Write:", "Create:"),
        ("Analyze", "Study"),
        ("Analyze", "Examine"),
        ("patterns", "structures"),
        ("patterns", "regularities"),
        ("Use numpy", "Use only numpy"),
        ("Use numpy", "Implement with numpy"),
        ("grid puzzles", "spatial reasoning tasks"),
        ("grid puzzles", "visual pattern tasks"),
        ("input/output", "before/after"),
        ("transformation", "mapping"),
        ("transformation", "operation"),
    ]
    old, new = random.choice(replacements)
    if old in p:
        return p.replace(old, new, 1)
    return p


def _mut_add_constraint(p: str) -> str:
    """Add an output constraint."""
    constraints = [
        "\nReturn result.tolist() at the end.",
        "\nThe function must be deterministic.",
        "\nHandle edge cases: empty rows, single cells.",
        "\nDo NOT hardcode grid dimensions.",
        "\nKeep the solution under 20 lines.",
        "\nPrefer vectorized numpy over loops.",
    ]
    return p + random.choice(constraints)


def _mut_simplify(p: str) -> str:
    """Remove adjectives and filler words."""
    fillers = ["very ", "really ", "carefully ", "clearly ", "simply ",
               "thoroughly ", "efficiently ", "precisely "]
    for f in fillers:
        if f in p:
            return p.replace(f, "", 1)
    return p


def _mut_add_color_legend(p: str) -> str:
    """Add color mapping info."""
    if "black" in p.lower() or "color" in p.lower():
        return p  # already has it
    legend = ("\nColors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, "
              "5=gray, 6=magenta, 7=orange, 8=lightblue, 9=maroon.")
    if "{examples}" in p:
        return p.replace("{examples}", "{examples}" + legend, 1)
    return p + legend


def _mut_crossover(p: str) -> str:
    """Cross with a random seed prompt."""
    other = random.choice(SEED_PROMPTS)
    # Take first half of p, second half of other
    p_lines = p.split("\n")
    o_lines = other.split("\n")
    mid_p = len(p_lines) // 2
    mid_o = len(o_lines) // 2
    # Ensure we keep {examples} and def transform
    result = "\n".join(p_lines[:mid_p] + o_lines[mid_o:])
    if "{examples}" not in result:
        result = "{examples}\n" + result
    if "def transform" not in result:
        result += "\ndef transform(grid: list[list[int]]) -> list[list[int]]"
    return result


# ═══════════════════════════════════════════════════════════════
# Security Radar
# ═══════════════════════════════════════════════════════════════

def extract_code(response: str) -> str | None:
    if not response:
        return None
    if "```" in response:
        parts = response.split("```")
        for p in parts[1:]:
            if p.startswith("python"):
                p = p[6:]
            if "def transform" in p:
                return p.strip()
    if "def transform" in response:
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if "def transform" in line:
                return "\n".join(lines[i:])
    return None


def verify_transform(transform_fn, task: dict) -> tuple[int, int]:
    """Security Radar: verify on ALL examples."""
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


def format_examples(task: dict) -> str:
    parts = []
    for i, ex in enumerate(task.get("train", [])[:4]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        parts.append(f"Example {i+1}:")
        parts.append(f"  input ({inp.shape[0]}x{inp.shape[1]}): {ex['input']}")
        parts.append(f"  output ({out.shape[0]}x{out.shape[1]}): {ex['output']}")
        parts.append("")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# AFL-style power schedule
# ═══════════════════════════════════════════════════════════════

@dataclass
class Seed:
    prompt: str
    id: str = ""
    tasks_solved: set = field(default_factory=set)
    attempts: int = 0
    energy: float = 1.0

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.prompt.encode()).hexdigest()[:8]


def compute_energy(seed: Seed, all_coverage: dict[int, int]) -> float:
    """AFL-style power schedule.

    Energy is inversely proportional to how often this seed's solved tasks
    have been solved by other seeds. Seeds that solve RARE tasks get more
    mutation budget.

    energy = sum(1/frequency(task)) for each task solved by this seed
    """
    if not seed.tasks_solved:
        return 1.0  # unsolved seeds get baseline energy

    energy = 0.0
    for task in seed.tasks_solved:
        freq = all_coverage.get(task, 1)
        energy += 1.0 / freq

    return max(0.1, energy)


# ═══════════════════════════════════════════════════════════════
# Main fuzzing loop
# ═══════════════════════════════════════════════════════════════

def fuzz(task_nums: list[int],
         model: str = "kimi",
         num_rounds: int = 5,
         mutations_per_round: int = 3,
         output_file: str = "fuzzer_results.json"):
    """Run coverage-guided prompt fuzzing.

    Each round:
      1. Select seed by energy (power schedule)
      2. Mutate it mutations_per_round times
      3. Test each mutation on all tasks
      4. If mutation solves NEW task → add to corpus
      5. Update coverage and energy
    """
    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        token_file = Path.home() / ".llmtoken"
        if token_file.exists():
            token = token_file.read_text().strip()

    from openai import OpenAI
    client = OpenAI(api_key=token, base_url="https://ellm.nrp-nautilus.io/v1")

    extra = {}
    if model == "kimi":
        extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    elif "qwen" in model:
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    # Load tasks
    tasks = {}
    for tn in task_nums:
        tf = ROOT / f"task{tn:03d}.json"
        if tf.exists():
            with open(tf) as f:
                tasks[tn] = json.load(f)

    # Initialize corpus with seed prompts
    corpus = [Seed(prompt=p) for p in SEED_PROMPTS]

    # Track coverage: which tasks have been solved by ANY prompt
    global_coverage = set()  # tasks solved
    task_frequency = {}  # task -> how many prompts solve it
    all_results = []  # log

    print(f"{'='*70}")
    print(f"COVERAGE-GUIDED PROMPT FUZZER")
    print(f"{'='*70}")
    print(f"Tasks: {len(tasks)}")
    print(f"Seeds: {len(corpus)}")
    print(f"Rounds: {num_rounds}")
    print(f"Mutations/round: {mutations_per_round}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    # Phase 0: Evaluate seed corpus
    print("Phase 0: Evaluating seed corpus...")
    for seed in corpus:
        _evaluate_seed(seed, tasks, client, model, extra,
                       global_coverage, task_frequency, all_results)
        print(f"  seed {seed.id}: solved {len(seed.tasks_solved)} tasks "
              f"({sorted(seed.tasks_solved)})", flush=True)

    print(f"\nSeed coverage: {len(global_coverage)}/{len(tasks)} tasks")
    print(f"Covered tasks: {sorted(global_coverage)}\n")

    # Phase 1: Fuzzing rounds
    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num}/{num_rounds} ---", flush=True)

        # Update energy for all seeds
        for seed in corpus:
            seed.energy = compute_energy(seed, task_frequency)

        # Select seed by energy (weighted random)
        total_energy = sum(s.energy for s in corpus)
        if total_energy == 0:
            selected = random.choice(corpus)
        else:
            weights = [s.energy / total_energy for s in corpus]
            selected = random.choices(corpus, weights=weights, k=1)[0]

        print(f"  Selected seed {selected.id} (energy={selected.energy:.2f}, "
              f"solved={len(selected.tasks_solved)})", flush=True)

        # Generate mutations
        for mut_num in range(mutations_per_round):
            mutated_prompt = mutate_prompt(selected.prompt)
            child = Seed(prompt=mutated_prompt)

            _evaluate_seed(child, tasks, client, model, extra,
                           global_coverage, task_frequency, all_results)

            new_tasks = child.tasks_solved - global_coverage
            if new_tasks:
                # New coverage! Add to corpus
                corpus.append(child)
                global_coverage.update(new_tasks)
                print(f"    NEW COVERAGE! +{new_tasks} "
                      f"(total: {len(global_coverage)}/{len(tasks)})", flush=True)
            else:
                solved_count = len(child.tasks_solved)
                print(f"    mutation {mut_num+1}: {solved_count} tasks "
                      f"(no new coverage)", flush=True)

        print(f"  Coverage: {len(global_coverage)}/{len(tasks)} | "
              f"Corpus: {len(corpus)} seeds\n", flush=True)

    # Final report
    print(f"\n{'='*70}")
    print(f"FUZZING COMPLETE")
    print(f"{'='*70}")
    print(f"Total coverage: {len(global_coverage)}/{len(tasks)} tasks "
          f"({len(global_coverage)/len(tasks):.0%})")
    print(f"Corpus size: {len(corpus)} seeds")
    print(f"Tasks solved: {sorted(global_coverage)}")

    # Show best seeds
    print(f"\nTop seeds by tasks solved:")
    ranked = sorted(corpus, key=lambda s: len(s.tasks_solved), reverse=True)
    for s in ranked[:5]:
        print(f"  {s.id}: {len(s.tasks_solved)} tasks, "
              f"energy={s.energy:.2f}")
        print(f"    prompt: {s.prompt[:100]}...")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "num_rounds": num_rounds,
        "total_coverage": len(global_coverage),
        "total_tasks": len(tasks),
        "covered_tasks": sorted(global_coverage),
        "corpus_size": len(corpus),
        "best_seeds": [
            {"id": s.id, "tasks_solved": sorted(s.tasks_solved),
             "prompt": s.prompt}
            for s in ranked[:10]
        ],
        "all_results": all_results,
    }
    output_path = ROOT / output_file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


def _evaluate_seed(seed: Seed, tasks: dict, client, model: str, extra: dict,
                    global_coverage: set, task_frequency: dict,
                    all_results: list):
    """Evaluate a seed prompt on all tasks."""
    for tn, task in tasks.items():
        examples = format_examples(task)
        prompt = seed.prompt.format(examples=examples)

        try:
            r = client.chat.completions.create(
                model=model, max_tokens=2000,
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
                    correct, total = verify_transform(transform_fn, task)
                    verified = (correct == total and total > 0)

                    if verified:
                        seed.tasks_solved.add(tn)
                        task_frequency[tn] = task_frequency.get(tn, 0) + 1

                    all_results.append({
                        "seed_id": seed.id,
                        "task": tn,
                        "verified": verified,
                        "correct": correct,
                        "total": total,
                    })
        except Exception:
            pass

        seed.attempts += 1


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Coverage-guided prompt fuzzer for ARC")
    ap.add_argument("--tasks", type=str, default=None,
                     help="Comma-separated task numbers (default: first 20 unsolved)")
    ap.add_argument("--model", default="kimi")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mutations", type=int, default=3,
                     help="Mutations per round")
    ap.add_argument("--output", default="fuzzer_results.json")
    args = ap.parse_args()

    if args.tasks:
        task_nums = [int(t) for t in args.tasks.split(",")]
    else:
        # Default: first 20 unsolved
        solved = set()
        for d in ["solutions_safe", "solutions_merged_latest"]:
            p = Path(d)
            if p.exists():
                solved.update(int(f.stem[4:]) for f in p.glob("task*.onnx"))
        task_nums = [tn for tn in range(1, 401)
                     if tn not in solved and (ROOT / f"task{tn:03d}.json").exists()][:20]

    fuzz(task_nums, args.model, args.rounds, args.mutations, args.output)


if __name__ == "__main__":
    main()
