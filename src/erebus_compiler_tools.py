"""Tools for Erebus to author and test compiler modules.

The compiler writes ONNX graphs. Each module implements a pattern
(flip, crop_bbox, color_remap, etc.) with the signature:

    def detect_X(task_examples: list[dict]) -> bool
    def compile_X(...) -> onnx.ModelProto   # or make_model(...)

These helpers let Erebus:
 - inspect existing modules as few-shot examples
 - list compiler modules with their detect/compile functions
 - cluster today's failures to decide which pattern to tackle
 - write a candidate module, syntax-check and runtime-test it
 - promote the module only on verified success

Designed to be callable from dream_synthesize_compiler AND from the
ToolExecutor agentic harness in tools.py.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

COMPILER_DIR = Path("/archive/neurogolf/src/compiler")
TASK_DIR = Path("/archive/neurogolf")
MEMORY_PATH = TASK_DIR / "arc_scientist_memory.json"


@dataclass
class ModuleInfo:
    name: str
    path: Path
    docstring: str
    detect_fns: list[str]
    compile_fns: list[str]
    line_count: int


def list_compiler_modules(compiler_dir: Path = COMPILER_DIR) -> list[ModuleInfo]:
    """Return info for every .py module in the compiler directory."""
    if not compiler_dir.exists():
        return []
    infos = []
    for fp in sorted(compiler_dir.glob("*.py")):
        if fp.name == "__init__.py":
            continue
        try:
            src = fp.read_text()
            tree = ast.parse(src)
            doc = (ast.get_docstring(tree) or "").strip().split("\n")[0]
            detects = [n.name for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("detect_")]
            compiles = [n.name for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and
                        (n.name.startswith("compile_") or n.name == "make_model")]
            infos.append(ModuleInfo(
                name=fp.stem, path=fp, docstring=doc,
                detect_fns=detects, compile_fns=compiles,
                line_count=src.count("\n"),
            ))
        except SyntaxError:
            # Skip broken modules (e.g. truncated dream outputs)
            continue
    return infos


def read_compiler_module(name: str,
                         compiler_dir: Path = COMPILER_DIR) -> str | None:
    """Return the full source of a compiler module."""
    fp = compiler_dir / f"{name}.py"
    if not fp.exists():
        return None
    return fp.read_text()


def get_few_shot_modules(compiler_dir: Path = COMPILER_DIR,
                         max_modules: int = 3,
                         max_chars_each: int = 3000) -> str:
    """Render 2-3 real compiler modules as few-shot examples for the LLM.

    Picks the shortest well-formed modules so they fit in context —
    these are the clearest patterns to imitate.
    """
    infos = [m for m in list_compiler_modules(compiler_dir)
             if m.detect_fns or m.compile_fns]
    # Prefer shorter, simpler modules as references
    infos.sort(key=lambda m: m.line_count)
    picks = infos[:max_modules]
    blocks = []
    for m in picks:
        src = m.path.read_text()[:max_chars_each]
        blocks.append(f"# === {m.name}.py ({m.line_count} lines) ===\n{src}")
    return "\n\n".join(blocks)


def cluster_failures(memory_path: Path = MEMORY_PATH,
                     day: str | None = None) -> list[dict]:
    """Group the day's failures by (error_type, similar_to).

    Returns clusters sorted by size — the biggest cluster is where a new
    compiler module would have the most impact.
    """
    if not memory_path.exists():
        return []
    mem = json.loads(memory_path.read_text())
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for tn_str, tk in mem.get("tasks", {}).items():
        for a in tk.get("attempts", []):
            if a.get("verified"):
                continue
            if day and not a.get("timestamp", "").startswith(day):
                continue
            key = (a.get("error_type", "unknown"),
                   a.get("similar_to", "") or "unclassified")
            buckets[key].append(int(tn_str))
    clusters = []
    for (et, pattern), tasks in buckets.items():
        uniq = sorted(set(tasks))
        clusters.append({
            "error_type": et,
            "pattern": pattern,
            "n_failures": len(tasks),
            "n_unique_tasks": len(uniq),
            "tasks": uniq[:20],
        })
    clusters.sort(key=lambda c: -c["n_unique_tasks"])
    return clusters


def syntax_check_module(code: str) -> tuple[bool, str]:
    """Parse the candidate module. Returns (ok, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def import_check_module(code: str) -> tuple[bool, str, list[str]]:
    """Write to temp, exec it, return (ok, error, defined_functions).

    Ignores missing imports — this is a smoke test, not a full build.
    """
    try:
        ns: dict = {}
        exec(code, ns)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", []
    fns = [k for k, v in ns.items()
           if callable(v) and (k.startswith(("detect_", "compile_", "make_")))]
    return True, "", fns


def test_compile_against_tasks(code: str, task_nums: list[int],
                               task_dir: Path = TASK_DIR) -> dict:
    """Build the ONNX model with the candidate module and run it on each task.

    Runs in an isolated subprocess so a crash doesn't kill the caller.
    Returns per-task verification counts.
    """
    script = _make_test_harness(code, task_nums, task_dir)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, "-u", tmp],
                           capture_output=True, text=True, timeout=120)
        # Last line is the JSON result
        lines = [ln for ln in r.stdout.strip().split("\n") if ln.startswith("{")]
        if not lines:
            return {"ok": False, "error": r.stderr[-500:] or "no output",
                    "per_task": {}}
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "per_task": {}}
    finally:
        Path(tmp).unlink(missing_ok=True)


def _make_test_harness(code: str, task_nums: list[int],
                       task_dir: Path) -> str:
    """Build a self-contained test script that exec's the candidate, finds
    compile_X/make_model, builds the ONNX graph, runs it, compares."""
    return f"""
import json, sys, traceback
from pathlib import Path
import numpy as np

# Candidate module source
_SRC = {code!r}

ns = {{}}
try:
    exec(_SRC, ns)
except Exception as e:
    print(json.dumps({{"ok": False, "error": f"exec: {{e}}", "per_task": {{}}}}))
    sys.exit(0)

# Find compile function and detector
compile_fn = next((ns[k] for k in ns
                   if callable(ns[k]) and k.startswith("compile_")), None)
detect_fn  = next((ns[k] for k in ns
                   if callable(ns[k]) and k.startswith("detect_")), None)
make_fn    = ns.get("make_model")

if not (compile_fn or make_fn):
    print(json.dumps({{"ok": False, "error": "no compile_X or make_model found",
                      "per_task": {{}}}}))
    sys.exit(0)

try:
    import onnxruntime as ort
except ImportError:
    print(json.dumps({{"ok": False, "error": "onnxruntime not installed",
                      "per_task": {{}}}}))
    sys.exit(0)

task_dir = Path({str(task_dir)!r})
results = {{}}
task_nums = {task_nums!r}
for tn in task_nums:
    tf = task_dir / f"task{{tn:03d}}.json"
    if not tf.exists():
        results[str(tn)] = {{"error": "task_not_found"}}
        continue
    try:
        task = json.loads(tf.read_text())
        # Get the model
        try:
            model = (compile_fn or make_fn)()
        except TypeError:
            # Some compile_fns need task or examples
            model = (compile_fn or make_fn)(task.get("train", []))

        sess = ort.InferenceSession(model.SerializeToString(),
                                     providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        correct = total = 0
        for split in ("train", "test"):
            for ex in task.get(split, []):
                total += 1
                arr = np.array(ex["input"], dtype=np.int64)
                try:
                    out = sess.run(None, {{input_name: arr}})[0]
                    if out.tolist() == ex["output"]:
                        correct += 1
                except Exception:
                    pass
        results[str(tn)] = {{"correct": correct, "total": total}}
    except Exception as e:
        results[str(tn)] = {{"error": f"{{type(e).__name__}}: {{e}}"}}

solved = sum(1 for v in results.values()
             if v.get("correct") and v.get("correct") == v.get("total"))
print(json.dumps({{"ok": True, "n_tasks": len(task_nums),
                  "n_solved": solved, "per_task": results}}))
"""


def promote_candidate(code: str, tag: str,
                      compiler_dir: Path = COMPILER_DIR,
                      overwrite: bool = False) -> Path:
    """Write candidate source to compiler_dir/dream_TAG.py.

    Raises if file exists and overwrite=False.
    """
    compiler_dir.mkdir(parents=True, exist_ok=True)
    path = compiler_dir / f"dream_{tag}.py"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.write_text(code)
    return path


def write_compiler_module(code: str, test_task_nums: list[int],
                          tag: str, min_solved_ratio: float = 0.5,
                          compiler_dir: Path = COMPILER_DIR,
                          task_dir: Path = TASK_DIR) -> dict:
    """Full pipeline: syntax-check → import-check → runtime-test → promote.

    Only saves the module if at least min_solved_ratio of the test tasks
    pass. Returns a dict describing what happened at each stage.
    """
    result: dict = {"tag": tag, "stages": []}

    ok, err = syntax_check_module(code)
    result["stages"].append({"stage": "syntax", "ok": ok, "error": err})
    if not ok:
        return result

    ok, err, fns = import_check_module(code)
    result["stages"].append({"stage": "import", "ok": ok, "error": err,
                             "functions": fns})
    if not ok:
        return result

    if test_task_nums:
        test = test_compile_against_tasks(code, test_task_nums, task_dir)
        result["stages"].append({"stage": "runtime_test", **test})
        if not test.get("ok"):
            return result
        ratio = test["n_solved"] / max(test["n_tasks"], 1)
        result["solved_ratio"] = ratio
        if ratio < min_solved_ratio:
            result["promoted"] = False
            result["reason"] = f"solved_ratio {ratio:.0%} < threshold {min_solved_ratio:.0%}"
            return result

    path = promote_candidate(code, tag, compiler_dir, overwrite=True)
    result["promoted"] = True
    result["path"] = str(path)
    return result
