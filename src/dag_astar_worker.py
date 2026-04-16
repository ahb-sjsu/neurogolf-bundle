"""Stateless DAG A* worker for atlas-burst / NRP distribution.

Reads a job descriptor from env var NEUROGOLF_JOB (JSON):
    {
      "task_nums": [1, 2, 3, ...],       # tasks to solve
      "tasks_url": "https://.../",        # optional: base URL for fetching tasks
      "time_budget_s": 60,
      "max_expansions": 3000,
      "max_depth": 4,
      "results_sink": "stdout"            # or "file:/path" or "http://..."
    }

For each task, runs astar_solve_task and emits a JSON record per solved
task to the results_sink. Exits 0 on success.

Design constraints:
- No persistent state required between tasks
- Reads task JSON from either local file task{NNN}.json OR HTTP GET tasks_url+task{NNN}.json
- Emits ONNX model as base64 in each result record
- Small resource footprint: ~500MB RAM, 1 CPU, no GPU needed
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
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import onnx

from dag_astar.search import astar_solve_task
from grammar.primitives import score_model


def fetch_task(task_num: int, tasks_url: Optional[str] = None) -> Optional[dict]:
    """Fetch task JSON either locally or via HTTP GET."""
    if tasks_url:
        import urllib.request
        url = tasks_url.rstrip("/") + f"/task{task_num:03d}.json"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return None

    # Fallback: look for local file
    for candidate in (ROOT / f"task{task_num:03d}.json",
                       Path("/data") / f"task{task_num:03d}.json",
                       Path("/data/tasks") / f"task{task_num:03d}.json"):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    return None


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def solve_one(task_num: int, task: dict, time_budget_s: float,
              max_expansions: int, max_depth: int) -> dict:
    """Run A* on one task, return a result record."""
    t0 = time.time()
    device = os.environ.get("NEUROGOLF_DEVICE") or _auto_device()
    try:
        state, info = astar_solve_task(
            task, task_num, device=device,
            time_budget_s=time_budget_s,
            max_depth=max_depth,
            max_expansions=max_expansions,
            verbose=False,
        )
    except Exception as e:
        return {"task": task_num, "status": "error",
                "error": str(e)[:500], "elapsed": time.time() - t0}

    if state is None:
        return {"task": task_num, "status": "unsolved",
                "elapsed": time.time() - t0}

    # Serialize ONNX model as base64 so it can be shipped over any channel
    model = state.build_model()
    real = score_model(model)
    if real is None:
        return {"task": task_num, "status": "score_error",
                "elapsed": time.time() - t0}

    buf = io.BytesIO()
    onnx.save(model, buf)
    model_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "task": task_num,
        "status": "solved",
        "cost": real["cost"],
        "score": round(real["score"], 3),
        "ops": info["ops"],
        "elapsed": round(time.time() - t0, 2),
        "expansions": info["expansions"],
        "model_b64": model_b64,
    }


def emit(rec: dict, sink: str):
    """Write one record to the configured sink."""
    line = json.dumps(rec) + "\n"
    if sink == "stdout":
        sys.stdout.write(line)
        sys.stdout.flush()
    elif sink.startswith("file:"):
        path = sink[5:]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line)
    elif sink.startswith("http://") or sink.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(sink, data=line.encode(),
                                       headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            # Best effort — fall back to stdout so we don't lose the record
            sys.stdout.write(line)
            sys.stdout.flush()
    else:
        raise ValueError(f"unknown sink: {sink}")


def load_job() -> dict:
    """Parse the job descriptor from env or command-line."""
    raw = os.environ.get("NEUROGOLF_JOB")
    if raw:
        return json.loads(raw)

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-nums", type=str, required=True,
                     help="comma-separated, e.g. '1,2,3'")
    ap.add_argument("--tasks-url", default=None)
    ap.add_argument("--time-budget-s", type=float, default=60.0)
    ap.add_argument("--max-expansions", type=int, default=3000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--results-sink", default="stdout")
    args = ap.parse_args()

    return {
        "task_nums": [int(t) for t in args.task_nums.split(",") if t.strip()],
        "tasks_url": args.tasks_url,
        "time_budget_s": args.time_budget_s,
        "max_expansions": args.max_expansions,
        "max_depth": args.max_depth,
        "results_sink": args.results_sink,
    }


def main():
    job = load_job()
    sink = job.get("results_sink", "stdout")
    for tn in job["task_nums"]:
        task = fetch_task(tn, job.get("tasks_url"))
        if task is None:
            emit({"task": tn, "status": "missing"}, sink)
            continue
        rec = solve_one(tn, task,
                         float(job.get("time_budget_s", 60.0)),
                         int(job.get("max_expansions", 3000)),
                         int(job.get("max_depth", 4)))
        emit(rec, sink)


if __name__ == "__main__":
    main()
