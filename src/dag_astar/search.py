"""A* search loop over DAG topologies.

Flow per task:
  1. Enumerate expansion rules (allowed op-at-depth-D combinations)
  2. Priority-queue A*: f = g + h, expand cheapest first
  3. For each "complete" candidate (produces canvas-shaped output),
     run weight training if parameterized, then verify
  4. First verified match wins (cheapest due to A* ordering)

Expansion rule set (Week 2 initial — keep narrow to make search tractable):
  Depth 1: {conv 1x1, conv 3x3, conv 5x5, conv 7x7,
            gather-identity, gather-rot90cw/ccw, gather-rot180,
            gather-flip-h, gather-flip-w, gather-transpose}
  Depth 2: {relu + anything from depth 1 that takes FLOAT (1,10,30,30)}
  Depth 3: {conv 1x1 after depth-2 (color remap on top of spatial op)}

Total distinct topologies at depth ≤ 3: ~50-100, tractable.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .graph import DAGState
from .heuristic import HeuristicDAG
from .tensor import C, H, W, CANVAS_SHAPE
from .weight_opt import train_weights, collect_tensors


# -----------------------------------------------------------------------
# Expansion rule set (initial Week 2 version)
# -----------------------------------------------------------------------

def expand_canvas_to_flat(state: DAGState):
    """If last tensor is (1,C,H,W), reshape it to (1,C,H*W) for Gather operations."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT))
    yield ("reshape", {"target_shape": [1, C, H * W]}, [state.last_tensor],
           "reshape_flat")


def expand_flat_with_gather(state: DAGState, grid_h=None, grid_w=None):
    """If last tensor is (1,C,H*W), offer Gather with affine-transform indices.

    Tries rotations/flips/transposes for MULTIPLE possible grid sizes
    (since the task's effective grid may be smaller than the 30x30 canvas).
    """
    import sys
    from pathlib import Path
    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT))
    from grammar.primitives import _affine_indices

    x = state.last_tensor
    if x.shape != (1, C, H * W):
        return

    # If specific grid size provided, use only that; else try common sizes
    if grid_h is not None and grid_w is not None:
        size_candidates = [(grid_h, grid_w)]
    else:
        size_candidates = [(H, W)] + [(k, k) for k in [3, 4, 5, 6, 7, 8, 9, 10]]

    seen = set()
    for gh, gw in size_candidates:
        transforms = {
            "rot90cw": (np.array([[0, -1], [1, 0]], dtype=np.float64),
                         np.array([gh - 1, 0], dtype=np.float64)),
            "rot90ccw": (np.array([[0, 1], [-1, 0]], dtype=np.float64),
                          np.array([0, gw - 1], dtype=np.float64)),
            "rot180": (np.array([[-1, 0], [0, -1]], dtype=np.float64),
                        np.array([gh - 1, gw - 1], dtype=np.float64)),
            "flip_h": (np.array([[-1, 0], [0, 1]], dtype=np.float64),
                        np.array([gh - 1, 0], dtype=np.float64)),
            "flip_w": (np.array([[1, 0], [0, -1]], dtype=np.float64),
                        np.array([0, gw - 1], dtype=np.float64)),
            "transpose": (np.array([[0, 1], [1, 0]], dtype=np.float64),
                           np.zeros(2, dtype=np.float64)),
        }
        for name, (A, b) in transforms.items():
            indices = _affine_indices(A, b, gh, gw)
            key = (name, indices.tobytes())
            if key in seen:
                continue
            seen.add(key)
            yield ("gather", {"indices": indices, "axis": 2}, [x],
                   f"gather({name}@{gh}x{gw})")


def expand_flat_to_canvas(state: DAGState):
    """Reshape from flat (1,C,H*W) back to canvas (1,C,H,W)."""
    if state.last_tensor.shape == (1, C, H * W):
        yield ("reshape", {"target_shape": [1, C, H, W]}, [state.last_tensor],
               "reshape_canvas")


def expansions_for_canvas_shape(state: DAGState, grid_h=None, grid_w=None):
    """Generate successor states from `state` where the last tensor is canvas-shaped."""
    x = state.last_tensor

    if x.shape != CANVAS_SHAPE:
        # Dispatch to shape-specific expanders
        if x.shape == (1, C, H * W):
            # Prefer the task's detected grid size (cuts fanout 8x) but fall back
            # to trying a few common sizes so search still works without detection.
            yield from expand_flat_with_gather(state, grid_h, grid_w)
            yield from expand_flat_to_canvas(state)
        elif len(x.shape) == 4 and x.shape[0] == 1 and x.shape[1] == C \
             and (x.shape[2] < H or x.shape[3] < W):
            # Sub-canvas: offer Pad to bring back to (1,C,H,W) with zero fill
            gh, gw = x.shape[2], x.shape[3]
            pad_h_end = H - gh
            pad_w_end = W - gw
            pads = [0, 0, 0, 0, 0, 0, pad_h_end, pad_w_end]
            yield ("pad", {"pads": pads, "value": 0.0}, [x],
                   f"pad_to_canvas_from_{gh}x{gw}")
        return

    # Canvas-shaped: offer Conv variants, ReLU, Reshape-to-flat, Slice-and-pad
    for k in (1, 3, 5):
        w = (np.random.RandomState(42 + k).randn(C, C, k, k) * 0.1).astype(np.float32)
        yield ("conv", {"weights": w}, [x], f"conv_{k}x{k}")
    yield ("relu", {}, [x], "relu")
    yield from expand_canvas_to_flat(state)

    # Slice + pad patterns: extract a sub-grid at the canvas origin and pad zeros.
    # Useful when the effective task operates on a small grid embedded in a larger one.
    # Only emit for grid sizes that a task might plausibly use.
    if grid_h is not None and grid_w is not None and (grid_h < H or grid_w < W):
        yield ("slice", {
            "starts": [0, 0, 0, 0],
            "ends": [1, C, grid_h, grid_w],
            "axes": [0, 1, 2, 3],
        }, [x], f"slice(0,0,{grid_h},{grid_w})")


# -----------------------------------------------------------------------
# Search loop
# -----------------------------------------------------------------------

@dataclass(order=True)
class _Node:
    priority: int
    counter: int
    state: DAGState = field(compare=False)


def _detect_grid_size(task):
    """Detect consistent (grid_h, grid_w) across examples, or None."""
    sizes = set()
    for sub in ('train', 'test', 'arc-gen'):
        for ex in task.get(sub, []):
            ih = len(ex['input'])
            iw = len(ex['input'][0]) if ex['input'] else 0
            oh = len(ex['output'])
            ow = len(ex['output'][0]) if ex['output'] else 0
            sizes.add((ih, iw, oh, ow))
    if len(sizes) == 1:
        ih, iw, oh, ow = sizes.pop()
        return ih, iw, oh, ow
    return None


def astar_solve_task(task: dict, task_num: int, device: str = 'cpu',
                     time_budget_s: float = 60.0,
                     max_depth: int = 4,
                     max_expansions: int = 200,
                     verbose: bool = True):
    """A* search for the cheapest DAG that solves task.

    Returns (DAGState, score_dict) on success, or (None, None).
    """
    X, Y = collect_tensors(task, device)
    if X is None:
        return None, None

    # Detect grid size once per task
    sizes = _detect_grid_size(task)
    if sizes:
        grid_h, grid_w = sizes[0], sizes[1]  # use input size; same as output for same-shape tasks
        # Only use detected size if both input and output match that size
        # (otherwise shape-changing transforms need all candidates)
        if sizes[0] != sizes[2] or sizes[1] != sizes[3]:
            grid_h, grid_w = None, None
    else:
        grid_h, grid_w = None, None

    heuristic = HeuristicDAG()
    t0 = time.time()

    start = DAGState.initial()

    # Fast path: if input==output, a bare Identity op is the cheapest solution.
    if torch.all(X == Y).item():
        id_state = start.extend('identity', [start.last_tensor], {})
        success, trained = train_weights(id_state, X, Y, device=device, steps=1, lr=0.01)
        if success:
            import math as _math
            cost = trained.total_cost
            score = max(1.0, 25.0 - _math.log(max(cost, 1)))
            if verbose:
                print(f"  task{task_num:03d}: identity cost={cost}", flush=True)
            return trained, {
                'cost': cost, 'score': score,
                'ops': list(trained.op_labels),
                'solve_time': time.time() - t0, 'expansions': 0,
            }

    open_set: list[_Node] = []
    closed = set()
    counter = 0

    def push(state: DAGState):
        nonlocal counter
        counter += 1
        f = heuristic.f_cost(state)
        heapq.heappush(open_set, _Node(priority=f, counter=counter, state=state))

    push(start)
    best = None
    best_cost = float('inf')
    expansions = 0

    while open_set and expansions < max_expansions:
        if time.time() - t0 > time_budget_s:
            break

        node = heapq.heappop(open_set)
        state = node.state

        # Dedup
        key = state.canonical_key()
        if key in closed:
            continue
        closed.add(key)

        # If f ≥ best, no point continuing
        if node.priority >= best_cost:
            continue

        # Candidate terminal check: if last tensor is canvas-shaped FP32,
        # this might already be a solution
        if state.last_tensor.shape == CANVAS_SHAPE and state.depth > 0:
            # Try to train weights (if any) and verify
            success, trained = train_weights(state, X, Y, device=device,
                                               steps=400, lr=0.05)
            if success:
                cost = trained.total_cost
                if cost < best_cost:
                    best_cost = cost
                    best = trained
                    if verbose:
                        print(f"  task{task_num:03d}: {' -> '.join(trained.op_labels)} "
                              f"cost={cost}", flush=True)
                continue  # don't expand further from a solution

        # Expand only if within depth budget
        if state.depth >= max_depth:
            continue

        for op_name, hp, inputs, label in expansions_for_canvas_shape(state, grid_h, grid_w):
            try:
                child = state.extend(op_name, inputs, hp)
            except Exception:
                continue
            expansions += 1
            if expansions >= max_expansions:
                break
            push(child)

    if best:
        import math
        score = max(1.0, 25.0 - math.log(best.total_cost))
        return best, {
            'cost': best.total_cost,
            'score': score,
            'ops': list(best.op_labels),
            'solve_time': time.time() - t0,
            'expansions': expansions,
        }
    return None, None
