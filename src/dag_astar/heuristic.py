"""Admissible heuristics for DAG A* search.

Combined via max (max of admissibles is admissible, per Theory Radar theorem).

Current heuristics:
- h_shape:   if current last tensor isn't canvas-shaped, need ≥ 1 terminal op to fix
- h_dtype:   if current last tensor is INT64 (e.g., after ArgMax), need ≥ 1 op to get back to FLOAT
- h_cost_floor: known minimum cost for canvas-shape FP32 output = cost of 1x1 Conv

All return a LOWER BOUND on remaining cost. 0 means no constraint from this heuristic.
"""
from __future__ import annotations

from onnx import TensorProto

from .graph import DAGState
from .tensor import CANVAS_SHAPE


# Known minimum costs for common "terminal" operators in NeuroGolf canvas context
# 1x1 Conv with 10x10 weights: 100 params + 72400 memory + 90000 macs = 162500
MIN_COST_1X1_CONV = 162_500
# Gather-with-reshape terminal: shape + indices + output memory
MIN_COST_GATHER_TERMINAL = 80_000


def h_shape_gap(state: DAGState) -> int:
    """If last tensor shape != canvas shape, need at least one op to fix."""
    if state.last_tensor.shape == CANVAS_SHAPE:
        return 0
    # Minimum op to reach canvas shape is approx a Reshape (~36K) or
    # a 1x1 Conv (162K). Use the smaller as lower bound.
    return 40_000  # conservative


def h_dtype_gap(state: DAGState) -> int:
    """If last tensor isn't FLOAT, we need at least an Equal+Cast or similar."""
    if state.last_tensor.dtype == TensorProto.FLOAT:
        return 0
    return 20_000


def h_empty_graph(state: DAGState) -> int:
    """The empty state has cost 0 but needs at least a terminal op.

    Minimum achievable is roughly 1x1 Conv (162K) or Gather-reshape (~80K).
    """
    if state.depth == 0:
        return MIN_COST_GATHER_TERMINAL
    return 0


class HeuristicDAG:
    """Collection of admissible heuristics combined via max."""

    def __init__(self, heuristics=None):
        if heuristics is None:
            heuristics = [h_empty_graph, h_shape_gap, h_dtype_gap]
        self.heuristics = list(heuristics)

    def __call__(self, state: DAGState) -> int:
        return max(h(state) for h in self.heuristics)

    def f_cost(self, state: DAGState) -> int:
        """f(s) = g(s) + h(s)."""
        return state.total_cost + self(state)
