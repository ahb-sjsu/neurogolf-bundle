"""Tensor reference types for DAG A* search.

A TensorRef is a symbolic handle to a tensor produced by some operator.
Each has:
- name: unique identifier in the DAG
- shape: statically-known tuple of ints
- dtype: ONNX TensorProto type (FLOAT, INT64, BOOL)
- source: Node that produced it (or None for graph input)

The DAG is immutable: adding an operator returns a NEW state (copy-on-write).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from onnx import TensorProto

# Canvas dimensions
C = 10
H = 30
W = 30
CANVAS_SHAPE = (1, C, H, W)


@dataclass(frozen=True)
class TensorRef:
    """Immutable reference to a tensor in the partial graph."""
    name: str
    shape: tuple[int, ...]
    dtype: int = TensorProto.FLOAT
    # For debugging only — which operator produced this tensor
    source_op: Optional[str] = None

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def bytes_fp32(self) -> int:
        return self.num_elements * 4

    def __repr__(self) -> str:
        return f"T({self.name}:{self.shape})"


def canvas_tensor(name: str = "input") -> TensorRef:
    """The standard NeuroGolf canvas tensor: (1, 10, 30, 30) fp32."""
    return TensorRef(name=name, shape=CANVAS_SHAPE, dtype=TensorProto.FLOAT,
                     source_op="input")
