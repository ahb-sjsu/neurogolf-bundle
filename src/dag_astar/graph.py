"""Partial DAG state for A* search.

A GraphState is an immutable snapshot of the partial graph being built.
Key operations:
- extend(op_name, input_tensor_ids, hyperparams) → new GraphState
- build_model(output_tensor) → onnx.ModelProto
- hash() and == for closed-set dedup in A*
- cost: total accumulated cost so far (g in A*)
- available_tensors: list of tensors that can be used as inputs to new ops

Design: immutable via "copy-on-extend". Each extend creates a new state
with the new node/initializers appended. Tensor refs are immutable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import onnx
from onnx import TensorProto, helper

from .ops import OP_REGISTRY, OpCost, Emitted
from .tensor import C, H, W, CANVAS_SHAPE, TensorRef, canvas_tensor


@dataclass(frozen=True)
class DAGState:
    """Immutable partial graph."""
    tensors: tuple = ()           # tuple[TensorRef, ...]  all tensors in scope
    nodes: tuple = ()             # tuple[onnx.NodeProto, ...]
    initializers: tuple = ()      # tuple[onnx.TensorProto, ...]
    total_cost: int = 0           # g(state) accumulated OpCost.total
    op_labels: tuple = ()         # tuple[str, ...]  for readability / hashing

    @classmethod
    def initial(cls) -> "DAGState":
        """Initial state: just the canvas input, nothing built."""
        return cls(
            tensors=(canvas_tensor(),),
            nodes=(),
            initializers=(),
            total_cost=0,
            op_labels=(),
        )

    @property
    def depth(self) -> int:
        return len(self.op_labels)

    @property
    def last_tensor(self) -> TensorRef:
        return self.tensors[-1]

    def extend(self, op_name: str, input_tensors: list[TensorRef],
               hp: dict, name_prefix: str = "") -> "DAGState":
        """Return a new state with one more operator applied."""
        if op_name not in OP_REGISTRY:
            raise ValueError(f"unknown op: {op_name}")
        builder = OP_REGISTRY[op_name]

        # Name generator for the new op
        counter = [0]
        def name_gen(base: str) -> str:
            counter[0] += 1
            return f"{name_prefix}{base}_{self.depth + 1}_{counter[0]}"

        em: Emitted = builder(input_tensors, hp, name_gen)

        return DAGState(
            tensors=self.tensors + (em.output,),
            nodes=self.nodes + tuple(em.nodes),
            initializers=self.initializers + tuple(em.initializers),
            total_cost=self.total_cost + em.cost.total,
            op_labels=self.op_labels + (em.label,),
        )

    def build_model(self, output_tensor: Optional[TensorRef] = None,
                     ir_version: int = 10, opset: int = 10) -> onnx.ModelProto:
        """Emit an ONNX ModelProto.

        output_tensor: which tensor to expose as graph output (defaults to last).
        """
        if output_tensor is None:
            output_tensor = self.last_tensor

        # Check output shape matches the canvas (we produce 1x10x30x30 outputs)
        in_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, list(CANVAS_SHAPE))

        # If last tensor isn't already named "output", add an Identity node
        nodes = list(self.nodes)
        if output_tensor.name != "output":
            nodes.append(helper.make_node(
                "Identity", [output_tensor.name], ["output"]))

        # Also need value_info for all intermediate tensors EXCEPT input
        value_info = []
        for t in self.tensors:
            if t.name == "input":
                continue
            value_info.append(helper.make_tensor_value_info(
                t.name, t.dtype, list(t.shape)))

        out_info = helper.make_tensor_value_info(
            "output", output_tensor.dtype, list(output_tensor.shape))

        graph = helper.make_graph(
            nodes, "dag_astar_graph",
            [in_info], [out_info],
            initializer=list(self.initializers),
            value_info=value_info,
        )
        return helper.make_model(
            graph, ir_version=ir_version,
            opset_imports=[helper.make_opsetid("", opset)],
        )

    def canonical_key(self) -> str:
        """Deterministic key for closed-set dedup.

        Uses the op_labels sequence + tensor shapes.
        """
        parts = []
        for t in self.tensors[1:]:  # skip input
            parts.append(f"{t.dtype}:{t.shape}")
        parts.extend(self.op_labels)
        s = "|".join(parts)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
