"""Full ONNX DAG A* solver (proper Theory Radar for NeuroGolf).

State = partial ONNX computation graph (DAG of operators).
Actions = add one operator node producing a new tensor.
Goal = graph whose last-produced tensor matches all task outputs.
g(s) = current cost (params + memory + MACs) of partial graph.
h(s) = admissible lower bound on cost-to-completion.

Design principles:
- Every tensor has a statically-known shape and dtype (ONNX opset 10 constraint)
- Operators are typed: know input arity, output shape given input shapes
- Learnable operators (Conv, MatMul) emit placeholder weights; gradient descent
  optimizes them after topology is fixed
- Admissible heuristics from multiple angles (combined via max)
- Monotone failure pruning: if a topology failed, supersets that only add
  capacity are unlikely to succeed

Modules:
- tensor: TensorRef (shape, dtype, source node)
- ops: operator catalog with type signatures and ONNX emitters
- graph: DAG state, extension, emission to ONNX
- heuristic: admissible lower bounds
- search: priority-queue A* loop
"""
