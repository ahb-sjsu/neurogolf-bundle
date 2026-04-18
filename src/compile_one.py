"""Compile one task: Python transform → ONNX via LLM translation.

Given a task number and its verified Python transform, asks the LLM
to translate it into equivalent onnx.helper code using known patterns.

Includes working ONNX examples as few-shot context so the LLM
knows the idioms: Gather for spatial remap, Conv 1x1 for color remap,
ReduceMax+ArgMax for grid detection, Clip trick for masking.

Usage: TASK=42 MODEL=kimi NRP_LLM_TOKEN=xxx python compile_one.py
"""
import io
import json
import os
import sys

import numpy as np

TASK_NUM = int(os.environ.get("TASK", "1"))
MODEL = os.environ.get("MODEL", "kimi")
TOKEN = os.environ.get("NRP_LLM_TOKEN", "")
TASK_DIR = os.environ.get("TASK_DIR", "..")
TRANSFORM_CODE = os.environ.get("TRANSFORM_CODE", "")

# If no inline code, try loading from file
if not TRANSFORM_CODE:
    tf = f"{TASK_DIR}/solutions_erebus_transforms/task{TASK_NUM:03d}.py"
    if os.path.exists(tf):
        TRANSFORM_CODE = open(tf).read()

if not TRANSFORM_CODE:
    print(json.dumps({"task": TASK_NUM, "status": "no_transform"}))
    sys.exit(0)

# Load task
task_file = f"{TASK_DIR}/task{TASK_NUM:03d}.json"
if not os.path.exists(task_file):
    print(json.dumps({"task": TASK_NUM, "status": "missing"}))
    sys.exit(0)

with open(task_file) as f:
    task = json.load(f)

# Format examples
examples = ""
for i, ex in enumerate(task.get("train", [])[:3]):
    examples += f"Example {i+1}:\n"
    examples += f"  input ({len(ex['input'])}x{len(ex['input'][0])}): {ex['input']}\n"
    examples += f"  output ({len(ex['output'])}x{len(ex['output'][0])}): {ex['output']}\n\n"

# Working ONNX pattern examples (the curriculum)
ONNX_EXAMPLES = '''
## EXAMPLE 1: Color remap via 1x1 Conv
# If output = input with colors substituted (e.g., 2->4, 3->6)
def build_onnx_color_remap():
    from onnx import TensorProto, helper
    import numpy as np
    W = np.zeros((10, 10, 1, 1), dtype=np.float32)
    # mapping: src_channel -> dst_channel
    mapping = {0:0, 2:4, 3:6}  # example
    for src in range(10):
        dst = mapping.get(src, src)
        W[dst, src, 0, 0] = 1.0
    nodes = [helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1,1])]
    inits = [helper.make_tensor("W", TensorProto.FLOAT, [10,10,1,1], W.flatten().tolist())]
    in_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])
    out_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,10,30,30])
    graph = helper.make_graph(nodes, "g", [in_info], [out_info], initializer=inits)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("",10)])

## EXAMPLE 2: Spatial remap via Gather (flip, crop, rotate, transpose)
# If output[r,c] = input[f(r,c)] for some index function f
def build_onnx_spatial_remap():
    from onnx import TensorProto, helper
    import numpy as np
    H, W, C = 30, 30, 10
    # Compute flat source indices: for each output position, where to read from
    indices = np.zeros(H*W, dtype=np.int64)
    for r in range(H):
        for c in range(W):
            src_r, src_c = H-1-r, c  # example: vertical flip
            indices[r*W+c] = src_r*W + src_c
    nodes = []
    inits = [
        helper.make_tensor("idx", TensorProto.INT64, [H*W], indices.tolist()),
        helper.make_tensor("s1", TensorProto.INT64, [3], [1,C,H*W]),
        helper.make_tensor("s2", TensorProto.INT64, [4], [1,C,H,W]),
    ]
    nodes.append(helper.make_node("Reshape", ["input","s1"], ["flat"]))
    nodes.append(helper.make_node("Gather", ["flat","idx"], ["gathered"], axis=2))
    nodes.append(helper.make_node("Reshape", ["gathered","s2"], ["output"]))
    in_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,C,H,W])
    out_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,H,W])
    graph = helper.make_graph(nodes, "g", [in_info], [out_info], initializer=inits)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("",10)])

## EXAMPLE 3: Conv network for learned transforms
# If the pattern is too complex for static compilation, use a small trained Conv
def build_onnx_conv_trained():
    from onnx import TensorProto, helper
    import numpy as np
    # Conv3x3 + Relu + Conv1x1 (trainable weights - set to identity as placeholder)
    W1 = np.random.randn(10,10,3,3).astype(np.float32) * 0.1
    W2 = np.eye(10, dtype=np.float32).reshape(10,10,1,1)
    nodes = [
        helper.make_node("Conv", ["input","W1"], ["c1"], kernel_shape=[3,3], pads=[1,1,1,1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1","W2"], ["output"], kernel_shape=[1,1]),
    ]
    inits = [
        helper.make_tensor("W1", TensorProto.FLOAT, [10,10,3,3], W1.flatten().tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, [10,10,1,1], W2.flatten().tolist()),
    ]
    in_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,10,30,30])
    out_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,10,30,30])
    graph = helper.make_graph(nodes, "g", [in_info], [out_info], initializer=inits)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("",10)])
'''

prompt = f"""This Python transform correctly solves an ARC-AGI task:

```python
{TRANSFORM_CODE}
```

Task examples:
{examples}

Convert this Python transform into an equivalent ONNX model.

{ONNX_EXAMPLES}

CONSTRAINTS:
- Input: 'input' shape [1,10,30,30] float32 one-hot (channel c = 1.0 where color=c)
- Output: 'output' shape [1,10,30,30] float32 one-hot
- Opset 10, ir_version 10
- The ONNX model must produce IDENTICAL output to the Python transform

Choose the simplest pattern that works:
1. Color remap (1x1 Conv) - if only colors change
2. Spatial remap (Gather) - if pixels move but colors stay
3. Conv network - if the pattern is learned/complex

Write ONLY the build_onnx() function. ```python ... ```"""

# Call LLM
try:
    from openai import OpenAI
    client = OpenAI(api_key=TOKEN, base_url="https://ellm.nrp-nautilus.io/v1")
    extra = {}
    if MODEL == "kimi":
        extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    elif "qwen" in MODEL:
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    r = client.chat.completions.create(
        model=MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    response = r.choices[0].message.content or ""
except Exception as e:
    print(json.dumps({"task": TASK_NUM, "status": "llm_error", "error": str(e)[:100]}))
    sys.exit(0)

# Extract build_onnx
code = None
if "```" in response:
    for part in response.split("```"):
        if part.startswith("python"):
            part = part[6:]
        if "def build_onnx" in part:
            code = part.strip()
            break
if not code and "def build_onnx" in response:
    lines = response.split("\n")
    for i, line in enumerate(lines):
        if "def build_onnx" in line:
            code = "\n".join(lines[i:])
            break

if not code:
    print(json.dumps({"task": TASK_NUM, "status": "no_code", "model": MODEL}))
    sys.exit(0)

# Build model
try:
    import onnx
    from onnx import helper, TensorProto
    ns = {"np": np, "onnx": onnx, "helper": helper, "TensorProto": TensorProto}
    exec(code.strip(), ns)
    build_fn = ns.get("build_onnx")
    if not build_fn:
        print(json.dumps({"task": TASK_NUM, "status": "no_fn", "model": MODEL}))
        sys.exit(0)
    model = build_fn()
    if model is None:
        print(json.dumps({"task": TASK_NUM, "status": "none", "model": MODEL}))
        sys.exit(0)
except Exception as e:
    print(json.dumps({"task": TASK_NUM, "status": "build_error", "model": MODEL,
                       "error": str(e)[:100]}))
    sys.exit(0)

# Verify
try:
    import onnxruntime
    buf = io.BytesIO()
    onnx.save(model, buf)
    sess = onnxruntime.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    correct = total = 0
    for split in ("train", "test"):
        for ex in task.get(split, []):
            total += 1
            t = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for ri, row in enumerate(ex["input"][:30]):
                for ci, v in enumerate(row[:30]):
                    if 0 <= v < 10:
                        t[0, v, ri, ci] = 1.0
            e = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for ri, row in enumerate(ex["output"][:30]):
                for ci, v in enumerate(row[:30]):
                    if 0 <= v < 10:
                        e[0, v, ri, ci] = 1.0
            out = sess.run(["output"], {"input": t})[0]
            if np.array_equal((out > 0).astype(np.float32), e):
                correct += 1
            else:
                break
    if correct == total and total > 0:
        print(json.dumps({"task": TASK_NUM, "status": "solved", "model": MODEL,
                           "correct": correct, "total": total}))
    else:
        print(json.dumps({"task": TASK_NUM, "status": "partial", "model": MODEL,
                           "correct": correct, "total": total}))
except Exception as e:
    print(json.dumps({"task": TASK_NUM, "status": "verify_error", "model": MODEL,
                       "error": str(e)[:100]}))
