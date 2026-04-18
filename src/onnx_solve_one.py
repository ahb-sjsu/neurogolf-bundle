"""Solve a single ARC task by generating ONNX directly via LLM.

Usage: TASK=42 MODEL=kimi NRP_LLM_TOKEN=xxx python onnx_solve_one.py
Prints JSON result to stdout. Exit 0 always (results in logs).
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

prompt = (
    "Write a Python function build_onnx() that returns an onnx.ModelProto.\n\n"
    f"{examples}\n"
    "Constraints:\n"
    "- Input: 'input' shape [1,10,30,30] float32 one-hot\n"
    "- Output: 'output' shape [1,10,30,30] float32\n"
    "- Opset 10, ir_version 10\n"
    "- Allowed: Conv, Gather, Reshape, Concat, Pad, Slice, Mul, Add, Relu,\n"
    "  ReduceMax, ArgMax, Squeeze, Unsqueeze, Transpose, Identity, Clip, Cast, Sub\n"
    "- BANNED: Loop, Scan, If, NonZero, Unique, Where, Resize\n\n"
    "Techniques:\n"
    "- Color remap: 1x1 Conv with 10x10 weight matrix\n"
    "- Spatial remap: Reshape to [1,C,900], Gather with index array, Reshape back\n"
    "- Dynamic grid: ReduceMax + ArgMax on non-bg channels\n"
    "- Masking: Clip(Cast(diff)+1, 0, 1)\n\n"
    "Write ONLY the function. ```python ... ```"
)

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
        model=MODEL, max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    response = r.choices[0].message.content or ""
except Exception as e:
    print(json.dumps({"task": TASK_NUM, "status": "llm_error", "error": str(e)[:100]}))
    sys.exit(0)

# Extract code
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
