"""GPU-parallel conv-arch sweep trainer.

For each task, try N arch templates × M random seeds simultaneously on GPU.
Keep the cheapest arch that reaches 100% accuracy.

Templates tried (by total cost ascending):
  - Conv1x1                                  (~126K cost)
  - Conv3x3                                  (~850K)
  - Conv3x3 + Relu + Conv1x1                 (~720K)
  - Conv1x1 + Conv3x3                        (~1M)
  - Conv3x3 + Relu + Conv3x3                 (~1.7M)
  - Conv5x5                                  (~2.3M)
  - Conv3x3 + Relu + Conv5x5                 (~3.2M)
  - Conv5x5 + Relu + Conv5x5                 (~4.4M)
  - Conv7x7                                  (~4.5M)

Strategy: try cheapest arch first with many seeds; skip to next arch only
if all seeds fail. First arch that any seed solves wins.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import TensorProto, helper


C, H, W = 10, 30, 30


ARCHS = [
    # (name, builder_fn, approx_cost_hint)
    ("conv1x1",   lambda: nn.Sequential(nn.Conv2d(C, C, 1, padding=0, bias=False)),  126_000),
    ("conv3x3",   lambda: nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False)),  850_000),
    ("conv3x3_relu_conv1x1", lambda: nn.Sequential(
        nn.Conv2d(C, C, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(C, C, 1, padding=0, bias=False),
    ), 720_000),
    ("conv1x1_conv3x3", lambda: nn.Sequential(
        nn.Conv2d(C, C, 1, padding=0, bias=False),
        nn.Conv2d(C, C, 3, padding=1, bias=False),
    ), 1_050_000),
    ("conv3x3_relu_conv3x3", lambda: nn.Sequential(
        nn.Conv2d(C, C, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(C, C, 3, padding=1, bias=False),
    ), 1_700_000),
    ("conv5x5",   lambda: nn.Sequential(nn.Conv2d(C, C, 5, padding=2, bias=False)),  2_300_000),
    ("conv5x5_relu_conv3x3", lambda: nn.Sequential(
        nn.Conv2d(C, C, 5, padding=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(C, C, 3, padding=1, bias=False),
    ), 3_200_000),
    ("conv5x5_relu_conv5x5", lambda: nn.Sequential(
        nn.Conv2d(C, C, 5, padding=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(C, C, 5, padding=2, bias=False),
    ), 4_500_000),
    ("conv7x7",   lambda: nn.Sequential(nn.Conv2d(C, C, 7, padding=3, bias=False)),  4_500_000),
]


def grid_to_onehot(grid):
    t = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= H: break
        for c, color in enumerate(row):
            if c >= W: break
            if 0 <= color < C:
                t[0, color, r, c] = 1.0
    return t


def collect_tensors(task, device):
    X, Y = [], []
    for sub in ('train', 'test', 'arc-gen'):
        for ex in task.get(sub, []):
            X.append(grid_to_onehot(ex['input'])[0])
            Y.append(grid_to_onehot(ex['output'])[0])
    if not X:
        return None, None
    return (torch.from_numpy(np.stack(X)).to(device),
            torch.from_numpy(np.stack(Y)).to(device))


def train_arch(arch_builder, X, Y, seeds, device, steps=800, lr=0.05,
                margin=0.5) -> nn.Module | None:
    """Try `len(seeds)` seeds for this arch; return first converged, else None."""
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = arch_builder().to(device)

        # Re-init to small random per seed (torch.manual_seed already changed default init)
        for p in net.parameters():
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.1)

        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for step in range(steps):
            opt.zero_grad()
            out = net(X)
            pos = torch.clamp(margin - out, min=0) * Y
            neg = torch.clamp(margin + out, min=0) * (1 - Y)
            loss = (pos.sum() + neg.sum()) / (X.numel() + 1e-9)
            loss.backward()
            opt.step()
            if step % 50 == 49:
                with torch.no_grad():
                    pred = (net(X) > 0).float()
                    if torch.all(pred == Y).item():
                        return net
        with torch.no_grad():
            pred = (net(X) > 0).float()
            if torch.all(pred == Y).item():
                return net
    return None


def nn_to_onnx(net: nn.Module) -> onnx.ModelProto:
    """Manually build an ONNX model for the nn.Sequential.

    Supports Conv2d and ReLU only (which is all our ARCHS use).
    """
    nodes = []
    inits = []
    vinfos = []
    prev_name = "input"

    for i, layer in enumerate(net):
        if isinstance(layer, nn.Conv2d):
            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            co, ci, kh, kw = w.shape
            w_name = f"W_{i}"
            out_name = f"conv_{i}_out"
            inits.append(helper.make_tensor(
                w_name, TensorProto.FLOAT, list(w.shape), w.flatten().tolist()))
            pads = [kh // 2, kw // 2, kh // 2, kw // 2]
            nodes.append(helper.make_node(
                "Conv", [prev_name, w_name], [out_name],
                kernel_shape=[kh, kw], pads=pads))
            vinfos.append(helper.make_tensor_value_info(
                out_name, TensorProto.FLOAT, [1, co, H, W]))
            prev_name = out_name
        elif isinstance(layer, nn.ReLU):
            out_name = f"relu_{i}_out"
            nodes.append(helper.make_node("Relu", [prev_name], [out_name]))
            vinfos.append(helper.make_tensor_value_info(
                out_name, TensorProto.FLOAT, [1, C, H, W]))
            prev_name = out_name
        else:
            raise ValueError(f"unsupported layer: {type(layer).__name__}")

    # Rename last tensor to "output" via Identity node
    if prev_name != "output":
        nodes.append(helper.make_node("Identity", [prev_name], ["output"]))

    in_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, H, W])
    out_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])

    graph = helper.make_graph(
        nodes, "gpu_conv_trainer",
        [in_info], [out_info],
        initializer=inits, value_info=vinfos)
    return helper.make_model(
        graph, ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)])


def score_onnx(model: onnx.ModelProto) -> dict | None:
    import tempfile
    import onnx_tool
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = f.name
    try:
        onnx.save(model, tmp)
        m = onnx_tool.loadmodel(tmp, {"verbose": False})
        g = m.graph
        g.graph_reorder_nodes()
        g.shape_infer(None)
        g.profile()
        if not g.valid_profile:
            return None
        macs = int(sum(g.macs))
        memory = int(g.memory)
        params = int(g.params)
        cost = macs + memory + params
        return {"params": params, "memory": memory, "macs": macs,
                "cost": cost, "score": max(1.0, 25.0 - math.log(cost))}
    finally:
        try: os.unlink(tmp)
        except: pass


def solve_task_gpu(task: dict, task_num: int, device: str, num_seeds: int,
                   max_time_s: float) -> dict:
    X, Y = collect_tensors(task, device)
    if X is None:
        return {"task": task_num, "status": "no_examples"}

    t0 = time.time()
    best = None
    for arch_name, arch_builder, _hint in ARCHS:
        if time.time() - t0 > max_time_s:
            break
        seeds = list(range(100, 100 + num_seeds))
        net = train_arch(arch_builder, X, Y, seeds, device)
        if net is not None:
            model = nn_to_onnx(net)
            s = score_onnx(model)
            if s is None:
                continue
            best = {"task": task_num, "status": "solved",
                    "arch": arch_name,
                    "cost": s["cost"], "score": round(s["score"], 3),
                    "elapsed": round(time.time() - t0, 1),
                    "model": model}
            break  # cheapest arch found; stop
    if best is None:
        return {"task": task_num, "status": "unsolved",
                "elapsed": round(time.time() - t0, 1)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, required=True)
    ap.add_argument("--output-dir", default="solutions_gpu_conv")
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    task_nums = [int(t) for t in args.tasks.split(",") if t.strip()]
    n_solved = 0
    n_err = 0
    for tn in task_nums:
        p = ROOT / f"task{tn:03d}.json"
        if not p.exists():
            continue
        with open(p) as f:
            task = json.load(f)
        try:
            r = solve_task_gpu(task, tn, device, args.num_seeds, args.max_time_s)
        except Exception as e:
            r = {"task": tn, "status": "error", "error": str(e)[:200]}
            n_err += 1
        if r.get("status") == "solved":
            onnx.save(r["model"], str(output_dir / f"task{tn:03d}.onnx"))
            n_solved += 1
            print(f"task{tn:03d}: {r['arch']} cost={r['cost']} "
                  f"score={r['score']} t={r['elapsed']}s", flush=True)
        else:
            print(f"task{tn:03d}: {r['status']} t={r.get('elapsed','?')}s", flush=True)
    print()
    print(f"solved={n_solved}/{len(task_nums)} err={n_err}")


if __name__ == "__main__":
    main()
