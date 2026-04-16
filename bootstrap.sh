#!/usr/bin/env bash
# Bootstrap inside an NRP pod: clone this repo, install deps, extract tasks,
# then run the dag_astar_worker.
#
# Used as the command inside a nats-bursting JobDescriptor:
#   command: ["bash", "-c", "curl -s https://raw.githubusercontent.com/ahb-sjsu/neurogolf-bundle/main/bootstrap.sh | bash"]
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ahb-sjsu/neurogolf-bundle.git}"
REPO_REF="${REPO_REF:-master}"
WORK_DIR="${WORK_DIR:-/tmp/ng}"

echo "[bootstrap] clone $REPO_URL@$REPO_REF -> $WORK_DIR"
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$WORK_DIR"

echo "[bootstrap] pip install deps"
# Detect GPU via nvidia-smi; install CUDA torch if present, CPU wheel otherwise.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] GPU detected -> installing torch-cuda (cu118 for broad compat including sm_61)"
  # CUDA 11.8 wheels support sm_50..sm_90 (1080 Ti, V100, A100, H100)
  pip install --quiet --root-user-action=ignore \
      torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
else
  echo "[bootstrap] no GPU -> installing torch-cpu"
  pip install --quiet --root-user-action=ignore \
      torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install --quiet --root-user-action=ignore \
    onnx onnxruntime onnx-tool numpy

echo "[bootstrap] extract tasks.tar.gz"
mkdir -p /data/tasks
tar xzf "$WORK_DIR/data/tasks.tar.gz" -C /data/tasks
echo "[bootstrap] tasks: $(ls /data/tasks | wc -l) files"

echo "[bootstrap] run worker — NEUROGOLF_JOB=$(echo ${NEUROGOLF_JOB:-<none>} | head -c 200)..."
cd "$WORK_DIR"
# Which worker? NEUROGOLF_WORKER=dag_astar (default) or gpu_conv
WORKER="${NEUROGOLF_WORKER:-dag_astar}"
if [ "$WORKER" = "gpu_conv" ]; then
  echo "[bootstrap] running gpu_conv_trainer"
  exec python3 src/gpu_conv_trainer.py \
      --tasks "${NEUROGOLF_TASKS}" \
      --num-seeds "${NEUROGOLF_NUM_SEEDS:-5}" \
      --max-time-s "${NEUROGOLF_MAX_TIME_S:-180}" \
      --emit-stdout-records \
      --output-dir /tmp/out
else
  exec python3 src/dag_astar_worker.py
fi
