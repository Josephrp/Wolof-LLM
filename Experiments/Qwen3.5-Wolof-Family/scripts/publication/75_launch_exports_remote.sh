#!/usr/bin/env bash
# Launch ONNX + GGUF family exports on the H200 host (no HF Jobs).
set -euo pipefail
ROOT="${ROOT:-/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family}"
cd "$ROOT"
mkdir -p logs results/full_pipeline

# Ensure export deps (idempotent).
.venv/bin/python -m pip install -q -U "optimum[onnx]" onnxruntime onnxconverter-common onnx 2>&1 | tail -5

echo "[launch] ONNX family (Instruct 08b/2b/4b; skip optional 9b)"
nohup .venv/bin/python -u scripts/publication/74_run_onnx_family.py \
  --config configs/publication/onnx_family.yaml \
  --state results/full_pipeline/onnx_state.json \
  --resume \
  --cleanup-after-publish \
  > logs/onnx_family.log 2>&1 &
echo "ONNX_PID=$!"

echo "[launch] GGUF family (all sizes CPT+Instruct)"
nohup .venv/bin/python -u scripts/publication/72_run_gguf_family.py \
  --config configs/publication/gguf_family.yaml \
  --state results/full_pipeline/gguf_state.json \
  --resume \
  --cleanup-after-publish \
  > logs/gguf_family.log 2>&1 &
echo "GGUF_PID=$!"

sleep 2
tail -n 20 logs/onnx_family.log || true
tail -n 20 logs/gguf_family.log || true
