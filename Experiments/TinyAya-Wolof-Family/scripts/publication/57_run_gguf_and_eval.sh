#!/usr/bin/env bash
# Convert+publish all TinyAya GGUFs (imatrix-calibrated), then eval+publish metrics.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/full_pipeline logs
LOG="logs/gguf_family_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "TINYAYA_GGUF_START $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$LOG"
if [[ -f .env.trackio ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.trackio
  set +a
fi
if [[ -z "${HF_TOKEN:-}" && -f .hf_token ]]; then
  HF_TOKEN="$(tr -d '\r\n' < .hf_token)"
  export HF_TOKEN
fi
export PYTHONUNBUFFERED=1
export TRACKIO_DISABLED="${TRACKIO_DISABLED:-0}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
echo "+ convert family"
"$PY" scripts/publication/54_run_gguf_family.py --resume
echo "+ eval publish"
"$PY" scripts/publication/59_eval_publish_gguf.py --resume
echo TINYAYA_GGUF_PIPELINE_OK
