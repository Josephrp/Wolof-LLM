#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect

CMD = f"""
set -euo pipefail
LOG=$(ls -t {REMOTE_ROOT}/results/full_pipeline/family_full_3ep_*.log 2>/dev/null | head -1)
echo LOG=$LOG
echo '===LAST_ERRORish==='
grep -nE 'Error|Traceback|Killed|CUDA|OOM|SIGNAL|STAGE_OK|CalledProcess' "$LOG" 2>/dev/null | tail -n 40 || true
echo '===TAIL==='
tail -c 4000 "$LOG" | tr '\\r' '\\n' | tail -n 60
echo '===CPT_CKPT==='
ls -lt {REMOTE_ROOT}/runs/clm/3b35/v1_ntp70_ortho_lora/checkpoint-* 2>/dev/null | head -5 || echo no_ckpt
ls -lt {REMOTE_ROOT}/runs/clm/3b35/v1_ntp70_ortho_lora/*/checkpoint-* 2>/dev/null | head -5 || true
find {REMOTE_ROOT}/runs/clm/3b35 -name 'trainer_state.json' 2>/dev/null | head -5 | while read f; do
  echo "== $f =="
  python3 -c "import json; s=json.load(open('$f')); print('step', s.get('global_step'), 'epoch', s.get('epoch'))"
done
echo '===QWEN==='
pgrep -af 'Qwen3.5-Wolof-Family/.venv' | head -10 || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
"""


def main() -> int:
    client = connect()
    try:
        _i, stdout, stderr = client.exec_command(CMD, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        if err.strip():
            sys.stdout.buffer.write(b"\nSTDERR ")
            sys.stdout.buffer.write(err[-1500:].encode("utf-8", errors="replace"))
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
