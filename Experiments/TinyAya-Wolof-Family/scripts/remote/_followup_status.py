#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect

LOCAL_QWEN_LADDER = (
    Path(__file__).resolve().parents[3]
    / "Qwen3.5-Wolof-Family"
    / "scripts"
    / "publication"
    / "51_run_ladder.py"
)


def main() -> int:
    client = connect()
    try:
        sftp = client.open_sftp()
        remote_ladder = (
            "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family"
            "/scripts/publication/51_run_ladder.py"
        )
        sftp.put(str(LOCAL_QWEN_LADDER), remote_ladder)
        print("put 51_run_ladder.py (HOLD gate)")
        sftp.close()

        cmd = rf"""
set -euo pipefail
HOLD=/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family/results/full_pipeline/HOLD_FOR_TINYAYA
mkdir -p "$(dirname "$HOLD")"
echo "paused for TinyAya full-3ep redo $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$HOLD"
pkill -9 -f '/Experiments/Qwen3.5-Wolof-Family/' || true
sleep 2
echo '===PROCS==='
pgrep -af 'Qwen3.5|TinyAya' | grep -v grep | head -25 || true
echo '===GPU==='
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo '===STATE==='
python3 -c "import json; d=json.load(open('{REMOTE_ROOT}/results/full_pipeline/tinyaya_state.json')); print(sorted((d.get('completed') or {}).keys()))"
LOG=$(ls -t {REMOTE_ROOT}/results/full_pipeline/family_full_3ep_*.log | head -1)
echo LOG=$LOG
tail -n 50 "$LOG" | tr '\r' '\n' | tail -n 50
# ensure family still running; if CPT died mid-way, leave note
if ! pgrep -f 'TinyAya-Wolof-Family.*50_run_family' >/dev/null; then
  echo TINYAYA_FAMILY_DEAD
else
  echo TINYAYA_FAMILY_ALIVE
fi
"""
        _i, stdout, stderr = client.exec_command(cmd, timeout=120)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print("STDERR", err[-1500:])
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
