#!/usr/bin/env python3
"""Stop Qwen ladder hard and disable any respawn wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import connect

CMD = r"""
set -euo pipefail
echo '===BEFORE==='
pgrep -af 'Qwen3.5|TinyAya|crontab|watchdog|ladder' | grep -v grep | head -40 || true
# Kill entire process trees matching Qwen experiment
for pat in 'Qwen3.5-Wolof-Family' '51_run_ladder' 'configs/sft/08b'; do
  pids=$(pgrep -f "$pat" || true)
  if [ -n "$pids" ]; then
    echo KILL_PAT $pat :: $pids
    kill -9 $pids || true
  fi
done
sleep 2
# Block accidental relaunch by renaming launch log marker / touching a hold file
HOLD=/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family/results/full_pipeline/HOLD_FOR_TINYAYA
mkdir -p "$(dirname "$HOLD")"
echo "paused for TinyAya full-3ep redo $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$HOLD"
# Disable common auto-restart shells if present
pkill -9 -f 'ladder_launch' || true
pkill -9 -f 'Qwen3.5-Wolof-Family.*50_run_family' || true
pkill -9 -f 'Qwen3.5-Wolof-Family.*11_train_sft' || true
sleep 1
echo '===AFTER==='
pgrep -af 'Qwen3.5|TinyAya' | grep -v grep | head -40 || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
# Confirm TinyAya CPT still alive
pgrep -af 'TinyAya.*03_train_cpt|TinyAya.*50_run_family' | grep -v grep || echo 'WARNING tinyaya missing'
"""


def main() -> int:
    client = connect()
    try:
        _i, stdout, stderr = client.exec_command(CMD, timeout=90)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print("STDERR", err)
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
