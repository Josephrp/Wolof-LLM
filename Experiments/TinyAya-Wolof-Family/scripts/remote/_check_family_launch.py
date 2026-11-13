#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect


def main() -> int:
    client = connect()
    try:
        cmd = f"""
set -euo pipefail
echo '===GPU==='
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo '===PROCS==='
ps -eo pid,etime,cmd | grep -E 'Qwen3.5|TinyAya|50_run_family|51_run_ladder|11_train_sft|03_train_cpt|01_materialize' | grep -v grep | head -40 || true
echo '===PIDFILE==='
cat {REMOTE_ROOT}/results/full_pipeline/family_full_3ep.pid 2>/dev/null || echo missing
echo '===LOG==='
ls -lt {REMOTE_ROOT}/results/full_pipeline/family_full_3ep_*.log 2>/dev/null | head -3
LOG=$(ls -t {REMOTE_ROOT}/results/full_pipeline/family_full_3ep_*.log 2>/dev/null | head -1)
if [ -n "${{LOG:-}}" ]; then
  echo LOG=$LOG
  wc -l "$LOG"
  tail -n 60 "$LOG"
fi
"""
        _i, stdout, stderr = client.exec_command(cmd, timeout=60)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print("STDERR", err)
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
