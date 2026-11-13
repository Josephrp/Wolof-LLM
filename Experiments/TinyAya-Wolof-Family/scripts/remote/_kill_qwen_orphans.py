#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import connect

CMD = r"""
set -euo pipefail
pids=$(pgrep -f 'Qwen3.5-Wolof-Family' || true)
if [ -n "$pids" ]; then
  echo KILL_QWEN $pids
  kill $pids || true
  sleep 3
  pids2=$(pgrep -f 'Qwen3.5-Wolof-Family' || true)
  if [ -n "$pids2" ]; then
    echo FORCE_KILL $pids2
    kill -9 $pids2 || true
  fi
fi
echo '===REMAINING==='
pgrep -af 'Qwen3.5|TinyAya' | grep -v grep | head -20 || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
"""


def main() -> int:
    client = connect()
    try:
        _i, stdout, stderr = client.exec_command(CMD, timeout=60)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print("STDERR", err)
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
