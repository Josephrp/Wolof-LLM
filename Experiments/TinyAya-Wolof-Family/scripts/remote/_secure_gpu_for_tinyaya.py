#!/usr/bin/env python3
"""Kill Qwen, write HOLD, verify TinyAya CPT, extract epochs from CPT launch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect

CMD = r"""
set -euo pipefail
HOLD=/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family/results/full_pipeline/HOLD_FOR_TINYAYA
mkdir -p "$(dirname "$HOLD")"
echo "paused for TinyAya full-3ep redo $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$HOLD"
# nuke qwen trees
pkill -9 -f '/Experiments/Qwen3.5-Wolof-Family/' || true
sleep 2
pkill -9 -f '/Experiments/Qwen3.5-Wolof-Family/' || true
sleep 1
echo '===PROCS==='
pgrep -af 'Qwen3.5|TinyAya' | grep -v grep | head -30 || true
echo '===GPU==='
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo '===TINYAYA_LOG_TAIL==='
LOG=$(ls -t /mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/TinyAya-Wolof-Family/results/full_pipeline/family_full_3ep_*.log | head -1)
tail -n 80 "$LOG" | tr '\r' '\n' | tail -n 80
echo '===STATE==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/TinyAya-Wolof-Family/results/full_pipeline/tinyaya_state.json")
print(sorted((json.load(p.open()).get("completed") or {}).keys()))
PY
echo HOLD=$HOLD
cat "$HOLD"
"""


def main() -> int:
    client = connect()
    try:
        _i, stdout, stderr = client.exec_command(CMD, timeout=120)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print("STDERR", err[-2000:])
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
