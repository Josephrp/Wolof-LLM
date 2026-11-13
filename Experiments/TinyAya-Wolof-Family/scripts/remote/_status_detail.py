#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect

QWEN = "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family"

CMD = f"""
set -euo pipefail
echo '===GPU==='
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo '===PROCS==='
pgrep -af 'Qwen3.5|TinyAya' | grep -v grep | head -30 || true
echo '===QWEN_STATE==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("{QWEN}/results/full_pipeline/qwen35_family_state.json")
if p.is_file():
  d=json.load(p.open())
  keys=sorted((d.get("completed") or {{}}).keys())
  print("n_completed", len(keys))
  print("completed", keys)
  print("mode", d.get("mode"))
else:
  print("MISSING")
PY
echo '===QWEN_LOG==='
tail -n 40 {QWEN}/results/full_pipeline/ladder_launch.log 2>/dev/null | tr '\\r' '\\n' | tail -n 40
echo '===TINYAYA_STATE==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("{REMOTE_ROOT}/results/full_pipeline/tinyaya_state.json")
if p.is_file():
  d=json.load(p.open())
  print("completed", sorted((d.get("completed") or {{}}).keys()))
  print("reset", d.get("reset_reason"), d.get("reset_at"))
else:
  print("MISSING")
PY
echo '===FLAGS==='
test -f {QWEN}/results/full_pipeline/HOLD_FOR_TINYAYA && echo HOLD_PRESENT || echo HOLD_ABSENT
test -f {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN && echo WAIT_PRESENT || echo WAIT_ABSENT
test -f {REMOTE_ROOT}/data/raw/materialize_meta.json && python3 -c "import json; m=json.load(open('{REMOTE_ROOT}/data/raw/materialize_meta.json')); print('tinyaya_smoke', m.get('smoke'), m.get('row_counts'))" || echo NO_META
"""


def main() -> int:
    client = connect()
    try:
        _i, stdout, stderr = client.exec_command(CMD, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        if err.strip():
            sys.stdout.buffer.write(b"STDERR ")
            sys.stdout.buffer.write(err[-2000:].encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
