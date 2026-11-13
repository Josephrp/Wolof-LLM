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
pgrep -af 'Qwen3.5-Wolof-Family/.venv|TinyAya-Wolof-Family/.venv' | grep -v grep | head -20 || echo none
echo '===TINYAYA_STATE==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("{REMOTE_ROOT}/results/full_pipeline/tinyaya_state.json")
if p.is_file():
  d=json.load(p.open())
  print("completed", sorted((d.get("completed") or {{}}).keys()))
else:
  print("MISSING")
PY
echo '===META==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("{REMOTE_ROOT}/data/raw/materialize_meta.json")
print(json.load(p.open()) if p.is_file() else "NO_META")
PY
echo '===CPT_CKPT==='
ls -d {REMOTE_ROOT}/runs/clm/3b35/v1_ntp70_ortho_lora/checkpoint-* 2>/dev/null | tail -5 || echo no_ckpt
echo '===QWEN_STATE==='
python3 - <<'PY'
import json
from pathlib import Path
p=Path("{QWEN}/results/full_pipeline/qwen35_family_state.json")
if p.is_file():
  d=json.load(p.open())
  keys=sorted((d.get("completed") or {{}}).keys())
  print("n", len(keys))
  print([k for k in keys if "9b" in k or k in ("materialize","tokenizer")])
else:
  print("MISSING")
PY
test -f {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN && echo WAIT_PRESENT || echo WAIT_ABSENT
"""


def main() -> int:
    client = connect()
    try:
        _i, stdout, stderr = client.exec_command(CMD, timeout=120)
        sys.stdout.buffer.write(stdout.read())
        err = stderr.read()
        if err.strip():
            sys.stdout.buffer.write(b"\nSTDERR ")
            sys.stdout.buffer.write(err[-1500:])
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
