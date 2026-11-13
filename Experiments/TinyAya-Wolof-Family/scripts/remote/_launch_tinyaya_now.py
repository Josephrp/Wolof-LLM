#!/usr/bin/env python3
"""Launch TinyAya full-3ep family now that the GPU is free."""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect, get_hf_token

SYNC = [
    "common/data_gate.py",
    "scripts/data/01_materialize_sources.py",
    "scripts/clm/03_train_cpt.py",
    "scripts/sft/11_train_sft.py",
    "scripts/sft/14_train_dpo.py",
    "scripts/publication/50_run_family.py",
    "configs/sft/3b35/dpo_v1.yaml",
    "configs/sft/3b35/dpo_std_v1.yaml",
    "configs/clm/3b35/v1_ntp70_ortho_lora.yaml",
    "configs/clm/3b35/v1_ntp70_ortho_std_lora.yaml",
    "configs/sft/3b35/from_cpt_v1.yaml",
    "configs/sft/3b35/from_cpt_std_v1.yaml",
    "configs/sft/3b35/from_earth_v1.yaml",
    "configs/sft/3b35/from_earth_std_v1.yaml",
    "configs/sft/3b35/from_global_v1.yaml",
    "configs/sft/3b35/from_global_std_v1.yaml",
]
LOCAL = Path(__file__).resolve().parents[2]


def main() -> int:
    client = connect()
    try:
        sftp = client.open_sftp()
        for rel in SYNC:
            local = LOCAL / rel
            remote = f"{REMOTE_ROOT}/{rel}"
            parent = str(Path(remote).parent).replace("\\", "/")
            client.exec_command(f"mkdir -p {parent}")[1].channel.recv_exit_status()
            sftp.put(str(local), remote)
            print(f"put {rel}", flush=True)
        sftp.close()

        launch = f"""
set -euo pipefail
cd {REMOTE_ROOT}
export PATH="$HOME/.local/bin:$PATH"
export TINYAYA_PRODUCTION=1
export TINYAYA_PIPELINE_RUN_ID=prod_full3ep
mkdir -p results/full_pipeline
rm -f results/full_pipeline/WAIT_FOR_QWEN
# Refuse if GPU already in use (avoids pgrep self-match on this shell text).
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
mem_i=${{mem%.*}}
if [ "${{mem_i:-0}}" -gt 8000 ]; then
  echo "REFUSE_GPU_BUSY mem=${{mem_i}}MiB"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  exit 2
fi
echo "GPU_CLEAR mem=${{mem_i}}MiB"
# Gate: full data required
uv run python -c "from common.data_gate import assert_production_data; assert_production_data()"
LOG=results/full_pipeline/family_full_3ep_$(date -u +%Y%m%dT%H%M%SZ).log
# Resume shared stages already done (datasets/tokenizer/...); train CPT onward
nohup uv run python scripts/publication/50_run_family.py --ortho both --resume \
  >> "$LOG" 2>&1 &
echo $! > results/full_pipeline/family_full_3ep.pid
echo LAUNCHED pid=$(cat results/full_pipeline/family_full_3ep.pid) log=$LOG
sleep 5
head -n 50 "$LOG" || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
pgrep -af 'TinyAya-Wolof-Family.*(50_run_family|03_train)' | grep -v grep | head -10 || true
"""
        payload = base64.b64encode(launch.encode()).decode()
        wrapper = (
            "python3 -c \"import base64,os,subprocess,sys;"
            "token=sys.stdin.readline().rstrip('\\n');"
            "env=os.environ.copy();env['HF_TOKEN']=token;"
            f"raise SystemExit(subprocess.run(base64.b64decode('{payload}'),"
            "shell=True,env=env,executable='/bin/bash').returncode)\""
        )
        stdin, stdout, stderr = client.exec_command(wrapper, timeout=300)
        stdin.write(get_hf_token() + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        if err.strip():
            sys.stdout.buffer.write(b"STDERR ")
            sys.stdout.buffer.write(err[-2000:].encode("utf-8", errors="replace"))
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
