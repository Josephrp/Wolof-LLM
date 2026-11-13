#!/usr/bin/env python3
"""Probe remote GPU/state, sync fixed files, clear TinyAya state, launch full family."""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect, get_hf_token  # noqa: E402

LOCAL_ROOT = ROOT
SYNC_PATHS = [
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


def run(client, command: str, *, with_hf: bool = False, timeout: int | None = 120) -> tuple[int, str, str]:
    if with_hf:
        payload = base64.b64encode(command.encode()).decode()
        wrapper = (
            "python3 -c \"import base64,os,subprocess,sys;"
            "token=sys.stdin.readline().rstrip('\\n');"
            "env=os.environ.copy();env['HF_TOKEN']=token;"
            f"raise SystemExit(subprocess.run(base64.b64decode('{payload}'),"
            "shell=True,env=env,executable='/bin/bash').returncode)\""
        )
        stdin, stdout, stderr = client.exec_command(wrapper, timeout=timeout)
        stdin.write(get_hf_token() + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()
    else:
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main() -> int:
    client = connect()
    try:
        code, out, err = run(
            client,
            "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader; "
            "echo '---PROCS---'; "
            "ps -eo pid,etime,cmd | grep -E '51_run_ladder|50_run_family|03_train_cpt|11_train_sft|14_train_dpo' "
            "| grep -v grep | head -30; "
            "echo '---META---'; "
            f"python3 - <<'PY'\n"
            "import json, os\n"
            f"root='{REMOTE_ROOT}'\n"
            "meta=os.path.join(root,'data/raw/materialize_meta.json')\n"
            "state=os.path.join(root,'results/full_pipeline/tinyaya_state.json')\n"
            "print('meta', open(meta).read() if os.path.isfile(meta) else 'MISSING')\n"
            "if os.path.isfile(state):\n"
            "  d=json.load(open(state))\n"
            "  print('completed', sorted((d.get('completed') or {}).keys()))\n"
            "else:\n"
            "  print('state MISSING')\n"
            "PY",
        )
        print(out)
        if err:
            print("STDERR", err)
        print("probe_exit", code)

        sftp = client.open_sftp()
        for rel in SYNC_PATHS:
            local = LOCAL_ROOT / rel
            remote = f"{REMOTE_ROOT}/{rel}"
            # ensure parent
            parent = str(Path(remote).parent).replace("\\", "/")
            run(client, f"mkdir -p {parent}")
            sftp.put(str(local), remote)
            print(f"put {rel}")
        sftp.close()

        # TinyAya full retrain needs the H200; pause Qwen ladder (resumable later).
        pause_qwen = """
set -euo pipefail
mapfile -t pids < <(pgrep -f 'Qwen3.5-Wolof-Family.*51_run_ladder' || true)
if [ "${#pids[@]}" -gt 0 ]; then
  echo "PAUSE_QWEN ${pids[*]}"
  kill "${pids[@]}" || true
  sleep 2
  pgrep -af '51_run_ladder' || echo 'qwen_ladder_stopped'
else
  echo 'qwen_ladder_not_running'
fi
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
"""
        code, out, err = run(client, pause_qwen)
        print(out)
        if err:
            print("STDERR", err)

        # Clear TinyAya completed state so full redo cannot skip stages.
        # Keep a backup; wipe smoke-derived raw caches so rematerialize is unambiguous.
        prep = f"""
set -euo pipefail
cd {REMOTE_ROOT}
mkdir -p results/full_pipeline
ts=$(date -u +%Y%m%dT%H%M%SZ)
if [ -f results/full_pipeline/tinyaya_state.json ]; then
  cp -a results/full_pipeline/tinyaya_state.json "results/full_pipeline/tinyaya_state.json.bak.$ts"
fi
python3 - <<PY
import json
from pathlib import Path
p = Path("results/full_pipeline/tinyaya_state.json")
p.parent.mkdir(parents=True, exist_ok=True)
payload = {{
  "completed": {{}},
  "updated_at": None,
  "best_instruct": {{}},
  "uploads": {{}},
  "reset_reason": "full-split-3epoch-redo",
  "reset_at": "$ts",
}}
json.dump(payload, p.open("w"), indent=2)
print("STATE_RESET", p)
PY
# Remove smoke-capped raw caches; datasets stage rematerializes full splits.
rm -rf data/raw/ntp_70_20_10 data/raw/sft_stratified_v1 data/raw/materialize_meta.json
rm -f data/ortho_ns_mixed_summary.json data/ortho_std_mixed_summary.json
echo PREP_OK
"""
        code, out, err = run(client, prep)
        print(out)
        if err:
            print("STDERR", err)
        if code != 0:
            return code

        # Launch production family without --resume (forced full redo).
        log = f"{REMOTE_ROOT}/results/full_pipeline/family_full_3ep_{int(time.time())}.log"
        launch = f"""
set -euo pipefail
cd {REMOTE_ROOT}
export PATH="$HOME/.local/bin:$PATH"
export TINYAYA_PRODUCTION=1
export TINYAYA_PIPELINE_RUN_ID=prod_full3ep
nohup uv run python scripts/publication/50_run_family.py --ortho both \
  >> {log} 2>&1 &
echo $! > results/full_pipeline/family_full_3ep.pid
echo LAUNCHED pid=$(cat results/full_pipeline/family_full_3ep.pid) log={log}
sleep 2
head -n 40 {log} || true
"""
        code, out, err = run(client, launch, with_hf=True, timeout=180)
        print(out)
        if err:
            print("STDERR", err)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
