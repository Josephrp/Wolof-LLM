#!/usr/bin/env python3
"""Poll until Qwen ladder is idle, then launch TinyAya full-3ep family.

Opportunity = no Qwen train/ladder processes AND GPU util/memory low,
or ladder log shows FAMILY_LADDER_OK.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect, get_hf_token

QWEN_ROOT = "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family"


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


# NOTE: only {QWEN_ROOT} is interpolated; remote Python braces must stay doubled.
PROBE = f"""
python3 - <<'PY'
import os, subprocess

def lines(cmd):
    out = subprocess.getoutput(cmd)
    return [ln for ln in out.splitlines() if ln.strip() and "pgrep" not in ln]

# pgrep uses BRE — keep patterns simple (no | groups).
qwen_lines = []
for pat in (
    "Qwen3.5-Wolof-Family.*51_run_ladder",
    "Qwen3.5-Wolof-Family.*50_run_family",
    "Qwen3.5-Wolof-Family.*03_train",
    "Qwen3.5-Wolof-Family.*11_train",
    "Qwen3.5-Wolof-Family.*14_train",
):
    qwen_lines.extend(lines("pgrep -af '%s' || true" % pat))
tiny_lines = []
for pat in (
    "TinyAya-Wolof-Family.*50_run_family",
    "TinyAya-Wolof-Family.*03_train",
    "TinyAya-Wolof-Family.*11_train",
):
    tiny_lines.extend(lines("pgrep -af '%s' || true" % pat))
qwen_lines = list(dict.fromkeys(qwen_lines))
tiny_lines = list(dict.fromkeys(tiny_lines))
gpu = subprocess.getoutput(
    "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits"
)
log = "{QWEN_ROOT}/results/full_pipeline/ladder_launch.log"
tail = ""
if os.path.isfile(log):
    with open(log, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 8000))
        tail = f.read().decode("utf-8", errors="replace")
done = "FAMILY_LADDER_OK" in tail
qwen_alive = bool(qwen_lines)
tiny_alive = bool(tiny_lines)
try:
    mem_s, util_s = [x.strip() for x in gpu.split(",")]
    mem, util = float(mem_s), float(util_s)
except Exception:
    mem, util = 99999.0, 100.0
opportunity = done or ((not qwen_alive) and mem < 4000 and util < 5)
print(
    "qwen_alive=%d tiny_alive=%d mem=%s util=%s ladder_done=%d opportunity=%d"
    % (int(qwen_alive), int(tiny_alive), mem, util, int(done), int(opportunity))
)
if qwen_alive:
    print("QWEN:", qwen_lines[0][:220])
if tiny_alive:
    print("TINYAYA:", tiny_lines[0][:220])
PY
"""


LAUNCH = f"""
set -euo pipefail
cd {REMOTE_ROOT}
export PATH="$HOME/.local/bin:$PATH"
export TINYAYA_PRODUCTION=1
export TINYAYA_PIPELINE_RUN_ID=prod_full3ep
rm -f results/full_pipeline/WAIT_FOR_QWEN
# Do not use --resume: force full redo from current (already full) data + empty/partial state
# If datasets already complete with full data, resume is OK after gate passes.
nohup uv run python scripts/publication/50_run_family.py --ortho both --resume \
  >> results/full_pipeline/family_full_3ep_deferred.log 2>&1 &
echo $! > results/full_pipeline/family_full_3ep.pid
echo TINYAYA_DEFERRED_LAUNCH pid=$(cat results/full_pipeline/family_full_3ep.pid)
sleep 2
head -n 30 results/full_pipeline/family_full_3ep_deferred.log || true
"""


def main() -> int:
    interval = 300
    max_hours = 168  # allow a full week of Qwen ladder
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            client = connect()
            try:
                code, out, err = run(client, PROBE)
                print(f"==== {stamp} ====", flush=True)
                print(out, flush=True)
                if err.strip():
                    print("STDERR", err[-500:], flush=True)
                if "opportunity=1" in out:
                    if "tiny_alive=1" in out:
                        print("TINYAYA_ALREADY_RUNNING", flush=True)
                        return 0
                    print("LAUNCHING_TINYAYA", flush=True)
                    code, out2, err2 = run(client, LAUNCH, with_hf=True, timeout=180)
                    print(out2, flush=True)
                    if err2.strip():
                        print("STDERR", err2[-1000:], flush=True)
                    return code
            finally:
                client.close()
        except Exception as exc:
            # Transient SSH blips must not kill the deferral.
            print(f"==== {stamp} ====\nPROBE_RETRY {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(60, interval))
            continue
        time.sleep(interval)
    print("WAIT_TIMEOUT", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
