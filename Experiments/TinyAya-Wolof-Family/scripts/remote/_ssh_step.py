#!/usr/bin/env python3
"""Run one remote bash snippet (argv[1] as literal step name)."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect, get_hf_token

QWEN_ROOT = "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family"

STEPS = {
    "status": f"""
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo '---'
pgrep -af 'Qwen3.5|TinyAya' | grep -v grep | head -25 || true
echo '---'
test -f {QWEN_ROOT}/results/full_pipeline/HOLD_FOR_TINYAYA && echo HOLD_PRESENT || echo HOLD_ABSENT
test -f {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN && echo WAIT_PRESENT || echo WAIT_ABSENT
""",
    "stop_tinyaya": f"""
pkill -9 -f 'TinyAya-Wolof-Family' || true
sleep 2
pgrep -af TinyAya | grep -v grep | head -10 || echo NO_TINYAYA
mkdir -p {REMOTE_ROOT}/results/full_pipeline
echo "defer TinyAya until Qwen frees GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" > {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN
rm -f {QWEN_ROOT}/results/full_pipeline/HOLD_FOR_TINYAYA
echo STOP_TINYAYA_OK
""",
    "ensure_qwen": f"""
cd {QWEN_ROOT}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p results/full_pipeline
rm -f results/full_pipeline/HOLD_FOR_TINYAYA
if pgrep -f 'Qwen3.5-Wolof-Family.*51_run_ladder' >/dev/null; then
  echo QWEN_ALREADY
else
  nohup bash -lc 'cd {QWEN_ROOT}; export PATH="$HOME/.local/bin:$PATH"; set -a; source .env.trackio 2>/dev/null || true; set +a; export QWEN_PRODUCTION=1; uv run python scripts/publication/51_run_ladder.py --sizes 08b 2b 4b 9b --keep-smoke-marks' >> results/full_pipeline/ladder_launch.log 2>&1 &
  echo $! > results/full_pipeline/ladder.pid
  echo QWEN_STARTED $(cat results/full_pipeline/ladder.pid)
  sleep 4
fi
pgrep -af 'Qwen3.5-Wolof-Family.*(51_run_ladder|50_run_family|11_train|03_train)' | grep -v grep | head -15 || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
""",
}


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "status"
    if name not in STEPS:
        raise SystemExit(f"unknown step {name}; choose from {list(STEPS)}")
    with_hf = name == "ensure_qwen"
    client = connect()
    try:
        command = STEPS[name]
        if with_hf:
            payload = base64.b64encode(command.encode()).decode()
            wrapper = (
                "python3 -c \"import base64,os,subprocess,sys;"
                "token=sys.stdin.readline().rstrip('\\n');"
                "env=os.environ.copy();env['HF_TOKEN']=token;"
                f"raise SystemExit(subprocess.run(base64.b64decode('{payload}'),"
                "shell=True,env=env,executable='/bin/bash').returncode)\""
            )
            stdin, stdout, stderr = client.exec_command(wrapper, timeout=180)
            stdin.write(get_hf_token() + "\n")
            stdin.flush()
            stdin.channel.shutdown_write()
        else:
            _i, stdout, stderr = client.exec_command(command, timeout=120)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print("STDERR", err[-1500:])
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
