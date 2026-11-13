#!/usr/bin/env python3
"""Stop TinyAya family, clear Qwen holds, ensure Qwen ladder owns the GPU.

Leaves TinyAya state intact so a later full-3ep run can resume/rematerialize.
Writes WAIT_FOR_QWEN so a deferred launcher knows not to start yet.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect, get_hf_token

QWEN_ROOT = "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family"
LOCAL_LADDER = (
    Path(__file__).resolve().parents[3]
    / "Qwen3.5-Wolof-Family"
    / "scripts"
    / "publication"
    / "51_run_ladder.py"
)


def run(client, command: str, *, with_hf: bool = False, timeout: int | None = 180) -> tuple[int, str, str]:
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
        # Sync ladder without HOLD gate (local file already reverted).
        sftp = client.open_sftp()
        sftp.put(
            str(LOCAL_LADDER),
            f"{QWEN_ROOT}/scripts/publication/51_run_ladder.py",
        )
        print("synced 51_run_ladder.py")
        sftp.close()

        prep = f"""
set -euo pipefail
# Stop TinyAya family (keep state/data for later full-3ep launch)
pkill -9 -f '{REMOTE_ROOT}/' || true
pkill -9 -f 'TinyAya-Wolof-Family' || true
sleep 2
rm -f {QWEN_ROOT}/results/full_pipeline/HOLD_FOR_TINYAYA
mkdir -p {REMOTE_ROOT}/results/full_pipeline
echo "defer TinyAya until Qwen ladder frees GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN
echo '===AFTER_KILL_TINYAYA==='
pgrep -af 'TinyAya|Qwen3.5' | grep -v grep | head -20 || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
"""
        code, out, err = run(client, prep)
        print(out)
        if err.strip():
            print("STDERR", err[-1500:])
        if code != 0:
            return code

        # Start Qwen ladder if not already running.
        launch = f"""
set -euo pipefail
cd {QWEN_ROOT}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p results/full_pipeline
if pgrep -f 'Qwen3.5-Wolof-Family.*51_run_ladder' >/dev/null; then
  echo QWEN_LADDER_ALREADY_RUNNING
  pgrep -af 'Qwen3.5-Wolof-Family.*(51_run_ladder|50_run_family|11_train|03_train)' | grep -v grep | head -15
else
  nohup bash -lc '
    set -euo pipefail
    cd {QWEN_ROOT}
    export PATH="$HOME/.local/bin:$PATH"
    set -a; source .env.trackio 2>/dev/null || true; set +a
    export QWEN_PRODUCTION=1
    uv run python scripts/publication/51_run_ladder.py --sizes 08b 2b 4b 9b --keep-smoke-marks
  ' >> results/full_pipeline/ladder_launch.log 2>&1 &
  echo $! > results/full_pipeline/ladder.pid
  echo QWEN_LADDER_LAUNCHED pid=$(cat results/full_pipeline/ladder.pid)
  sleep 3
fi
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
pgrep -af 'Qwen3.5-Wolof-Family.*(51_run_ladder|50_run_family|11_train|03_train)' | grep -v grep | head -15 || true
tail -n 15 results/full_pipeline/ladder_launch.log | tr '\\r' '\\n' | tail -n 15
"""
        code, out, err = run(client, launch, with_hf=True, timeout=180)
        print(out)
        if err.strip():
            print("STDERR", err[-1500:])
        print(f"defer_note={REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN")
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
