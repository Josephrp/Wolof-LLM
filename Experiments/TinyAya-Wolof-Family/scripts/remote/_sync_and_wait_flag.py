#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect

LOCAL = Path(__file__).resolve().parents[2]


def main() -> int:
    client = connect()
    try:
        sftp = client.open_sftp()
        sftp.put(
            str(LOCAL / "scripts/publication/50_run_family.py"),
            f"{REMOTE_ROOT}/scripts/publication/50_run_family.py",
        )
        print("synced 50_run_family.py")
        sftp.close()
        cmd = f"""
mkdir -p {REMOTE_ROOT}/results/full_pipeline
echo 'defer TinyAya; Qwen 9b owns GPU; CPT ckpt-250 saved' > {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN
cat {REMOTE_ROOT}/results/full_pipeline/WAIT_FOR_QWEN
pgrep -af 'Qwen3.5-Wolof-Family/.venv' | head -8 || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
"""
        _i, stdout, stderr = client.exec_command(cmd, timeout=60)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            print("STDERR", err[-1000:])
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
