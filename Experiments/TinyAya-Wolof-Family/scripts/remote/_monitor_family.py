#!/usr/bin/env python3
"""Poll TinyAya full-family retrain until done or failure."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect

DONE_RE = re.compile(r"TINYAYA_FAMILY_OK|STAGE_OK publish")


def snapshot(client) -> str:
    cmd = f"""
PID=$(cat {REMOTE_ROOT}/results/full_pipeline/family_full_3ep.pid 2>/dev/null || echo)
LOG=$(ls -t {REMOTE_ROOT}/results/full_pipeline/family_full_3ep_*.log 2>/dev/null | head -1)
alive=0
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then alive=1; fi
echo PID=$PID alive=$alive
echo LOG=$LOG
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
if [ -n "$LOG" ]; then
  echo '---TAIL---'
  tail -n 35 "$LOG" | tr '\\r' '\\n' | tail -n 35
  echo '---STATE---'
  python3 -c "import json,os; p='{REMOTE_ROOT}/results/full_pipeline/tinyaya_state.json'; d=json.load(open(p)) if os.path.isfile(p) else {{}}; print('completed', sorted((d.get('completed') or {{}}).keys()))"
fi
"""
    _i, stdout, _stderr = client.exec_command(cmd, timeout=90)
    out = stdout.read().decode("utf-8", errors="replace")
    stdout.channel.recv_exit_status()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=168)
    args = parser.parse_args()
    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            client = connect()
            try:
                text = snapshot(client)
            finally:
                client.close()
        except Exception as exc:
            print(f"==== {stamp} ====\nPROBE_RETRY {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(60, args.interval))
            continue
        print(f"\n==== {stamp} ====", flush=True)
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.flush()
        if DONE_RE.search(text):
            print("MONITOR_DONE", flush=True)
            return 0
        first = text.splitlines()[0] if text else ""
        if "alive=0" in first:
            print("MONITOR_DEAD", flush=True)
            return 1
        time.sleep(args.interval)
    print("MONITOR_TIMEOUT", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
