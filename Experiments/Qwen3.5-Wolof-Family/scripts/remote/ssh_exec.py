#!/usr/bin/env python3
"""SSH/SFTP helper for the Qwen3.5 Wolof family train host."""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

import paramiko
from rapidocr_onnxruntime import RapidOCR

HOST = "41.173.184.228"
IMAGE = Path(
    r"C:\Users\claud\.cursor\projects\c-Users-claud-galsenai-Wolof-LLM-main"
    r"\assets\C__Users_claud_.cursor_projects_c-Users-claud-galsenai-Wolof-LLM-main_assets_"
    r"c__Users_claud_AppData_Roaming_Cursor_User_workspaceStorage_b7cac3df649144d020712a1a5ea0cc8e_"
    r"images_image-d15a7895-f415-4e79-b47d-f1ecd8391336.png"
)
TRANSCRIPTS = Path(
    r"C:\Users\claud\.cursor\projects\c-Users-claud-galsenai-Wolof-LLM-main"
    r"\agent-transcripts"
)
REMOTE_ROOT = "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family"


def get_password() -> str:
    result, _ = RapidOCR()(str(IMAGE))
    rows = []
    for box, text, _score in result or []:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        rows.append((sum(xs) / len(xs), sum(ys) / len(ys), text.strip()))
    label = next(row for row in rows if row[2].lower() == "password")
    return min(
        (row for row in rows if row[0] > label[0] and abs(row[1] - label[1]) < 30),
        key=lambda row: row[0],
    )[2]


def get_hf_token() -> str:
    candidates = []
    for path in TRANSCRIPTS.rglob("*.jsonl"):
        text = path.read_text(encoding="utf-8", errors="replace")
        candidates.extend(
            (path.stat().st_mtime_ns, match.start(), match.group())
            for match in re.finditer(r"hf_[A-Za-z0-9]{20,}", text)
        )
    if not candidates:
        raise RuntimeError("Hugging Face token not found")
    return max(candidates)[2]


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username="ubuntu",
        password=get_password(),
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
    )
    return client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-b64")
    parser.add_argument("--remote")
    parser.add_argument("--local", type=Path)
    parser.add_argument("--put", action="store_true")
    parser.add_argument("--with-hf-token", action="store_true")
    parser.add_argument("--mkdir-remote-root", action="store_true")
    args = parser.parse_args()
    client = connect()
    try:
        if args.mkdir_remote_root:
            _stdin, stdout, stderr = client.exec_command(f"mkdir -p {REMOTE_ROOT}")
            stdout.channel.recv_exit_status()
            print(REMOTE_ROOT)
            return 0
        if args.put and args.local and args.remote:
            sftp = client.open_sftp()
            remote = args.remote
            # ensure parent
            client.exec_command(f"mkdir -p $(dirname {remote})")
            sftp.put(str(args.local), remote)
            sftp.close()
            print(f"put {args.local} -> {remote}")
            return 0
        if args.command_b64:
            command = base64.b64decode(args.command_b64).decode()
            if args.with_hf_token:
                payload = base64.b64encode(command.encode()).decode()
                wrapper = (
                    "python3 -c \"import base64,os,subprocess,sys;"
                    "token=sys.stdin.readline().rstrip('\\n');"
                    "env=os.environ.copy();env['HF_TOKEN']=token;"
                    f"raise SystemExit(subprocess.run(base64.b64decode('{payload}'),"
                    "shell=True,env=env,executable='/bin/bash').returncode)\""
                )
                stdin, stdout, stderr = client.exec_command(wrapper, timeout=None)
                stdin.write(get_hf_token() + "\n")
                stdin.flush()
                stdin.channel.shutdown_write()
            else:
                _stdin, stdout, stderr = client.exec_command(command, timeout=None)
            out, err = stdout.read(), stderr.read()
            code = stdout.channel.recv_exit_status()
            sys.stdout.buffer.write(out)
            sys.stderr.buffer.write(err)
            return code
        raise SystemExit("provide --command-b64 or --put/--mkdir-remote-root")
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
