#!/usr/bin/env python3
"""Upload Qwen3.5-Wolof-Family tarball and optionally start the family pipeline."""

from __future__ import annotations

import argparse
import base64
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_exec import REMOTE_ROOT, connect, get_hf_token  # noqa: E402

SKIP_TOP = {".venv", "data", "runs", "models", "results", "tokenizers"}
SKIP_ANY = {"__pycache__"}


def make_tarball(dest: Path) -> Path:
    src = ROOT
    with tarfile.open(dest, "w:gz") as handle:
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            if rel.parts and rel.parts[0] in SKIP_TOP:
                continue
            if any(part in SKIP_ANY for part in rel.parts):
                continue
            handle.add(path, arcname=str(Path("Qwen3.5-Wolof-Family") / rel))
    return dest


def run_remote(client, command: str, *, with_token: bool = False) -> int:
    if with_token:
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
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-smoke", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "qwen35-wolof-family.tgz"
        make_tarball(tar_path)
        print(f"tarball {tar_path.stat().st_size} bytes", flush=True)
        client = connect()
        try:
            parent = "/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments"
            remote_tar = "/tmp/qwen35-wolof-family.tgz"
            sftp = client.open_sftp()
            sftp.put(str(tar_path), remote_tar)
            sftp.close()
            code = run_remote(
                client,
                f"mkdir -p {parent} && tar -xzf {remote_tar} -C {parent} && "
                f"test -f {REMOTE_ROOT}/pyproject.toml && echo SYNC_OK",
            )
            if code != 0:
                return code
            if args.start_smoke:
                log = f"{REMOTE_ROOT}/results/full_pipeline/smoke_launch.log"
                cmd = (
                    f"mkdir -p {REMOTE_ROOT}/results/full_pipeline && "
                    f"cd {REMOTE_ROOT} && "
                    "uv sync --group train --group data && "
                    "nohup uv run python scripts/publication/50_run_family.py "
                    "--smoke --only-size 08b --skip-dpo "
                    f"> {log} 2>&1 & echo $! > {REMOTE_ROOT}/results/full_pipeline/smoke.pid "
                    f"&& echo STARTED && tail -n 5 {log} || true"
                )
                if args.detach:
                    # start and return quickly — check log later
                    code = run_remote(client, cmd, with_token=True)
                else:
                    code = run_remote(client, cmd, with_token=True)
                return code
            return 0
        finally:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
