#!/usr/bin/env python3
"""Smoke-verify a Tiny Aya GGUF with llama-completion / llama-cli."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify GGUF load + short generation")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument(
        "--llama-root",
        type=Path,
        default=Path("/mnt/volume_w2l8nkp/tools/llama.cpp"),
    )
    parser.add_argument("--prompt", default="Salaam aleekum, naka nga def?")
    parser.add_argument("--n-predict", type=int, default=48)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def resolve_cli(llama_root: Path) -> Path:
    for name in ("llama-completion", "llama-cli"):
        candidate = llama_root / "build" / "bin" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no llama-completion/llama-cli under {llama_root}")


def run_verify(args: argparse.Namespace) -> int:
    gguf = args.gguf if args.gguf.is_absolute() else ROOT / args.gguf
    if not gguf.is_file():
        raise FileNotFoundError(gguf)
    cli = resolve_cli(args.llama_root)
    cmd = [
        str(cli),
        "-m",
        str(gguf),
        "-p",
        args.prompt,
        "-n",
        str(args.n_predict),
        "-ngl",
        str(args.n_gpu_layers),
        "-no-cnv",
        "--single-turn",
    ]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False, capture_output=True)
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    ok = proc.returncode == 0 and len(stdout.strip()) > 0
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "gguf": gguf.name,
        "ok": ok,
        "returncode": proc.returncode,
        "prompt": args.prompt,
        "n_predict": args.n_predict,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-1500:],
    }
    out = args.out
    if out is not None:
        out_path = out if out.is_absolute() else ROOT / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "gguf": gguf.name, "returncode": proc.returncode}, indent=2))
    if not ok:
        print(stderr[-2000:], file=sys.stderr)
        return 1
    return 0


def main() -> int:
    try:
        return run_verify(build_parser().parse_args())
    except (OSError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
