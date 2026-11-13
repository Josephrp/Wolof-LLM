#!/usr/bin/env python3
"""Validate and run GGUF evaluation configs with uv."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

EVAL_ROOT = Path(__file__).resolve().parents[1]


def load_entries(pattern: str) -> list[tuple[Path, dict]]:
    entries = []
    outputs = set()
    for path in sorted((EVAL_ROOT / "configs" / "gguf").glob(pattern)):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        required = {"gguf_path", "corpus_csv", "output_json"}
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        output = config["output_json"]
        if output in outputs:
            raise ValueError(f"duplicate output_json: {output}")
        outputs.add(output)
        entries.append((path, config))
    return entries


def run(args: argparse.Namespace) -> int:
    entries = load_entries(args.pattern)
    runnable = 0
    for path, config in entries:
        if config.get("placeholder", False):
            print(f"SKIP placeholder: {path.name}")
            continue
        if not Path(config["gguf_path"]).is_file():
            print(f"SKIP missing gguf: {path.name} -> {config['gguf_path']}")
            continue
        command = [
            "uv",
            "run",
            "python",
            "scripts/evaluate_gguf.py",
            "--config",
            str(path),
        ]
        print(shlex.join(command))
        runnable += 1
        if not args.dry_run:
            subprocess.run(command, cwd=EVAL_ROOT, check=True)
    if not entries:
        raise ValueError("no configs matched")
    print(f"{len(entries)} configs validated; {runnable} runnable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run GGUF family evaluations")
    parser.add_argument("--pattern", default="lfm_*.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
