#!/usr/bin/env python3
"""Run configured CLM recipes by model size, failing on the first error."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate size-to-config CLM runs with uv run"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "family_runs.yaml",
    )
    parser.add_argument(
        "--size",
        action="append",
        default=[],
        help="Size key to run; repeatable. Default: all enabled sizes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pass --smoke to each training command",
    )
    return parser


def _load_manifest(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sizes = data.get("sizes")
    if not isinstance(sizes, dict):
        raise TypeError("manifest must contain a 'sizes' mapping")
    result: dict[str, list[str]] = {}
    for size, entry in sizes.items():
        if isinstance(entry, dict):
            if not entry.get("enabled", True):
                continue
            configs = entry.get("configs", [])
        else:
            configs = entry
        if not isinstance(configs, list) or not all(
            isinstance(config, str) for config in configs
        ):
            raise ValueError(f"sizes.{size}.configs must be a list of paths")
        result[str(size)] = configs
    return result


def run(args: argparse.Namespace) -> int:
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    sizes = _load_manifest(manifest)
    selected = args.size or list(sizes)
    unknown = sorted(set(selected) - set(sizes))
    if unknown:
        raise ValueError(
            f"unknown or disabled size(s): {', '.join(unknown)}; "
            f"available: {', '.join(sizes)}"
        )

    commands: list[list[str]] = []
    for size in selected:
        for config in sizes[size]:
            config_path = ROOT / config
            if not config_path.is_file():
                raise ValueError(f"configured recipe does not exist: {config}")
            command = [
                "uv",
                "run",
                "python",
                "scripts/clm/03_train.py",
                "--config",
                config,
            ]
            if args.smoke:
                command.append("--smoke")
            if args.resume:
                command.append("--resume")
            commands.append(command)

    if not commands:
        raise ValueError("selection contains no configured runs")

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
