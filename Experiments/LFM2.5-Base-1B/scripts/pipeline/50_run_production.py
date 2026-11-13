#!/usr/bin/env python3
"""Run the gated production pipeline with durable resume state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import PipelineLock, PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable LFM2.5 Wolof production stages")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "production_pipeline.yaml",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "full_pipeline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-size",
        choices=("230m", "350m", "1.2b", "2.6b", "8b-a1b"),
        default=None,
    )
    return parser


def run(args: argparse.Namespace) -> int:
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    runner = PipelineRunner(
        root=ROOT,
        manifest_path=manifest,
        state_path=run_dir / "pipeline_state.json",
        logs_dir=run_dir / "logs",
        resume=args.resume,
        dry_run=args.dry_run,
    )
    with PipelineLock(run_dir / "pipeline.lock"):
        return runner.run(args.only_size)


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
