#!/usr/bin/env python3
"""Run evaluate_morena.py over a family YAML list (skips placeholder: true)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

EVAL_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EVAL_ROOT / "configs" / "morena" / "family_stock.yaml")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    jobs = manifest.get("jobs") or []
    script = EVAL_ROOT / "scripts" / "evaluate_morena.py"
    failed = []
    for job in jobs:
        if job.get("placeholder"):
            print(f"[skip] {job.get('name')}")
            continue
        cfg = EVAL_ROOT / job["config"]
        print(f"[run] {job.get('name')} -> {cfg}")
        code = subprocess.call([args.python, str(script), "--config", str(cfg)], cwd=str(EVAL_ROOT))
        if code != 0:
            failed.append(job.get("name"))
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    print("MORENA_STOCK_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
