#!/usr/bin/env python3
"""Enforce the v1 synthetic-token budget (at most 30%)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.synthetic import read_jsonl, token_budget, whitespace_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description="Check synthetic share by whitespace-token estimate")
    parser.add_argument("--real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--fields", nargs="+", default=["text", "prompt", "chosen"])
    parser.add_argument("--maximum", type=float, default=0.30)
    args = parser.parse_args()
    fields = tuple(args.fields)
    result = token_budget(
        whitespace_tokens(read_jsonl(args.real), fields),
        whitespace_tokens(read_jsonl(args.synthetic), fields),
        args.maximum,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
