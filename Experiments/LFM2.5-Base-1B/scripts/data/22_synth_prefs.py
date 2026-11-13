#!/usr/bin/env python3
"""Validate generated preference pairs for downstream DPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.synthetic import read_jsonl, write_jsonl


def validate_pairs(rows: list[dict]) -> list[dict]:
    accepted = []
    for row in rows:
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("prompt", "chosen", "rejected")):
            continue
        if row["chosen"].strip() == row["rejected"].strip():
            continue
        accepted.append({**row, "synthetic": True, "requires_bpb_gate": True})
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DPO candidate JSONL; does not generate responses")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()
    write_jsonl(args.output, validate_pairs(read_jsonl(args.input))[: args.max_rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
