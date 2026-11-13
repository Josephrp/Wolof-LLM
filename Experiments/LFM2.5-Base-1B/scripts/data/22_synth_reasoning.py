#!/usr/bin/env python3
"""Filter an externally generated CoT pilot and create its G0 review manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.synthetic import read_jsonl, write_jsonl

ANSWER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def valid_reasoning(row: dict) -> tuple[bool, str]:
    for field in ("prompt", "reasoning", "answer", "gold_answer"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            return False, f"missing_{field}"
    steps = [part for part in re.split(r"\n+|\d+\.", row["reasoning"]) if part.strip()]
    if len(steps) < 2:
        return False, "too_few_steps"
    predicted = ANSWER_RE.findall(row["answer"].replace(",", "."))
    gold = ANSWER_RE.findall(row["gold_answer"].replace(",", "."))
    if predicted and gold and predicted[-1] != gold[-1]:
        return False, "answer_mismatch"
    words = row["reasoning"].lower().split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.25:
        return False, "degenerate"
    return True, "ok"


def run(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    accepted, rejected = [], {}
    for row in rows:
        ok, reason = valid_reasoning(row)
        if ok:
            accepted.append(row)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    if len(accepted) > args.max_rows:
        accepted = accepted[: args.max_rows]
    write_jsonl(args.output, accepted)
    review = random.Random(args.seed).sample(accepted, min(50, len(accepted)))
    manifest = [
        {"sample_id": index, "prompt": row["prompt"], "reasoning": row["reasoning"],
         "answer": row["answer"], "wolof_ok": None, "logic_ok": None, "notes": ""}
        for index, row in enumerate(review)
    ]
    Path(args.review).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"input": len(rows), "accepted": len(accepted), "rejected": rejected}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter Wolof CoT pilot; no generation API is called")
    parser.add_argument("--input", required=True, help="Generated candidate JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--review", default="results/sft/reasoning_g0_review.json")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
