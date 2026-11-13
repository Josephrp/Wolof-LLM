#!/usr/bin/env python3
"""Build simple DPO preference pairs from anchored SFT (LFM-style columns)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.anchoring import normalize_messages
from common.pipeline import atomic_json_write, utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Wolof DPO prefs from SFT")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "ortho_ns_mixed_v1" / "sft_stratified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "sft" / "preferences_v1.parquet",
    )
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def last_user_assistant(messages: list[dict]) -> tuple[str, str] | None:
    user = None
    assistant = None
    for message in messages:
        if message["role"] == "user":
            user = message["content"]
        elif message["role"] == "assistant":
            assistant = message["content"]
    if not user or not assistant:
        return None
    return user, assistant


def main() -> int:
    args = build_parser().parse_args()
    df = pd.read_parquet(args.input / "train.parquet")
    if args.smoke:
        df = df.head(200)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    pairs = []
    rejected_pool: list[str] = []
    for row in df.to_dict(orient="records"):
        try:
            messages = normalize_messages(row.get("messages"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        ua = last_user_assistant(messages)
        if not ua:
            continue
        prompt, chosen = ua
        rejected_pool.append(chosen)
        if len(rejected_pool) < 2:
            continue
        # Use a different assistant reply as weak rejected.
        rejected = rejected_pool[-(2 if len(rejected_pool) > 1 else 1)]
        if rejected == chosen:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(pairs) >= (200 if args.smoke else args.max_pairs):
            break
    out = pd.DataFrame(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    meta = {
        "created_at": utc_now(),
        "n_pairs": int(len(out)),
        "smoke": args.smoke,
        "note": "weak negative = shuffled other assistant replies; refine before production DPO",
    }
    atomic_json_write(args.output.with_suffix(".meta.json"), meta)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
