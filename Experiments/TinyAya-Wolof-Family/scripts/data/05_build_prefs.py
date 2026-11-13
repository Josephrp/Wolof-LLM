#!/usr/bin/env python3
"""Build simple DPO preference pairs from filtered SFT (LFM-style columns)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.anchoring import normalize_messages
from common.ortho_variants import get_variant, resolve_variants
from common.pipeline import atomic_json_write, utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Wolof DPO prefs from SFT")
    parser.add_argument(
        "--ortho",
        default="both",
        help="ns | std | both",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override SFT stratified input (single ortho)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override prefs parquet path (single ortho)",
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


def build_prefs(
    *,
    input_dir: Path,
    output: Path,
    smoke: bool,
    seed: int,
    max_pairs: int,
    ortho: str,
) -> dict:
    df = pd.read_parquet(input_dir / "train.parquet")
    if smoke:
        df = df.head(200)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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
        rejected = rejected_pool[-(2 if len(rejected_pool) > 1 else 1)]
        if rejected == chosen:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(pairs) >= (200 if smoke else max_pairs):
            break
    out = pd.DataFrame(pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    meta = {
        "created_at": utc_now(),
        "ortho": ortho,
        "n_pairs": int(len(out)),
        "smoke": smoke,
        "input": str(input_dir),
        "note": "weak negative = shuffled other assistant replies; refine before production DPO",
    }
    atomic_json_write(output.with_suffix(".meta.json"), meta)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> int:
    args = build_parser().parse_args()
    keys = resolve_variants(args.ortho)
    if (args.input is not None or args.output is not None) and len(keys) != 1:
        raise SystemExit("--input/--output require a single --ortho")
    for key in keys:
        variant = get_variant(key)
        input_dir = args.input
        if input_dir is None:
            input_dir = ROOT / "data" / variant["data_dirname"] / "sft_stratified"
        elif not input_dir.is_absolute():
            input_dir = ROOT / input_dir
        output = args.output
        if output is None:
            name = "preferences_v1.parquet" if key == "ns" else "preferences_std_v1.parquet"
            output = ROOT / "data" / "sft" / name
        elif not output.is_absolute():
            output = ROOT / output
        build_prefs(
            input_dir=input_dir,
            output=output,
            smoke=args.smoke,
            seed=args.seed,
            max_pairs=args.max_pairs,
            ortho=key,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
