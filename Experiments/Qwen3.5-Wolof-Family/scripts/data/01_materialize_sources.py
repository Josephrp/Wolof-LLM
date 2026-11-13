#!/usr/bin/env python3
"""Cache NTP 70/20/10 and stratified SFT locally for the Qwen3.5 Wolof family."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now

NTP_REPO = "galsenai/ntp_data_for_multilingual"
NTP_VARIANT = "NTP_data_for_multilingual_70_20_10"
SFT_REPO = "galsenai/wolof_sft_cleaned_v1_stratified"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize NTP + stratified SFT")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get(args.hf_token_env) or get_token()
    out = args.out_root
    ntp_dir = out / "ntp_70_20_10"
    sft_dir = out / "sft_stratified_v1"
    ntp_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)

    ntp_files = {
        "train": f"{NTP_VARIANT}/train.parquet",
        "test": f"{NTP_VARIANT}/test.parquet",
        "provenance": f"{NTP_VARIANT}/provenance.parquet",
    }
    for name, rel in ntp_files.items():
        print(f"loading NTP {rel}", flush=True)
        ds = load_dataset(NTP_REPO, data_files=rel, split="train", token=token)
        if args.smoke:
            ds = ds.select(range(min(2000 if name != "provenance" else 5000, len(ds))))
        path = ntp_dir / f"{name}.parquet"
        ds.to_parquet(str(path))
        print(f"  wrote {path} rows={len(ds)}", flush=True)

    print(f"loading SFT {SFT_REPO}", flush=True)
    sft = load_dataset(SFT_REPO, token=token)
    for split in ("train", "test"):
        part = sft[split]
        if args.smoke:
            part = part.select(range(min(2000, len(part))))
        path = sft_dir / f"{split}.parquet"
        part.to_parquet(str(path))
        print(f"  wrote {path} rows={len(part)}", flush=True)

    meta = {
        "created_at": utc_now(),
        "ntp_repo": NTP_REPO,
        "ntp_variant": NTP_VARIANT,
        "sft_repo": SFT_REPO,
        "smoke": args.smoke,
        "ntp_dir": str(ntp_dir),
        "sft_dir": str(sft_dir),
    }
    atomic_json_write(out / "materialize_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
