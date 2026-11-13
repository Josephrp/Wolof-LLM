#!/usr/bin/env python3
"""Materialize galsenai/wolof_sft_cleaned_v1_stratified into local train/val parquet."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default="galsenai/wolof_sft_cleaned_v1_stratified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "sft" / "stratified_v1",
    )
    parser.add_argument(
        "--val-from",
        choices=("test", "carve"),
        default="test",
        help="Use Hub test split as val (default) or carve 2%% from train",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_id, token=token)
    train = ds["train"].to_pandas()
    if args.val_from == "test":
        val = ds["test"].to_pandas()
        val_source = "hub_test_as_val"
    else:
        # stratified carve by task
        import pandas as pd

        parts_tr, parts_va = [], []
        for _, group in train.groupby("task", sort=False):
            n_val = max(1, round(len(group) * 0.02)) if len(group) > 1 else 0
            n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
            if n_val <= 0:
                parts_tr.append(group)
                continue
            va = group.sample(n=n_val, random_state=42)
            parts_va.append(va)
            parts_tr.append(group.drop(va.index))
        train = pd.concat(parts_tr, ignore_index=True).sample(frac=1.0, random_state=42)
        val = pd.concat(parts_va, ignore_index=True).sample(frac=1.0, random_state=43)
        val_source = "carve_2pct_from_train"

    train_path = out / "train.parquet"
    val_path = out / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    meta = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "val_from": args.val_from,
        "val_source": val_source,
        "n_train": len(train),
        "n_val": len(val),
        "task_counts_train": train["task"].value_counts().to_dict(),
        "task_counts_val": val["task"].value_counts().to_dict(),
        "updated_at": datetime.now(UTC).isoformat(),
        "paths": {"train": str(train_path), "val": str(val_path)},
    }
    (out / "mixture_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
