#!/usr/bin/env python3
"""P3.2 — Build stratified SFT mixture from galsenai/wolof_sft (train only).

Never touches the Hub test split. Carves val from selected train rows.

Example:
  uv run python scripts/data/21_build_sft_mixture.py --config configs/sft/mixture_v1.yaml
  uv run python scripts/data/21_build_sft_mixture.py --max-per-task 100   # dry-run / CI
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from huggingface_hub import get_token

from common.tracking import TrackingRun

ROOT = Path(__file__).resolve().parents[2]


def _require_token(env_name: str = "HF_TOKEN") -> str:
    token = os.environ.get(env_name) or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print(
            "ERROR: Hugging Face authentication is unavailable.\n"
            "  Private galsenai/wolof_sft requires authentication.\n"
            "  Authenticate first, e.g.:\n"
            "    export HF_TOKEN=hf_...\n"
            "    # or: uv run huggingface-cli login\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return token


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_by_task(
    df: pd.DataFrame,
    task_caps: dict,
    seed: int,
    max_per_task: int | None,
) -> pd.DataFrame:
    """Stratified sample: for each task, take up to min(cap, max_per_task, n)."""
    parts = []
    rng = seed
    for task, group in df.groupby("task", sort=False):
        task_key = str(task)
        cap = task_caps.get(task_key)
        if cap is None:
            # Unknown task: keep all unless dry-run cap applies
            n = len(group)
        else:
            n = int(cap)
        if max_per_task is not None:
            n = min(n, max_per_task)
        n = min(n, len(group))
        if n <= 0:
            continue
        parts.append(group.sample(n=n, random_state=rng))
        rng += 1
    if not parts:
        return df.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _carve_val(df: pd.DataFrame, val_pct: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified val carve by task; never uses Hub test."""
    if val_pct <= 0 or len(df) == 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    train_parts, val_parts = [], []
    rng = seed
    for _, group in df.groupby("task", sort=False):
        n_val = max(1, round(len(group) * val_pct)) if len(group) > 1 else 0
        n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
        if n_val <= 0:
            train_parts.append(group)
            continue
        val = group.sample(n=n_val, random_state=rng)
        train = group.drop(val.index)
        train_parts.append(train)
        val_parts.append(val)
        rng += 1

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed + 1)
        .reset_index(drop=True)
        if val_parts
        else df.iloc[0:0].copy()
    )
    return train_df, val_df


def _task_counts(df: pd.DataFrame) -> dict[str, int]:
    return dict(Counter(df["task"].astype(str).tolist()))


def _translation_pct(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    return counts.get("translation", 0) / total


def _load_train_only(dataset_id: str, token: str, loader=load_dataset):
    """Load exactly the source train split; the benchmark test split is forbidden."""
    return loader(dataset_id, split="train", token=token)


def _build_metadata(
    *,
    cfg_path: Path,
    dataset_id: str,
    seed: int,
    val_pct: float,
    max_per_task: int | None,
    task_caps: dict,
    n_hub_train: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selected: pd.DataFrame,
    translation_max_pct: float,
    train_path: Path,
    val_path: Path,
    notes,
) -> dict:
    selected_counts = _task_counts(selected)
    return {
        "schema_version": 1,
        "config": str(cfg_path.relative_to(ROOT)) if cfg_path.is_relative_to(ROOT) else str(cfg_path),
        "dataset_id": dataset_id,
        "source_split": "train",
        "seed": seed,
        "val_pct": val_pct,
        "max_per_task": max_per_task,
        "task_caps": task_caps,
        "n_hub_train": n_hub_train,
        "n_selected": len(selected),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "counts_selected": selected_counts,
        "counts_train": _task_counts(train_df),
        "counts_val": _task_counts(val_df),
        "translation_pct": _translation_pct(selected_counts),
        "translation_max_pct": translation_max_pct,
        "hub_test_touched": False,
        "paths": {
            "train": str(train_path.relative_to(ROOT)) if train_path.is_relative_to(ROOT) else str(train_path),
            "val": str(val_path.relative_to(ROOT)) if val_path.is_relative_to(ROOT) else str(val_path),
        },
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build stratified wolof_sft mixture")
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sft" / "mixture_v1.yaml",
    )
    ap.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Dry-run / CI: cap each task to at most N rows",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = _load_cfg(cfg_path)
    token = _require_token(args.hf_token_env)

    dataset_id = cfg.get("dataset_id", "galsenai/wolof_sft")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    val_pct = float(cfg.get("val_pct", 0.02))
    translation_max_pct = float(cfg.get("translation_max_pct", 0.35))
    task_caps = {str(k): v for k, v in (cfg.get("task_caps") or {}).items() if v is not None}

    out_dir = args.out_dir or Path(cfg.get("output_dir", "data/sft/mixture_v1"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tracking = TrackingRun(
        size="common",
        stage="sft-mixture",
        tokenizer_family="none",
        method="stratified",
        seed=seed,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", "mixture-v1"),
        config=cfg,
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=out_dir / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    print(f"Loading {dataset_id} train only (seed={seed}) …")
    ds = _load_train_only(dataset_id, token)
    df = ds.to_pandas()
    if "task" not in df.columns:
        raise SystemExit("ERROR: expected 'task' column on wolof_sft train")

    print(f"  hub train rows: {len(df):,}")
    selected = _sample_by_task(df, task_caps, seed=seed, max_per_task=args.max_per_task)
    print(f"  after task_caps: {len(selected):,}")

    train_df, val_df = _carve_val(selected, val_pct=val_pct, seed=seed)
    print(f"  mixture train: {len(train_df):,} | val: {len(val_df):,}")

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    meta = _build_metadata(
        cfg_path=cfg_path,
        dataset_id=dataset_id,
        seed=seed,
        val_pct=val_pct,
        max_per_task=args.max_per_task,
        task_caps=task_caps,
        n_hub_train=len(df),
        train_df=train_df,
        val_df=val_df,
        selected=selected,
        translation_max_pct=translation_max_pct,
        train_path=train_path,
        val_path=val_path,
        notes=cfg.get("notes"),
    )
    translation_pct = meta["translation_pct"]

    meta_path = out_dir / "mixture_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")
    print(f"Wrote {meta_path}")
    print(f"translation_pct={translation_pct:.4f} (max {translation_max_pct:.4f})")

    if translation_pct > translation_max_pct:
        print(
            f"WARN: translation_pct {translation_pct:.1%} > "
            f"translation_max_pct {translation_max_pct:.1%} — soft fail; "
            "tighten task_caps or reweight.",
            file=sys.stderr,
        )
        tracking.alert(
            "Mixture translation gate failed",
            f"translation percentage {translation_pct:.4f} exceeds limit",
            level="error",
        )
        tracking.finish(success=False)
        # Soft-fail: non-zero exit for CI visibility, but artifacts still written
        return 2

    tracking.log(
        {
            "rows_train": len(train_df),
            "rows_validation": len(val_df),
            "translation_percentage": translation_pct,
            **{
                f"task_count/{task}": count
                for task, count in meta["counts_selected"].items()
            },
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
