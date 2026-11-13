"""Refuse production training when local caches are still smoke-capped."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

# Full Hub splits (approx); smoke caps at 2000.
MIN_NTP_TRAIN = 100_000
MIN_SFT_TRAIN = 50_000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def assert_production_data(root: Path | None = None) -> dict[str, Any]:
    """Exit hard if materialize/ortho caches look like a smoke run."""
    base = root or ROOT
    meta_path = base / "data" / "raw" / "materialize_meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            "production refused: missing data/raw/materialize_meta.json — "
            "run scripts/data/01_materialize_sources.py without --smoke"
        )
    meta = _read_json(meta_path)
    if meta.get("smoke"):
        raise SystemExit(
            "production refused: materialize_meta.json has smoke=true — "
            "rematerialize full NTP/SFT splits without --smoke"
        )

    ntp_train = base / "data" / "raw" / "ntp_70_20_10" / "train.parquet"
    sft_train = base / "data" / "raw" / "sft_stratified_v1" / "train.parquet"
    if not ntp_train.is_file() or not sft_train.is_file():
        raise SystemExit("production refused: missing raw NTP/SFT train parquet")

    ntp_n = _parquet_rows(ntp_train)
    sft_n = _parquet_rows(sft_train)
    if ntp_n < MIN_NTP_TRAIN:
        raise SystemExit(
            f"production refused: NTP train rows={ntp_n} < {MIN_NTP_TRAIN} (smoke-sized?)"
        )
    if sft_n < MIN_SFT_TRAIN:
        raise SystemExit(
            f"production refused: SFT train rows={sft_n} < {MIN_SFT_TRAIN} (smoke-sized?)"
        )

    for name in ("ortho_ns_mixed_summary.json", "ortho_std_mixed_summary.json"):
        path = base / "data" / name
        if not path.is_file():
            continue
        summary = _read_json(path)
        if summary.get("smoke"):
            raise SystemExit(
                f"production refused: {name} has smoke=true — re-run ortho filter without --smoke"
            )

    report = {
        "ntp_train_rows": ntp_n,
        "sft_train_rows": sft_n,
        "materialize_smoke": bool(meta.get("smoke")),
    }
    print(f"DATA_GATE_OK {report}", flush=True)
    return report
