#!/usr/bin/env python3
"""Publish Qwen3.5 Wolof family processed datasets to private Hub repos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import utc_now

DEFAULTS = {
    "cpt": (
        ROOT / "data" / "ortho_ns_mixed_v1" / "cpt",
        "Qwen3.5-Wolof-NTP-ortho_ns_mixed_v1",
    ),
    "sft": (
        ROOT / "data" / "ortho_ns_mixed_v1" / "sft_stratified",
        "Qwen3.5-Wolof-SFT-ortho_ns_mixed_stratified_v1",
    ),
    "prepared": (
        ROOT / "data" / "sft" / "prepared_anchored_v1",
        "Qwen3.5-Wolof-SFT-prepared_anchored_v1",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish Qwen Wolof datasets")
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--skip-cpt", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-prepared", action="store_true")
    return parser


def publish_dir(api: HfApi, local: Path, repo_id: str, private: bool) -> None:
    if not local.is_dir():
        raise SystemExit(f"missing dataset dir: {local}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload Qwen3.5 Wolof family dataset ({utc_now()})",
    )
    print(f"published {local} -> {repo_id}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)
    private = not args.public
    mapping = []
    if not args.skip_cpt:
        mapping.append(DEFAULTS["cpt"])
    if not args.skip_sft:
        mapping.append(DEFAULTS["sft"])
    if not args.skip_prepared:
        mapping.append(DEFAULTS["prepared"])
    for local, name in mapping:
        publish_dir(api, local, f"{args.namespace}/{name}", private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
