#!/usr/bin/env python3
"""Publish Tiny Aya Wolof family processed datasets to private Hub repos."""

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
        "TinyAya-Wolof-NTP-ortho_ns_mixed_v1",
    ),
    "sft": (
        ROOT / "data" / "ortho_ns_mixed_v1" / "sft_stratified",
        "TinyAya-Wolof-SFT-ortho_ns_mixed_stratified_v1",
    ),
    "prepared": (
        ROOT / "data" / "sft" / "prepared_anchored_v1",
        "TinyAya-Wolof-SFT-prepared_anchored_v1",
    ),
    "prefs": (
        ROOT / "data" / "sft" / "preferences_v1.parquet",
        "TinyAya-Wolof-prefs_v1",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish Tiny Aya Wolof datasets")
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--skip-cpt", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-prepared", action="store_true")
    parser.add_argument("--skip-prefs", action="store_true")
    return parser


def publish_dir(api: HfApi, local: Path, repo_id: str, private: bool) -> None:
    if not local.is_dir():
        raise SystemExit(f"missing dataset dir: {local}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload Tiny Aya Wolof family dataset ({utc_now()})",
    )
    print(f"published {local} -> {repo_id}", flush=True)


def publish_prefs_file(api: HfApi, local: Path, repo_id: str, private: bool) -> None:
    if not local.is_file():
        print(f"skip prefs (missing): {local}", flush=True)
        return
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=local.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload Tiny Aya Wolof prefs ({utc_now()})",
    )
    print(f"published {local} -> {repo_id}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)
    private = not args.public
    if not args.skip_cpt:
        local, name = DEFAULTS["cpt"]
        publish_dir(api, local, f"{args.namespace}/{name}", private)
    if not args.skip_sft:
        local, name = DEFAULTS["sft"]
        publish_dir(api, local, f"{args.namespace}/{name}", private)
    if not args.skip_prepared:
        local, name = DEFAULTS["prepared"]
        publish_dir(api, local, f"{args.namespace}/{name}", private)
    if not args.skip_prefs:
        local, name = DEFAULTS["prefs"]
        publish_prefs_file(api, local, f"{args.namespace}/{name}", private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
