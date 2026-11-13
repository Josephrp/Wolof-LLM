#!/usr/bin/env python3
"""Verify authenticated private Hub writes before expensive training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from common.publication import select_namespace, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and verify a private Hub smoke repo")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "hub_write_scope.json",
    )
    parser.add_argument("--preferred-namespace", default="galsenai")
    return parser


def run(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    api = HfApi(token=token)
    identity = api.whoami()
    namespace = select_namespace(identity, args.preferred_namespace)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    repo_id = f"{namespace}/lfm25-wolof-private-write-smoke-{timestamp}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="lfm25-write-scope-") as temporary:
        folder = Path(temporary)
        readme = folder / "README.md"
        metadata = folder / "smoke.json"
        readme.write_text(
            "# Private write-scope smoke\n\nPrivate production infrastructure check only.\n",
            encoding="utf-8",
        )
        metadata.write_text(
            json.dumps({"kind": "private-write-scope", "created_at": timestamp}) + "\n",
            encoding="utf-8",
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
            commit_message="Verify private production write scope",
        )
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=commit.oid))
        if not {"README.md", "smoke.json"} <= files:
            raise RuntimeError("private smoke repo readback is missing expected files")
        info = api.repo_info(repo_id=repo_id, repo_type="model", revision=commit.oid)
        if info.private is not True:
            raise RuntimeError("Hub smoke repo is not private")
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="smoke.json",
                revision=commit.oid,
                token=token,
            )
        )
        if sha256_file(downloaded) != sha256_file(metadata):
            raise RuntimeError("Hub smoke readback checksum mismatch")
    output = args.out if args.out.is_absolute() else ROOT / args.out
    atomic_json_write(
        output,
        {
            "schema_version": 1,
            "namespace": namespace,
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "revision": commit.oid,
            "private": True,
            "verified_at": datetime.now(UTC).isoformat(),
        },
    )
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
