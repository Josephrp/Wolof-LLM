#!/usr/bin/env python3
"""Publish a validated model artifact privately and verify Hub readback."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from common.publication import (
    build_checksums,
    publishable_files,
    scan_for_secrets,
    select_namespace,
    sha256_file,
    validate_model_card,
    verify_remote_files,
)
from common.tracking import TrackingRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish and verify a private model artifact"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--stage", choices=("CPT", "Instruct"), required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "publication_manifest.json",
    )
    parser.add_argument("--preferred-namespace", default="galsenai")
    return parser


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "models": {}, "updated_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "publication" / "45_publish_training_assets.py"),
            "--preferred-namespace",
            args.preferred_namespace,
        ],
        cwd=ROOT,
        check=True,
    )
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    validate_model_card(artifact / "README.md")
    files = publishable_files(artifact)
    scan_for_secrets(artifact, files)
    checksums = build_checksums(artifact, files)
    checksum_path = artifact / "SHA256SUMS.json"
    checksum_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = publishable_files(artifact)
    scan_for_secrets(artifact, files)
    expected = build_checksums(artifact, files)

    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.preferred_namespace)
    repo_id = f"{namespace}/{args.repo_name}"
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    )
    tracking = TrackingRun(
        size=args.size,
        stage=f"publish-{args.stage.lower()}",
        tokenizer_family="artifact",
        method="private-hub",
        seed=42,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", args.repo_name),
        config={
            "size": args.size,
            "stage": args.stage,
            "repo_id": repo_id,
            "private": True,
        },
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=manifest_path.parent
        / "tracking"
        / f"publish-{args.size}-{args.stage.lower()}.json",
    ).start()
    atexit.register(tracking.finish, success=False)
    manifest = _load_manifest(manifest_path)
    existing = manifest.get("models", {}).get(args.size, {}).get(args.stage)
    if existing and existing.get("repo_id") != repo_id:
        raise RuntimeError(
            "publication manifest maps this size/stage to a different repository"
        )
    if not existing:
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=True, exist_ok=False
        )
    else:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if info.private is not True:
            raise RuntimeError("refusing to update a non-private repository")

    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=artifact,
        commit_message=f"Publish verified private {args.size} {args.stage} artifact",
        ignore_patterns=["*.log", "**/logs/**", "**/data/**", "**/.env"],
    )
    remote_files = api.list_repo_files(
        repo_id=repo_id, repo_type="model", revision=commit.oid
    )
    verify_remote_files(expected, remote_files)
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=commit.oid)
    if info.private is not True:
        raise RuntimeError("published repository is not private")

    readback_candidates = [
        name
        for name, metadata in expected.items()
        if Path(name).suffix.lower() in {".json", ".md", ".jinja"}
        and int(metadata["size"]) <= 2 * 1024 * 1024
    ]
    with tempfile.TemporaryDirectory(prefix="lfm25-hub-readback-") as temporary:
        for filename in readback_candidates:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=commit.oid,
                    token=token,
                    local_dir=temporary,
                )
            )
            if sha256_file(downloaded) != expected[filename]["sha256"]:
                raise RuntimeError(f"Hub readback checksum mismatch: {filename}")

    entry = {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": commit.oid,
        "private": True,
        "artifact": str(artifact.relative_to(ROOT)),
        "checksums": expected,
        "verified_files": readback_candidates,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    tracking.log(
        {
            "artifact_file_count": len(expected),
            "artifact_total_bytes": sum(
                int(metadata["size"]) for metadata in expected.values()
            ),
            "verified_readback_files": len(readback_candidates),
            "hub_revision_recorded": 1,
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    entry["trackio"] = tracking.metadata()
    manifest.setdefault("models", {}).setdefault(args.size, {})[args.stage] = entry
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json_write(manifest_path, manifest)
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
