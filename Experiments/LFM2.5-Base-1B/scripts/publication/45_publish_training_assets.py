#!/usr/bin/env python3
"""Publish resumable checkpoints and tokenizers to private Hub model repos."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from common.publication import (
    TEXT_SUFFIXES,
    publishable_files,
    scan_for_secrets,
    select_namespace,
)

TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
PRIVATE_ROOT_PATTERN = re.compile(r"/mnt/(?:volume|data)[^/\s\"']*/")
# Hub rejects model-card base_model values that are local paths (incl. redacted).
HUB_BASE_BY_SIZE = (
    (re.compile(r"(?:^|[^0-9])350m(?:[^0-9]|$)|LFM2\.5-350M", re.I), "LiquidAI/LFM2.5-350M-Base"),
    (re.compile(r"(?:^|[^0-9])230m(?:[^0-9]|$)|LFM2\.5-230M", re.I), "LiquidAI/LFM2.5-230M-Base"),
    (re.compile(r"(?:^|[^0-9])2\.?6b(?:[^0-9]|$)|LFM2\.5-2\.?6B", re.I), "LiquidAI/LFM2.5-2.6B-Base"),
    (re.compile(r"(?:^|[^0-9])8b(?:[^0-9]|$)|LFM2\.5-8B|moe", re.I), "LiquidAI/LFM2.5-8B-A1B"),
    (re.compile(r"(?:^|[^0-9])1\.?2b(?:[^0-9]|$)|LFM2\.5-1\.?2B|lfm25-wolof-ext", re.I), "LiquidAI/LFM2.5-1.2B-Base"),
)
LOCAL_PATH_PREFIX = r"(?:<REDACTED_PRIVATE_ROOT>|/mnt/|[A-Za-z]:\\\\|/home/)"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")


def _hub_base_model_for_path(path: str) -> str:
    for pattern, hub_id in HUB_BASE_BY_SIZE:
        if pattern.search(path):
            return hub_id
    return "LiquidAI/LFM2.5-1.2B-Base"


def _rewrite_local_base_models(text: str) -> str:
    text = re.sub(
        rf"^base_model:\s*({LOCAL_PATH_PREFIX}.+)$",
        lambda match: f"base_model: {_hub_base_model_for_path(match.group(1))}",
        text,
        flags=re.M,
    )
    text = re.sub(
        rf"base_model:adapter:({LOCAL_PATH_PREFIX}[^\s\"']+)",
        lambda match: f"base_model:adapter:{_hub_base_model_for_path(match.group(1))}",
        text,
    )
    text = re.sub(
        rf'"base_model_name_or_path"\s*:\s*"({LOCAL_PATH_PREFIX}[^"]+)"',
        lambda match: f'"base_model_name_or_path": "{_hub_base_model_for_path(match.group(1))}"',
        text,
    )
    return text


def discover_checkpoints(root: Path) -> list[Path]:
    return sorted(
        {
            path
            for path in root.rglob("checkpoint-*")
            if path.is_dir() and (path / "trainer_state.json").is_file()
        }
    )


def discover_tokenizers(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "tokenizer_config.json").is_file()
    )


@contextlib.contextmanager
def sanitized_staging(source: Path):
    """Hard-link large assets and redact coordinates in copied text metadata."""
    staging_root = ROOT / "results" / ".asset_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
        staged = Path(temporary) / source.name
        shutil.copytree(source, staged, copy_function=os.link)
        for path in publishable_files(staged):
            if (
                path.suffix.lower() not in TEXT_SUFFIXES
                or path.stat().st_size > 10 * 1024 * 1024
            ):
                continue
            original = path.read_text(encoding="utf-8", errors="replace")
            redacted = TOKEN_PATTERN.sub("<REDACTED_TOKEN>", original)
            redacted = PRIVATE_ROOT_PATTERN.sub("<REDACTED_PRIVATE_ROOT>/", redacted)
            redacted = _rewrite_local_base_models(redacted)
            if redacted != original:
                path.unlink()
                path.write_text(redacted, encoding="utf-8")
        scan_for_secrets(staged)
        yield staged


def _ensure_private_repo(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    if api.model_info(repo_id).private is not True:
        raise RuntimeError(f"refusing to upload to non-private repository: {repo_id}")


def _verify_readback(
    api: HfApi, token: str, repo_id: str, revision: str, filename: str
) -> None:
    files = set(api.list_repo_files(repo_id, revision=revision))
    if filename not in files:
        raise RuntimeError(f"Hub readback missing {filename} in {repo_id}")
    with tempfile.TemporaryDirectory(prefix="lfm25-asset-readback-") as temporary:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            local_dir=temporary,
        )
        if Path(downloaded).stat().st_size <= 0:
            raise RuntimeError(f"Hub readback is empty: {repo_id}/{filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--tokenizers-root", type=Path, default=ROOT / "tokenizers")
    parser.add_argument("--preferred-namespace", default="galsenai")
    parser.add_argument(
        "--checkpoint-repo-name",
        default="LFM2.5-Wolof-Training-Checkpoints",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "training_assets_manifest.json",
    )
    parser.add_argument("--checkpoints-only", action="store_true")
    parser.add_argument("--tokenizers-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.checkpoints_only and args.tokenizers_only:
        raise ValueError("choose at most one asset-only option")
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.preferred_namespace)
    checkpoints: list[dict[str, object]] = []
    tokenizers: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "private": True,
        "checkpoints": checkpoints,
        "tokenizers": tokenizers,
        "verified_at": None,
    }

    if not args.tokenizers_only:
        checkpoint_repo = f"{namespace}/{args.checkpoint_repo_name}"
        _ensure_private_repo(api, checkpoint_repo)
        remote_files = set(api.list_repo_files(checkpoint_repo))
        for checkpoint in discover_checkpoints(args.runs_root):
            relative = checkpoint.relative_to(args.runs_root).as_posix()
            with sanitized_staging(checkpoint) as staged:
                expected = {
                    f"{relative}/{path.relative_to(staged).as_posix()}"
                    for path in publishable_files(staged)
                }
                if expected <= remote_files:
                    revision = api.model_info(checkpoint_repo).sha
                else:
                    commit = api.upload_folder(
                        repo_id=checkpoint_repo,
                        folder_path=staged,
                        path_in_repo=relative,
                        commit_message=f"Back up resumable checkpoint {relative}",
                    )
                    revision = commit.oid
                    remote_files.update(expected)
            state_file = f"{relative}/trainer_state.json"
            _verify_readback(api, token, checkpoint_repo, revision, state_file)
            checkpoints.append(
                {
                    "repo_id": checkpoint_repo,
                    "path": relative,
                    "revision": revision,
                    "private": True,
                    "verified_file": state_file,
                }
            )

    if not args.checkpoints_only:
        for tokenizer in discover_tokenizers(args.tokenizers_root):
            repo_id = f"{namespace}/LFM2.5-Wolof-Tokenizer-{_slug(tokenizer.name)}"
            _ensure_private_repo(api, repo_id)
            with sanitized_staging(tokenizer) as staged:
                expected = {
                    path.relative_to(staged).as_posix()
                    for path in publishable_files(staged)
                }
                if expected <= set(api.list_repo_files(repo_id)):
                    revision = api.model_info(repo_id).sha
                else:
                    commit = api.upload_folder(
                        repo_id=repo_id,
                        folder_path=staged,
                        commit_message=f"Publish private tokenizer {tokenizer.name}",
                    )
                    revision = commit.oid
            _verify_readback(api, token, repo_id, revision, "tokenizer_config.json")
            tokenizers.append(
                {
                    "repo_id": repo_id,
                    "source": tokenizer.name,
                    "revision": revision,
                    "private": True,
                    "verified_file": "tokenizer_config.json",
                }
            )

    manifest["verified_at"] = datetime.now(UTC).isoformat()
    output = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    atomic_json_write(output, manifest)
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
