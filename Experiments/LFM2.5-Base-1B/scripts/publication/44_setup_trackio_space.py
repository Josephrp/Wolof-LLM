#!/usr/bin/env python3
"""Create/verify a private Trackio Space and persist its non-secret ID."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import trackio
from huggingface_hub import (
    HfApi,
    bucket_info,
    download_bucket_files,
    get_token,
    list_bucket_tree,
)
from huggingface_hub.errors import BucketNotFoundError
from trackio.fragments import parse_fragment_bytes
from trackio.remote_client import RemoteClient
from trackio.server import get_run_summary
from trackio.sqlite_storage import SQLiteStorage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from common.publication import select_namespace
from common.tracking import PROJECT, deterministic_run_name


def resolve_space_id(namespace: str, space_name: str, configured: str | None) -> str:
    """Reuse the preflight-verified Space when production exports its ID."""
    if configured:
        if configured.count("/") != 1:
            raise ValueError("TRACKIO_SPACE_ID must use namespace/name format")
        return configured
    return f"{namespace}/{space_name}"


def classify_dashboard_status(stage: str, paused_status: str) -> str:
    """Keep dashboard compute separate from the remote-persistence gate."""
    normalized = stage.upper()
    if normalized == "RUNNING":
        return "running"
    if normalized == "PAUSED":
        return paused_status
    return normalized.lower()


def inspect_dashboard(api, space_id: str, paused_status: str) -> dict:
    try:
        runtime = api.get_space_runtime(space_id)
        raw_stage = getattr(runtime.stage, "value", runtime.stage)
        stage = str(raw_stage).upper()
        return {
            "stage": stage,
            "dashboard_status": classify_dashboard_status(stage, paused_status),
        }
    except Exception as exc:  # noqa: BLE001 - dashboard compute is optional
        return {
            "stage": "UNKNOWN",
            "dashboard_status": "runtime_query_unavailable",
            "warning": type(exc).__name__,
        }


def ensure_private_bucket(api, bucket_id: str) -> None:
    """Create the Trackio Bucket privately or correct Trackio's public default."""
    try:
        info = api.bucket_info(bucket_id)
    except BucketNotFoundError:
        api.create_bucket(bucket_id, private=True)
    else:
        if info.private is not True:
            api.update_bucket_settings(bucket_id, private=True)
    if api.bucket_info(bucket_id).private is not True:
        raise RuntimeError("Trackio Bucket could not be made private")


def verify_bucket_persistence(
    *,
    bucket_id: str,
    project: str,
    run_name: str,
    run_id: str,
    token: str,
    timeout: float = 120,
) -> dict:
    """Download and parse the exact immutable metric fragment from a private Bucket."""
    info = bucket_info(bucket_id, token=token)
    if info.private is not True:
        raise RuntimeError("Trackio Bucket is not private")
    deadline = time.monotonic() + timeout
    while True:
        files = [
            item
            for item in list_bucket_tree(
                bucket_id,
                prefix="trackio/inbox",
                recursive=True,
                token=token,
            )
            if getattr(item, "type", None) == "file"
            and str(getattr(item, "path", "")).endswith(".jsonl")
        ]
        for item in files:
            with tempfile.TemporaryDirectory(prefix="trackio-bucket-readback-") as temporary:
                destination = Path(temporary) / "fragment.jsonl"
                download_bucket_files(
                    bucket_id,
                    files=[(item, destination)],
                    token=token,
                )
                payload = destination.read_bytes()
            for record in parse_fragment_bytes(payload):
                if (
                    record.get("kind") == "metric"
                    and record.get("project") == project
                    and record.get("run") == run_name
                    and record.get("run_id") == run_id
                    and record.get("metrics", {}).get("sync_smoke") == 1
                    and record.get("metrics", {}).get("gate_status") == 1
                ):
                    return {
                        "status": "verified",
                        "bucket_id": bucket_id,
                        "private": True,
                        "bucket_size": info.size,
                        "bucket_file_count": info.total_files,
                        "fragment_path": item.path,
                        "fragment_size": len(payload),
                        "fragment_sha256": hashlib.sha256(payload).hexdigest(),
                        "fragment_revision": item.xet_hash,
                        "record_log_id": record.get("log_id"),
                        "record_step": record.get("step"),
                        "metrics": record["metrics"],
                    }
        if time.monotonic() >= deadline:
            raise RuntimeError("Trackio smoke run was not readable from the private Bucket")
        time.sleep(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Set up private Trackio Space persistence")
    parser.add_argument("--preferred-namespace", default="galsenai")
    parser.add_argument("--space-name", default="lfm25-wolof-trackio")
    parser.add_argument(
        "--paused-dashboard-status",
        choices=("paused", "paused_quota"),
        default="paused",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "trackio_space.json",
    )
    parser.add_argument(
        "--env-out",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "trackio_env.json",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.preferred_namespace)
    space_id = resolve_space_id(
        namespace,
        args.space_name,
        os.environ.get("TRACKIO_SPACE_ID"),
    )
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=True,
        exist_ok=True,
    )
    info = api.repo_info(repo_id=space_id, repo_type="space")
    if info.private is not True:
        raise RuntimeError("Trackio Space is not private")
    dashboard = inspect_dashboard(api, space_id, args.paused_dashboard_status)
    bucket_id = os.environ.get("TRACKIO_BUCKET_ID") or f"{space_id}-bucket"
    ensure_private_bucket(api, bucket_id)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = deterministic_run_name(
        size="common",
        stage="trackio-smoke",
        tokenizer_family="none",
        method="sync",
        seed=42,
        run_id=timestamp,
    )
    run = trackio.init(
        project=PROJECT,
        name=run_name,
        group="common/trackio-smoke",
        config={"kind": "private-space-sync-smoke", "seed": 42},
        space_id=space_id,
        bucket_id=bucket_id,
    )
    run_id = str(getattr(run, "id", run_name))
    if dashboard["dashboard_status"] != "running":
        message = (
            f"Live Trackio dashboard unavailable "
            f"({dashboard['dashboard_status']}); using verified private Bucket persistence"
        )
        warnings.warn(message, stacklevel=1)
        trackio.alert(
            title="Trackio dashboard degraded",
            text=message,
            level=trackio.AlertLevel.WARN,
        )
    trackio.log({"sync_smoke": 1, "gate_status": 1}, step=0)
    trackio.finish()
    persistence = verify_bucket_persistence(
        bucket_id=bucket_id,
        project=PROJECT,
        run_name=run_name,
        run_id=run_id,
        token=token,
    )
    local_db = SQLiteStorage.get_project_db_path(PROJECT)
    local_summary = get_run_summary(PROJECT, run_name, run_id=run_id)
    if not local_db.is_file() or local_db.stat().st_size == 0:
        raise RuntimeError("Trackio durable local database is missing")
    if local_summary.get("run") != run_name:
        raise RuntimeError("Trackio durable local query did not return the smoke run")

    live_query = {"status": "skipped", "reason": dashboard["dashboard_status"]}
    if dashboard["dashboard_status"] == "running":
        try:
            queried = RemoteClient(space_id, hf_token=token).predict(
                project=PROJECT,
                run=run_name,
                run_id=run_id,
                api_name="/get_run_summary",
            )
            if not isinstance(queried, dict) or queried.get("run") != run_name:
                raise RuntimeError("live dashboard query returned the wrong run")
            live_query = {"status": "verified"}
        except Exception as exc:  # noqa: BLE001 - live dashboard is optional
            live_query = {"status": "unavailable", "warning": type(exc).__name__}

    files = api.list_repo_files(repo_id=space_id, repo_type="space")
    revision = api.repo_info(repo_id=space_id, repo_type="space").sha
    dashboard_url = f"https://huggingface.co/spaces/{space_id}"
    output = args.out if args.out.is_absolute() else ROOT / args.out
    env_output = args.env_out if args.env_out.is_absolute() else ROOT / args.env_out
    atomic_json_write(
        output,
        {
            "schema_version": 2,
            "project": PROJECT,
            "group": "common/trackio-smoke",
            "run_name": run_name,
            "run_id": run_id,
            "space_id": space_id,
            "dashboard_url": dashboard_url,
            "private": True,
            "remote_persistence_gate": "pass",
            "live_dashboard_gate": live_query["status"],
            "dashboard": dashboard,
            "live_query": live_query,
            "bucket": persistence,
            "local_state": {
                "database_exists": True,
                "database_size": local_db.stat().st_size,
                "database_sha256": hashlib.sha256(local_db.read_bytes()).hexdigest(),
                "run_query": "verified",
            },
            "sync_status": "verified_remote_bucket",
            "revision": revision,
            "remote_file_count": len(files),
            "verified_at": datetime.now(UTC).isoformat(),
        },
    )
    atomic_json_write(
        env_output,
        {"TRACKIO_SPACE_ID": space_id, "TRACKIO_BUCKET_ID": bucket_id},
    )
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
