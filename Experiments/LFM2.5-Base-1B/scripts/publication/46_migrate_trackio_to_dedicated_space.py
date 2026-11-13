#!/usr/bin/env python3
"""Migrate Trackio history into a dedicated Space + private dataset.

Copies immutable Bucket inbox fragments from legacy Buckets into:
  1) the local Trackio SQLite project DB (without deleting source Buckets)
  2) a private Hugging Face Dataset snapshot for durable retrieval
  3) a new dedicated private Trackio Space + Bucket for remaining runs

Source Buckets are never deleted. Dataset persistence remains available for
retrieval even though Trackio prefers Buckets for live Spaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import trackio
from huggingface_hub import (
    HfApi,
    batch_bucket_files,
    bucket_info,
    create_repo,
    download_bucket_files,
    get_token,
    list_bucket_tree,
    upload_folder,
)
from huggingface_hub.errors import BucketNotFoundError
from trackio.bucket_storage import upload_project_to_bucket
from trackio.deploy import deploy_as_space
from trackio.fragments import (
    BUCKET_INBOX_PREFIX,
    import_records,
    parse_fragment_bytes,
)
from trackio.remote_client import RemoteClient
from trackio.server import get_run_summary
from trackio.sqlite_storage import SQLiteStorage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from common.publication import select_namespace
from common.tracking import PROJECT, deterministic_run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Port Trackio Bucket history to a dedicated Space and dataset"
    )
    parser.add_argument("--preferred-namespace", default="galsenai")
    parser.add_argument("--space-name", default="lfm25-wolof-trackio")
    parser.add_argument(
        "--dataset-name",
        default="lfm25-wolof-trackio-history",
        help="Private dataset that stores recovered Trackio history",
    )
    parser.add_argument(
        "--source-bucket",
        action="append",
        default=None,
        help="Legacy Bucket to import (repeatable). Defaults to known LFM buckets.",
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
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "trackio_migration.json",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Reuse an existing downloaded history cache instead of re-fetching Buckets",
    )
    parser.add_argument(
        "--skip-bucket-seed",
        action="store_true",
        help="Skip reseeding the destination Bucket when fragments were already copied",
    )
    return parser


def _default_source_buckets(namespace: str) -> list[str]:
    return [
        f"{namespace}/trackio-monitoring-dev-bucket",
        f"{namespace}/trackio-monitoring-dev-bucket-2",
        "Tonic/trackio-monitoring-dev-bucket",
        "Tonic/trackio-monitoring-dev-bucket-2",
    ]


def ensure_private_bucket(api: HfApi, bucket_id: str) -> None:
    try:
        info = api.bucket_info(bucket_id)
    except BucketNotFoundError:
        api.create_bucket(bucket_id, private=True)
    else:
        if info.private is not True:
            api.update_bucket_settings(bucket_id, private=True)
    if api.bucket_info(bucket_id).private is not True:
        raise RuntimeError(f"Trackio Bucket is not private: {bucket_id}")


def list_inbox_paths(bucket_id: str, token: str) -> list[str]:
    try:
        items = list(
            list_bucket_tree(
                bucket_id,
                prefix=BUCKET_INBOX_PREFIX,
                recursive=True,
                token=token,
            )
        )
    except Exception as exc:  # noqa: BLE001 - missing legacy buckets are OK
        print(f"skip bucket {bucket_id}: {type(exc).__name__}: {exc}")
        return []
    return sorted(
        str(item.path)
        for item in items
        if getattr(item, "type", None) == "file"
        and str(getattr(item, "path", "")).endswith(".jsonl")
    )


def download_bucket_history(
    *,
    bucket_ids: list[str],
    destination: Path,
    token: str,
) -> dict:
    """Download inbox fragments without deleting the source Buckets."""
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for bucket_id in bucket_ids:
        paths = list_inbox_paths(bucket_id, token)
        if not paths:
            continue
        bucket_dir = destination / bucket_id.replace("/", "__")
        bucket_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        bytes_total = 0
        for index, remote_path in enumerate(paths):
            safe_name = remote_path.replace("/", "__")
            local_path = bucket_dir / f"{index:06d}_{safe_name}"
            try:
                download_bucket_files(
                    bucket_id,
                    files=[(remote_path, str(local_path))],
                    token=token,
                )
            except Exception as exc:  # noqa: BLE001 - continue best-effort
                print(f"download failed {bucket_id}/{remote_path}: {exc}")
                if local_path.exists():
                    local_path.unlink()
                continue
            if not local_path.is_file() or local_path.stat().st_size == 0:
                print(f"empty download skipped {bucket_id}/{remote_path}")
                if local_path.exists():
                    local_path.unlink()
                continue
            downloaded += 1
            bytes_total += local_path.stat().st_size
        summary[bucket_id] = {
            "files_listed": len(paths),
            "files_downloaded": downloaded,
            "bytes": bytes_total,
            "local_dir": str(bucket_dir),
        }
        print(
            f"downloaded {downloaded}/{len(paths)} fragments from {bucket_id} "
            f"({bytes_total} bytes)"
        )
    return summary


def import_local_fragments(history_root: Path) -> dict:
    """Import downloaded JSONL into the local Trackio SQLite DB (idempotent).

    Legacy remote logging often used the default ``huggingface`` project name.
    Those records are remapped into ``PROJECT`` so the dedicated Space shows one
    unified history for remaining runs.
    """
    records: list[dict] = []
    fragment_count = 0
    remapped = 0
    for path in sorted(history_root.rglob("*.jsonl")):
        if not path.is_file():
            continue
        # Dataset archives also ship a concatenated metrics_flat.jsonl; prefer
        # the per-fragment inbox copies to avoid double counting when both exist.
        if path.name == "metrics_flat.jsonl":
            continue
        try:
            parsed = parse_fragment_bytes(path.read_bytes())
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001 - skip corrupt fragments
            print(f"parse failed {path}: {exc}")
            continue
        if not parsed:
            continue
        fragment_count += 1
        for record in parsed:
            project = record.get("project")
            if project and project != PROJECT:
                remapped += 1
                record = {**record, "project": PROJECT}
            records.append(record)
    imported = import_records(records) if records else 0
    runs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        project = record.get("project")
        run = record.get("run")
        if project and run:
            runs[str(project)].add(str(run))
    return {
        "fragments": fragment_count,
        "records": len(records),
        "imported": imported,
        "remapped_to_project": remapped,
        "projects": {project: sorted(names) for project, names in runs.items()},
        "run_count": sum(len(names) for names in runs.values()),
    }


def copy_fragments_to_bucket(
    *,
    history_root: Path,
    bucket_id: str,
    token: str,
    batch_size: int = 25,
) -> int:
    """Seed the new Space Bucket with recovered inbox fragments (no source deletes)."""
    paths = sorted(history_root.rglob("*.jsonl"))
    uploaded = 0
    batch: list[tuple[bytes, str]] = []
    writer = "migrated"
    for index, path in enumerate(paths):
        remote = f"{BUCKET_INBOX_PREFIX}/{writer}/{index:08d}.jsonl"
        batch.append((path.read_bytes(), remote))
        if len(batch) >= batch_size:
            batch_bucket_files(bucket_id, add=batch, token=token)
            uploaded += len(batch)
            batch = []
    if batch:
        batch_bucket_files(bucket_id, add=batch, token=token)
        uploaded += len(batch)
    return uploaded


def export_history_dataset(
    *,
    api: HfApi,
    dataset_id: str,
    history_root: Path,
    token: str,
) -> dict:
    """Publish recovered fragments + optional parquet snapshot to a private dataset."""
    create_repo(
        dataset_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
    if info.private is not True:
        raise RuntimeError(f"Trackio history dataset is not private: {dataset_id}")

    with tempfile.TemporaryDirectory(prefix="trackio-history-") as temporary:
        package = Path(temporary)
        raw = package / "bucket_inbox"
        if history_root.exists():
            shutil.copytree(history_root, raw, dirs_exist_ok=True)
        parquet_dir = package / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_warning = None
        try:
            SQLiteStorage.export_to_parquet()
            trackio_dir = SQLiteStorage.get_project_db_path(PROJECT).parent
            for path in trackio_dir.glob(f"{PROJECT}*.parquet"):
                shutil.copy2(path, parquet_dir / path.name)
        except Exception as exc:  # noqa: BLE001 - raw inbox + sqlite remain authoritative
            parquet_warning = f"{type(exc).__name__}: {exc}"
            print(f"parquet export skipped: {parquet_warning}")
        db_path = SQLiteStorage.get_project_db_path(PROJECT)
        if db_path.is_file():
            shutil.copy2(db_path, package / f"{PROJECT}.db")
        # Also dump a compact JSONL of recovered metrics for dataset viewers.
        metrics_jsonl = package / "metrics_flat.jsonl"
        with metrics_jsonl.open("wb") as handle:
            for path in sorted(history_root.rglob("*.jsonl")):
                if not path.is_file():
                    continue
                payload = path.read_bytes()
                handle.write(payload)
                if not payload.endswith(b"\n"):
                    handle.write(b"\n")
        manifest = {
            "schema_version": 1,
            "project": PROJECT,
            "exported_at": datetime.now(UTC).isoformat(),
            "note": (
                "Recovered Trackio history for LFM2.5 Wolof. "
                "Live Spaces prefer Buckets; this dataset is the durable archive."
            ),
            "parquet_warning": parquet_warning,
            "files": sorted(
                str(path.relative_to(package)).replace("\\", "/")
                for path in package.rglob("*")
                if path.is_file()
            ),
        }
        (package / "README.md").write_text(
            "---\n"
            "license: other\n"
            "tags:\n"
            "- trackio\n"
            "- experiment-tracking\n"
            "---\n\n"
            "# LFM2.5 Wolof Trackio history\n\n"
            "Private archive of Trackio metrics recovered from legacy Buckets "
            "for the dedicated `lfm25-wolof-trackio` Space.\n\n"
            "Contents:\n"
            "- `bucket_inbox/`: immutable JSONL fragments from source Buckets\n"
            "- `metrics_flat.jsonl`: concatenated recovered records\n"
            "- `lfm25-wolof.db`: local Trackio SQLite snapshot when available\n"
            "- `parquet/`: optional Parquet export when schema-compatible\n",
            encoding="utf-8",
        )
        (package / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        upload_folder(
            folder_path=str(package),
            repo_id=dataset_id,
            repo_type="dataset",
            token=token,
            commit_message="Import recovered Trackio history from legacy Buckets",
        )
    revision = api.repo_info(repo_id=dataset_id, repo_type="dataset").sha
    return {
        "dataset_id": dataset_id,
        "private": True,
        "revision": revision,
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "file_count": len(manifest["files"]),
        "parquet_warning": parquet_warning,
    }


def space_has_trackio_app(api: HfApi, space_id: str) -> bool:
    files = set(api.list_repo_files(repo_id=space_id, repo_type="space"))
    return "app.py" in files and "requirements.txt" in files


def ensure_trackio_space_app(
    *,
    api: HfApi,
    space_id: str,
    bucket_id: str,
    dataset_id: str,
) -> None:
    """Deploy Trackio app files even when an empty Gradio Space already exists.

    ``trackio.sync`` / ``create_space_if_not_exists`` skip ``deploy_as_space`` when
    the Space repo already exists, which leaves Hub-created Spaces in NO_APP_FILE.
    """
    if space_has_trackio_app(api, space_id):
        return
    print(f"repairing missing Trackio app files on {space_id}")
    deploy_as_space(
        space_id,
        bucket_id=bucket_id,
        private=True,
    )
    try:
        api.add_space_variable(space_id, "TRACKIO_DATASET_ID", dataset_id)
    except Exception as exc:  # noqa: BLE001 - variable attach is best-effort
        print(f"TRACKIO_DATASET_ID space variable warning: {type(exc).__name__}: {exc}")
    if not space_has_trackio_app(api, space_id):
        raise RuntimeError(
            f"Trackio app files still missing on {space_id} after deploy_as_space"
        )


def ensure_space_running(api: HfApi, space_id: str, timeout: float = 480) -> dict:
    """Poll Space runtime after Trackio has deployed app files."""
    deadline = time.monotonic() + timeout
    last = {"stage": "UNKNOWN", "dashboard_status": "unknown"}
    while time.monotonic() < deadline:
        try:
            runtime = api.get_space_runtime(space_id)
            stage = str(getattr(runtime.stage, "value", runtime.stage)).upper()
            last = {
                "stage": stage,
                "dashboard_status": "running" if stage == "RUNNING" else stage.lower(),
            }
            if stage == "RUNNING":
                return last
            if stage == "NO_APP_FILE":
                raise RuntimeError(
                    f"Space {space_id} is still NO_APP_FILE after Trackio deploy. "
                    "See build logs on the Hub."
                )
            if stage in {"CONFIG_ERROR", "RUNTIME_ERROR", "BUILD_ERROR"}:
                raise RuntimeError(
                    f"Space {space_id} entered terminal stage {stage}. "
                    "Fix README.md or app files; see build logs on the Hub."
                )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - poll until timeout
            last = {
                "stage": "UNKNOWN",
                "dashboard_status": "runtime_query_unavailable",
                "warning": type(exc).__name__,
            }
        time.sleep(5)
    return last


def verify_bucket_smoke(
    *,
    bucket_id: str,
    project: str,
    run_name: str,
    run_id: str,
    token: str,
    space_id: str | None = None,
    timeout: float = 120,
) -> dict:
    """Confirm smoke persistence via Bucket inbox and/or live Space summary.

    Newer Trackio Bucket Spaces primarily persist through the mounted SQLite DB.
    Inbox JSONL fragments may be empty after ``upload_project_to_bucket``, so a
    successful live ``/get_run_summary`` is treated as verified persistence.
    """
    info = bucket_info(bucket_id, token=token)
    if info.private is not True:
        raise RuntimeError("Trackio Bucket is not private")
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while True:
        paths = list_inbox_paths(bucket_id, token)
        for remote_path in paths:
            with tempfile.TemporaryDirectory(prefix="trackio-readback-") as temporary:
                destination = Path(temporary) / "fragment.jsonl"
                download_bucket_files(
                    bucket_id,
                    files=[(remote_path, str(destination))],
                    token=token,
                )
                if not destination.is_file():
                    continue
                payload = destination.read_bytes()
            for record in parse_fragment_bytes(payload):
                if (
                    record.get("kind") == "metric"
                    and record.get("project") == project
                    and record.get("run") == run_name
                    and record.get("run_id") == run_id
                    and record.get("metrics", {}).get("migration_smoke") == 1
                ):
                    return {
                        "status": "verified",
                        "method": "bucket_inbox",
                        "bucket_id": bucket_id,
                        "private": True,
                        "fragment_path": remote_path,
                        "fragment_sha256": hashlib.sha256(payload).hexdigest(),
                        "bucket_file_count": info.total_files,
                        "bucket_size": info.size,
                    }
        if space_id:
            try:
                queried = RemoteClient(space_id, hf_token=token).predict(
                    project=project,
                    run=run_name,
                    run_id=run_id,
                    api_name="/get_run_summary",
                )
                if (
                    isinstance(queried, dict)
                    and queried.get("run") == run_name
                    and "migration_smoke" in (queried.get("metrics") or [])
                ):
                    return {
                        "status": "verified",
                        "method": "live_space",
                        "bucket_id": bucket_id,
                        "private": True,
                        "bucket_file_count": info.total_files,
                        "bucket_size": info.size,
                        "run_summary": {
                            "num_logs": queried.get("num_logs"),
                            "metrics": queried.get("metrics"),
                            "last_step": queried.get("last_step"),
                        },
                    }
            except Exception as exc:  # noqa: BLE001 - keep polling until timeout
                last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            detail = f" last_error={last_error}" if last_error else ""
            raise RuntimeError(
                "migration smoke metric was not readable from new Bucket/Space"
                + detail
            )
        time.sleep(2)


def run(args: argparse.Namespace) -> int:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.preferred_namespace)
    space_id = f"{namespace}/{args.space_name}"
    dataset_id = f"{namespace}/{args.dataset_name}"
    bucket_id = f"{space_id}-bucket"
    source_buckets = args.source_bucket or _default_source_buckets(namespace)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    sources = []
    for bucket in source_buckets:
        if bucket not in seen:
            sources.append(bucket)
            seen.add(bucket)

    if args.history_dir is not None:
        history_root = (
            args.history_dir
            if args.history_dir.is_absolute()
            else ROOT / args.history_dir
        )
        if not history_root.exists():
            raise RuntimeError(f"history dir does not exist: {history_root}")
        download_summary = {"reused": str(history_root)}
    else:
        history_root = (
            ROOT
            / "results"
            / "full_pipeline"
            / f"trackio_history_cache_{os.getpid()}_{int(time.time())}"
        )
        if history_root.exists():
            shutil.rmtree(history_root)
        download_summary = download_bucket_history(
            bucket_ids=sources,
            destination=history_root,
            token=token,
        )
    import_summary = import_local_fragments(history_root)
    print(
        f"imported {import_summary['imported']} records across "
        f"{import_summary['run_count']} runs "
        f"(remapped={import_summary.get('remapped_to_project', 0)})"
    )

    ensure_private_bucket(api, bucket_id)
    if args.skip_bucket_seed:
        uploaded_fragments = 0
        print(f"skipping bucket seed for {bucket_id}")
    else:
        uploaded_fragments = copy_fragments_to_bucket(
            history_root=history_root,
            bucket_id=bucket_id,
            token=token,
        )
        print(f"seeded {uploaded_fragments} fragments into {bucket_id}")

    # Always upload the unified SQLite snapshot so the Space can retrieve history
    # even when inbox fragments are later compacted away by Trackio sync.
    upload_project_to_bucket(PROJECT, bucket_id)
    print(f"uploaded project DB for {PROJECT} to {bucket_id}")

    dataset_summary = export_history_dataset(
        api=api,
        dataset_id=dataset_id,
        history_root=history_root,
        token=token,
    )
    print(f"history dataset: {dataset_summary['url']}")

    # Empty Hub Gradio Spaces never get Trackio app.py from sync alone.
    ensure_trackio_space_app(
        api=api,
        space_id=space_id,
        bucket_id=bucket_id,
        dataset_id=dataset_id,
    )
    print(f"syncing Trackio project to dedicated Space {space_id}")
    sync_warning = None
    try:
        trackio.sync(
            project=PROJECT,
            space_id=space_id,
            bucket_id=bucket_id,
            private=True,
            force=True,
        )
    except Exception as exc:  # noqa: BLE001 - bucket upload may succeed before visibility
        sync_warning = f"{type(exc).__name__}: {exc}"
        print(f"trackio.sync warning (continuing with Bucket verification): {sync_warning}")
    try:
        api.add_space_variable(space_id, "TRACKIO_DATASET_ID", dataset_id)
    except Exception as exc:  # noqa: BLE001 - variable attach is best-effort
        print(f"TRACKIO_DATASET_ID space variable warning: {type(exc).__name__}: {exc}")
    try:
        api.restart_space(space_id)
    except Exception as exc:  # noqa: BLE001 - restart may race with first boot
        print(f"restart_space warning: {type(exc).__name__}: {exc}")
    dashboard = ensure_space_running(api, space_id)
    info = api.repo_info(repo_id=space_id, repo_type="space")
    if info.private is not True:
        raise RuntimeError(f"Trackio Space is not private: {space_id}")

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    smoke_name = deterministic_run_name(
        size="common",
        stage="trackio-migration-smoke",
        tokenizer_family="none",
        method="dedicated-space",
        seed=42,
        run_id=timestamp,
    )
    smoke = trackio.init(
        project=PROJECT,
        name=smoke_name,
        group="common/trackio-migration",
        config={
            "kind": "dedicated-space-migration-smoke",
            "dataset_id": dataset_id,
            "source_buckets": sources,
        },
        space_id=space_id,
        bucket_id=bucket_id,
        private=True,
    )
    smoke_id = str(getattr(smoke, "id", smoke_name))
    trackio.log(
        {
            "migration_smoke": 1,
            "gate_status": 1,
            "recovered_runs": import_summary["run_count"],
        },
        step=0,
    )
    trackio.finish()
    persistence = verify_bucket_smoke(
        bucket_id=bucket_id,
        project=PROJECT,
        run_name=smoke_name,
        run_id=smoke_id,
        token=token,
        space_id=space_id,
    )

    live_query = {"status": "skipped", "reason": dashboard.get("dashboard_status")}
    if dashboard.get("dashboard_status") == "running":
        try:
            queried = RemoteClient(space_id, hf_token=token).predict(
                project=PROJECT,
                run=smoke_name,
                run_id=smoke_id,
                api_name="/get_run_summary",
            )
            if not isinstance(queried, dict) or queried.get("run") != smoke_name:
                raise RuntimeError("live dashboard query returned the wrong smoke run")
            live_query = {"status": "verified"}
        except Exception as exc:  # noqa: BLE001 - live API may lag first boot
            live_query = {"status": "unavailable", "warning": type(exc).__name__}

    local_db = SQLiteStorage.get_project_db_path(PROJECT)
    local_summary = get_run_summary(PROJECT, smoke_name, run_id=smoke_id)
    if not local_db.is_file() or local_summary.get("run") != smoke_name:
        raise RuntimeError("local Trackio smoke verification failed")

    env_payload = {
        "TRACKIO_SPACE_ID": space_id,
        "TRACKIO_BUCKET_ID": bucket_id,
        "TRACKIO_DATASET_ID": dataset_id,
    }
    report = {
        "schema_version": 1,
        "project": PROJECT,
        "space_id": space_id,
        "dashboard_url": f"https://huggingface.co/spaces/{space_id}",
        "bucket_id": bucket_id,
        "dataset": dataset_summary,
        "source_download": download_summary,
        "import": import_summary,
        "seeded_fragments": uploaded_fragments,
        "dashboard": dashboard,
        "live_query": live_query,
        "bucket_smoke": persistence,
        "sync_warning": sync_warning,
        "smoke_run": smoke_name,
        "smoke_run_id": smoke_id,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    space_payload = {
        "schema_version": 3,
        "project": PROJECT,
        "space_id": space_id,
        "dashboard_url": report["dashboard_url"],
        "private": True,
        "remote_persistence_gate": "pass",
        "live_dashboard_gate": live_query["status"],
        "dashboard": dashboard,
        "live_query": live_query,
        "bucket": persistence,
        "dataset": dataset_summary,
        "recovered_runs": import_summary["run_count"],
        "sync_status": "dedicated_space_migrated",
        "verified_at": report["verified_at"],
    }

    out = args.out if args.out.is_absolute() else ROOT / args.out
    env_out = args.env_out if args.env_out.is_absolute() else ROOT / args.env_out
    report_out = (
        args.report_out if args.report_out.is_absolute() else ROOT / args.report_out
    )
    atomic_json_write(out, space_payload)
    atomic_json_write(env_out, env_payload)
    atomic_json_write(report_out, report)
    print(json.dumps({"space_id": space_id, "dataset_id": dataset_id, **env_payload}, indent=2))
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
