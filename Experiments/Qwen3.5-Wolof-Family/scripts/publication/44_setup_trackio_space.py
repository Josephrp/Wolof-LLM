#!/usr/bin/env python3
"""Create/verify private Trackio Space + Bucket for the Qwen3.5 Wolof family."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now
from common.tracking import PROJECT

DEFAULT_SPACE = "qwen35-wolof-trackio"
DEFAULT_BUCKET = "qwen35-wolof-trackio-bucket"
DEFAULT_DATASET = "qwen35-wolof-trackio-history"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup Qwen3.5 Wolof Trackio Gradio Space")
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument("--space-name", default=DEFAULT_SPACE)
    parser.add_argument("--bucket-name", default=DEFAULT_BUCKET)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--restart", action="store_true", help="Request Space restart")
    return parser


def write_env(space_id: str, bucket_id: str, dataset_id: str) -> Path:
    path = ROOT / ".env.trackio"
    path.write_text(
        "\n".join(
            [
                f"export TRACKIO_SPACE_ID={space_id}",
                f"export TRACKIO_BUCKET_ID={bucket_id}",
                f"export TRACKIO_DATASET_ID={dataset_id}",
                "export TRACKIO_DISABLED=0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)
    private = not args.public
    space_id = f"{args.namespace}/{args.space_name}"
    bucket_id = f"{args.namespace}/{args.bucket_name}"
    dataset_id = f"{args.namespace}/{args.dataset_name}"

    # Private history dataset (optional archive; live runs use Bucket).
    api.create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # Bucket for live Trackio persistence.
    try:
        api.create_bucket(bucket_id, private=private)
    except Exception as exc:  # noqa: BLE001
        print(f"bucket create note: {exc}", flush=True)
        try:
            info = api.bucket_info(bucket_id)
            if getattr(info, "private", None) is not True and private:
                api.update_bucket_settings(bucket_id, private=True)
        except Exception as exc2:  # noqa: BLE001
            print(f"bucket inspect note: {exc2}", flush=True)

    # Gradio Trackio Space — created via trackio.init which scaffolds the dashboard.
    import trackio

    run = trackio.init(
        project=PROJECT,
        name="setup-probe",
        group="setup",
        space_id=space_id,
        bucket_id=bucket_id,
        config={"family": "qwen35-wolof", "purpose": "space_bootstrap"},
    )
    trackio.log({"setup_ok": 1.0}, step=0)
    trackio.finish()
    print(f"trackio probe run={getattr(run, 'id', None)}", flush=True)

    # Ensure Space exists / is private, then wake it.
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"space create note: {exc}", flush=True)
    if args.restart:
        try:
            api.restart_space(space_id)
        except Exception as exc:  # noqa: BLE001
            print(f"restart note: {exc}", flush=True)

    # Poll runtime briefly.
    stage = "UNKNOWN"
    for _ in range(12):
        try:
            runtime = api.get_space_runtime(space_id)
            stage = str(getattr(runtime.stage, "value", runtime.stage)).upper()
            print(f"space runtime: {stage}", flush=True)
            if stage in {"RUNNING", "BUILDING", "APP_STARTING"}:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"runtime query: {exc}", flush=True)
        time.sleep(5)

    env_path = write_env(space_id, bucket_id, dataset_id)
    meta = {
        "created_at": utc_now(),
        "project": PROJECT,
        "space_id": space_id,
        "bucket_id": bucket_id,
        "dataset_id": dataset_id,
        "dashboard_url": f"https://huggingface.co/spaces/{space_id}",
        "runtime_stage": stage,
        "env_path": str(env_path),
    }
    atomic_json_write(ROOT / "results" / "full_pipeline" / "trackio_setup.json", meta)
    print(meta, flush=True)
    print("TRACKIO_SETUP_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
