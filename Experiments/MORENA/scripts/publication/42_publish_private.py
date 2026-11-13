#!/usr/bin/env python3
"""Create/update a private Tonic MORENA Hub repo from a local artifact package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now  # noqa: E402
from common.tracking import TrackingRun, load_trackio_env  # noqa: E402

PUBLISH_FILES = {
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "modeling_morena.py",
    "README.md",
    "SHA256SUMS.json",
    "metrics.json",
    "export_meta.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="namespace/name")
    parser.add_argument("--size", default="morena")
    parser.add_argument("--stage", default="publish")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "hub_publish_one.json",
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    load_trackio_env(ROOT)
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    if not artifact.is_dir():
        raise SystemExit(f"missing artifact: {artifact}")

    required = {"model.safetensors", "config.json", "tokenizer.json", "README.md"}
    present = {p.name for p in artifact.iterdir() if p.is_file()}
    missing = required - present
    if missing:
        raise SystemExit(f"artifact missing required files: {sorted(missing)}")

    expected = {
        name: {"sha256": sha256_file(artifact / name), "size": (artifact / name).stat().st_size}
        for name in sorted(present & PUBLISH_FILES)
    }

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo: {args.repo_id}")

    tracking = None
    try:
        tracking = TrackingRun(
            size=args.size,
            stage=f"publish-{args.stage}",
            tokenizer_family="65k",
            method="private-hub",
            seed=42,
            run_id=os.environ.get("MORENA_PIPELINE_RUN_ID", args.repo_id.replace("/", "__")),
            config={"repo_id": args.repo_id, "artifact": str(artifact)},
            production=bool(os.environ.get("TRACKIO_SPACE_ID")),
            state_path=ROOT / "results" / "full_pipeline" / "tracking" / f"publish_{args.repo_id.replace('/', '__')}.json",
        ).start()
    except Exception as exc:  # noqa: BLE001
        print(f"[trackio] publish tracking unavailable: {exc}", flush=True)

    try:
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(artifact),
            commit_message=f"Publish MORENA Wolof {args.size} {args.stage}",
            allow_patterns=sorted(PUBLISH_FILES),
            ignore_patterns=["*.log", "**/.env*", "**/ckpt/**"],
        )
        remote = set(api.list_repo_files(repo_id=args.repo_id, repo_type="model", revision=commit.oid))
        missing_remote = sorted(set(expected) - remote)
        if missing_remote:
            raise RuntimeError(f"remote missing files: {missing_remote}")
        info2 = api.repo_info(repo_id=args.repo_id, repo_type="model", revision=commit.oid)
        if info2.private is not True:
            raise RuntimeError("published repository is not private")

        readback = {}
        if not args.skip_verify:
            with tempfile.TemporaryDirectory(prefix="morena-hub-rb-") as temporary:
                for name in ("config.json", "README.md", "SHA256SUMS.json", "tokenizer.json"):
                    if name not in expected:
                        continue
                    local = Path(
                        hf_hub_download(
                            args.repo_id,
                            name,
                            token=token,
                            revision=commit.oid,
                            local_dir=temporary,
                        )
                    )
                    digest = sha256_file(local)
                    if digest != expected[name]["sha256"]:
                        raise RuntimeError(f"readback checksum mismatch: {name}")
                    readback[name] = "ok"

        report = {
            "updated_at": utc_now(),
            "repo_id": args.repo_id,
            "revision": commit.oid,
            "private": True,
            "files": expected,
            "readback": readback,
            "status": "ok",
            "url": f"https://huggingface.co/{args.repo_id}",
        }
        atomic_json_write(
            args.report if args.report.is_absolute() else ROOT / args.report,
            report,
        )
        print(json.dumps(report, indent=2)[:2000])
        if tracking is not None:
            tracking.log({"publish_ok": 1, "n_files": len(expected)}, step=0)
            tracking.finish(success=True)
        return 0
    except Exception as exc:
        if tracking is not None:
            tracking.finish(success=False, error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
