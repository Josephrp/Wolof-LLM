#!/usr/bin/env python3
"""Run ONNX convert → Hub publish for Qwen3.5 Wolof Instruct family (WebGPU)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hub_utils import select_namespace
from common.pipeline import atomic_json_write


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "publication" / "onnx_family.yaml",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "onnx_state.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip-optional", action="store_true", default=True)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--cleanup-after-publish", action="store_true")
    return parser


def model_key(model: dict) -> str:
    return f"{model['size']}:{model['stage']}"


def resolve_artifact(model: dict, cache_root: Path) -> Path:
    local = ROOT / model["artifact"]
    if local.is_dir() and (local / "config.json").is_file():
        return local
    hub = model["hub_source"]
    dest = cache_root / hub.replace("/", "__")
    print(f"[download] {hub} -> {dest}", flush=True)
    snapshot_download(hub, local_dir=str(dest), token=os.environ.get("HF_TOKEN"))
    return dest


def run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "publication" / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def publish_onnx(folder: Path, repo_name: str, base_repo: str, *, private: bool, namespace: str) -> str:
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    repo_id = f"{namespace}/{repo_name}"
    readme = folder / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if "base_model:" not in text:
            readme.write_text(
                text.replace("---\n", f"---\nbase_model: {base_repo}\n", 1),
                encoding="utf-8",
            )
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    if not private:
        try:
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model")
        except Exception as exc:  # noqa: BLE001
            print(f"visibility warn: {exc}", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        repo_type="model",
        commit_message=f"Publish ONNX WebGPU export of {base_repo}",
    )
    return f"{repo_id}@{commit.oid}"


def main() -> int:
    args = build_parser().parse_args()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if args.resume and state_path.is_file()
        else {"schema_version": 1, "status": "running", "models": {}}
    )
    cache_root = ROOT / cfg.get("cache_root", "results/hub_cache")
    output_root = ROOT / cfg.get("output_root", "results/onnx/v1")
    dtypes = ",".join(cfg.get("dtypes", ["fp16", "q4", "q4f16", "quantized"]))
    cleanup = args.cleanup_after_publish or bool(cfg.get("cleanup_after_publish"))
    private = bool(cfg.get("private", False))
    include_optional = args.include_optional or not args.skip_optional

    models = cfg["models"]
    if args.only:
        needle = args.only.lower()
        models = [
            m
            for m in models
            if needle in {m["size"].lower(), m["stage"].lower(), model_key(m).lower()}
        ]
    if not include_optional:
        models = [m for m in models if not m.get("optional")]

    namespace = cfg.get("namespace_preferred", "Tonic")
    token = os.environ.get("HF_TOKEN") or get_token()
    if token:
        namespace = select_namespace(HfApi(token=token).whoami(), namespace)

    for model in models:
        key = model_key(model)
        if args.resume and state["models"].get(key, {}).get("status") == "complete":
            print(f"SKIP complete {key}", flush=True)
            continue
        state["models"][key] = {"status": "running", "started_at": utc_now()}
        atomic_json_write(state_path, state)
        try:
            artifact = resolve_artifact(model, cache_root)
            out_dir = output_root / model["size"] / model["stage"].lower()
            run_py(
                "73_convert_onnx.py",
                "--artifact",
                str(artifact),
                "--output-dir",
                str(out_dir),
                "--dtypes",
                dtypes,
            )
            if not args.skip_publish:
                rev = publish_onnx(
                    out_dir,
                    model["onnx_repo_name"],
                    model["base_repo"],
                    private=private,
                    namespace=namespace,
                )
                state["models"][key]["repo"] = rev
                if cleanup:
                    onnx_blobs = out_dir / "onnx"
                    if onnx_blobs.is_dir():
                        shutil.rmtree(onnx_blobs)
                        print(f"cleaned onnx weights under {out_dir}", flush=True)
            state["models"][key].update({"status": "complete", "finished_at": utc_now()})
        except Exception as exc:  # noqa: BLE001
            state["models"][key] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "finished_at": utc_now(),
            }
            atomic_json_write(state_path, state)
            raise
        atomic_json_write(state_path, state)

    state["status"] = "complete"
    state["updated_at"] = utc_now()
    atomic_json_write(state_path, state)
    print("ONNX_FAMILY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
