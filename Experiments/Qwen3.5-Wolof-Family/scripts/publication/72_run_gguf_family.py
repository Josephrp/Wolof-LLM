#!/usr/bin/env python3
"""Run GGUF convert → publish for the full Qwen3.5 Wolof family on the H200 host."""

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
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "publication" / "gguf_family.yaml",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "gguf_state.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", default=None, help="Filter size, stage, or size:stage")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--cleanup-after-publish", action="store_true")
    return parser


def model_key(model: dict) -> str:
    return f"{model['size']}:{model['stage']}"


def selected(models: list[dict], only: str | None) -> list[dict]:
    if not only:
        return models
    needle = only.lower()
    out = [
        m
        for m in models
        if needle in {m["size"].lower(), m["stage"].lower(), model_key(m).lower()}
    ]
    if not out:
        raise SystemExit(f"--only matched nothing: {only}")
    return out


def resolve_artifact(model: dict, cache_root: Path) -> Path:
    local = ROOT / model["artifact"]
    if local.is_dir() and (local / "config.json").is_file():
        return local
    hub = model.get("hub_source")
    if not hub:
        raise FileNotFoundError(f"missing local artifact and hub_source for {model_key(model)}")
    dest = cache_root / hub.replace("/", "__")
    print(f"[download] {hub} -> {dest}", flush=True)
    snapshot_download(
        hub,
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    return dest


def run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "publication" / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = build_parser().parse_args()
    cfg = yaml.safe_load((args.config if args.config.is_absolute() else ROOT / args.config).read_text())
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if args.resume and state_path.is_file()
        else {"schema_version": 1, "status": "running", "models": {}, "updated_at": utc_now()}
    )
    cache_root = ROOT / cfg.get("cache_root", "results/hub_cache")
    output_root = ROOT / cfg.get("output_root", "results/gguf/v1")
    llama_root = Path(cfg.get("llamacpp_root", "/mnt/volume_w2l8nkp/tools/llama.cpp"))
    quants = ",".join(cfg.get("quant_set", ["BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"]))
    cleanup = args.cleanup_after_publish or bool(cfg.get("cleanup_after_publish"))
    private = bool(cfg.get("private", False))

    models = selected(cfg["models"], args.only)
    for model in models:
        key = model_key(model)
        prior = state["models"].get(key, {})
        if args.resume and prior.get("status") == "complete":
            print(f"SKIP complete {key}", flush=True)
            continue
        state["models"][key] = {"status": "running", "started_at": utc_now()}
        atomic_json_write(state_path, state)
        try:
            artifact = resolve_artifact(model, cache_root)
            out_dir = output_root / model["size"] / model["stage"].lower()
            run_py(
                "70_convert_gguf.py",
                "--artifact",
                str(artifact),
                "--output-dir",
                str(out_dir),
                "--file-stem",
                model["file_stem"],
                "--llama-root",
                str(llama_root),
                "--quants",
                quants,
            )
            if not args.skip_publish:
                pub_args = [
                    "71_publish_gguf.py",
                    "--gguf-dir",
                    str(out_dir),
                    "--size",
                    model["size"],
                    "--stage",
                    model["stage"],
                    "--repo-name",
                    model["gguf_repo_name"],
                    "--base-repo",
                    model["base_repo"],
                    "--file-stem",
                    model["file_stem"],
                    "--preferred-namespace",
                    cfg.get("namespace_preferred", "Tonic"),
                ]
                if private:
                    pub_args.append("--private")
                run_py(*pub_args)
                if cleanup:
                    for path in out_dir.glob("*.gguf"):
                        path.unlink(missing_ok=True)
                        print(f"cleaned {path.name}", flush=True)
            state["models"][key] = {"status": "complete", "finished_at": utc_now()}
        except Exception as exc:  # noqa: BLE001
            state["models"][key] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "finished_at": utc_now(),
            }
            atomic_json_write(state_path, state)
            print(f"FAIL {key}: {exc}", flush=True)
            raise
        atomic_json_write(state_path, state)

    state["status"] = "complete"
    state["updated_at"] = utc_now()
    atomic_json_write(state_path, state)
    print("GGUF_FAMILY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
