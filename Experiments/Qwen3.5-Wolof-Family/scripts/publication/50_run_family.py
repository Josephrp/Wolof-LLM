#!/usr/bin/env python3
"""Durable Qwen3.5 Wolof family orchestrator with continuous Hub uploads."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now

STATE = ROOT / "results" / "full_pipeline" / "qwen35_family_state.json"


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

STAGES = [
    "materialize",
    "ortho_filter",
    # Tokenizer before prepare so SFT ids match the production extended vocab.
    "tokenizer",
    "ext_init",
    "anchor_prepare",
    "prefs",
    "stock_bpb",
    "cpt_train",
    "cpt_merge",
    "cpt_bpb",
    "sft_from_cpt",
    "sft_from_instruct",
    "sft_merge_cpt",
    "dpo",
]

# Upload whatever exists after each stage (incremental / roll-in).
STAGE_UPLOADS: dict[str, list[str]] = {
    "materialize": ["state"],
    "ortho_filter": ["datasets", "state"],
    "anchor_prepare": ["datasets", "state"],
    "prefs": ["prefs", "state"],
    "tokenizer": ["tokenizer", "state"],
    "ext_init": ["ext_init", "state"],
    "stock_bpb": ["bpb", "state"],
    "cpt_train": ["cpt_adapter", "state"],
    "cpt_merge": ["cpt_adapter", "cpt_merged", "state"],
    "cpt_bpb": ["bpb", "cpt_merged", "state"],
    "sft_from_cpt": ["sft_cpt_adapter", "state"],
    "sft_from_instruct": ["sft_instruct_adapter", "state"],
    "sft_merge_cpt": ["sft_cpt_adapter", "instruct_merged", "state"],
    "dpo": ["state"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 Wolof family pipeline")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-stage", choices=STAGES)
    parser.add_argument("--only-size", default="08b")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    return parser


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": {}, "uploads": {}, "updated_at": None}


def mark(state: dict, key: str) -> None:
    state.setdefault("completed", {})[key] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json_write(STATE, state)


def mark_upload(state: dict, key: str, which: list[str]) -> None:
    state.setdefault("uploads", {})[key] = {"at": utc_now(), "which": which}
    state["updated_at"] = utc_now()
    atomic_json_write(STATE, state)


def done(state: dict, key: str, resume: bool) -> bool:
    return resume and key in state.get("completed", {})


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def uv(*args: str) -> list[str]:
    return ["uv", "run", "python", *args]


def publish_now(size: str, which: list[str]) -> None:
    if not which:
        return
    run_cmd(
        uv(
            "scripts/publication/06_publish_artifacts.py",
            "--size",
            size,
            "--which",
            *which,
        )
    )


def publish_metrics_now(size: str) -> None:
    """Refresh FAMILY_EVAL + Hub model cards for this size."""
    run_cmd(
        uv(
            "scripts/publication/56_publish_hub_metrics.py",
            "--sizes",
            size,
        )
    )


def stage_key(stage: str, size: str) -> str:
    shared = {
        "materialize",
        "ortho_filter",
        "anchor_prepare",
        "prefs",
        "tokenizer",
    }
    return stage if stage in shared else f"{stage}:{size}"


def ext_init_output(size: str) -> str:
    return "models/qwen35-0.8b-ext-init" if size == "08b" else f"models/qwen35-{size}-ext-init"


def main() -> int:
    args = build_parser().parse_args()
    load_trackio_env()
    os.environ.setdefault("QWEN_PRODUCTION", "1")
    os.environ.setdefault(
        "QWEN_PIPELINE_RUN_ID",
        f"qwen35-smoke-{args.only_size}" if args.smoke else f"qwen35-{args.only_size}",
    )
    state = load_state()
    family = yaml.safe_load((ROOT / "configs" / "family.yaml").read_text(encoding="utf-8"))
    size = args.only_size
    size_meta = next(s for s in family["sizes"] if s["key"] == size)
    smoke = ["--smoke"] if args.smoke else []
    print(
        "trackio:",
        os.environ.get("TRACKIO_SPACE_ID"),
        os.environ.get("TRACKIO_BUCKET_ID"),
        "disabled=",
        os.environ.get("TRACKIO_DISABLED"),
        flush=True,
    )

    stages = [args.only_stage] if args.only_stage else list(STAGES)
    if args.skip_dpo:
        stages = [s for s in stages if s != "dpo"]

    for stage in stages:
        key = stage_key(stage, size)
        if done(state, key, args.resume):
            print(f"skip completed {key}", flush=True)
            # Still re-upload existing artifacts so Hub stays current.
            if not args.skip_publish:
                which = STAGE_UPLOADS.get(stage, ["state"])
                try:
                    publish_now(size, which)
                    mark_upload(state, key, which)
                    print(f"UPLOAD_OK {key} {which}", flush=True)
                except subprocess.CalledProcessError as exc:
                    print(f"UPLOAD_WARN {key}: {exc}", flush=True)
            continue

        if stage == "materialize":
            run_cmd(uv("scripts/data/01_materialize_sources.py", *smoke))
        elif stage == "ortho_filter":
            run_cmd(uv("scripts/data/02_ortho_label_and_filter.py", *smoke))
        elif stage == "anchor_prepare":
            tok = "tokenizers/qwen35-wolof-ext-anchors"
            tok_args = (
                ["--tokenizer-model", tok, "--template-model", tok]
                if (ROOT / tok).is_dir()
                else []
            )
            run_cmd(uv("scripts/data/03_anchor_and_prepare_sft.py", *smoke, *tok_args))
        elif stage == "prefs":
            run_cmd(uv("scripts/data/05_build_prefs.py", *smoke))
        elif stage == "tokenizer":
            tok_cfg = "configs/tokenizer/ext_wolof_anchors.yaml"
            extra = ["--max-rows", "20000"] if args.smoke else []
            run_cmd(uv("scripts/clm/02_build_tokenizer.py", "--config", tok_cfg, *extra))
        elif stage == "ext_init":
            out = ext_init_output(size)
            if size == "08b" and (ROOT / out).is_dir():
                print(f"ext_init already present: {out}", flush=True)
            else:
                run_cmd(
                    uv(
                        "scripts/clm/02b_resize_ext_to_size.py",
                        "--base-id",
                        size_meta["base_id"],
                        "--tokenizer",
                        "tokenizers/qwen35-wolof-ext-anchors",
                        "--output",
                        out,
                    )
                )
        elif stage == "stock_bpb":
            out = ROOT / "results" / "bpb" / f"stock_{size}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(
                uv(
                    "scripts/clm/04_evaluate_bpb.py",
                    "--model",
                    size_meta["base_id"],
                    "--output",
                    str(out),
                    "--max-texts",
                    "50" if args.smoke else "200",
                )
            )
        elif stage == "cpt_train":
            cfg = (
                f"configs/clm/{size}/smoke.yaml"
                if args.smoke and size == "08b"
                else f"configs/clm/{size}/v1_ntp70_ortho_lora.yaml"
            )
            train_extra = ["--resume"] if args.resume else []
            run_cmd(
                uv("scripts/clm/03_train_cpt.py", "--config", cfg, *smoke, *train_extra)
            )
        elif stage == "cpt_merge":
            adapter = ROOT / "results" / "smoke" / f"qwen35_{size}_cpt_smoke-cpt" / "adapter"
            if not adapter.is_dir():
                adapter = ROOT / f"runs/clm/{size}/v1_ntp70_ortho_lora" / "adapter"
            merged = ROOT / f"models/qwen35-{size}-cpt-v1-merged"
            base = (
                str(ROOT / "models" / "qwen35-0.8b-ext-init")
                if size == "08b"
                else str(ROOT / "models" / f"qwen35-{size}-ext-init")
            )
            run_cmd(
                uv(
                    "scripts/clm/05_merge_adapter.py",
                    "--base",
                    base,
                    "--adapter",
                    str(adapter),
                    "--output",
                    str(merged),
                )
            )
        elif stage == "cpt_bpb":
            merged = ROOT / f"models/qwen35-{size}-cpt-v1-merged"
            out = ROOT / "results" / "bpb" / f"cpt_{size}.json"
            run_cmd(
                uv(
                    "scripts/clm/04_evaluate_bpb.py",
                    "--model",
                    str(merged),
                    "--output",
                    str(out),
                    "--max-texts",
                    "50" if args.smoke else "200",
                )
            )
        elif stage == "sft_from_cpt":
            train_extra = ["--resume"] if args.resume else []
            run_cmd(
                uv(
                    "scripts/sft/11_train_sft.py",
                    "--config",
                    f"configs/sft/{size}/from_cpt_v1.yaml",
                    *smoke,
                    *train_extra,
                )
            )
        elif stage == "sft_from_instruct":
            train_extra = ["--resume"] if args.resume else []
            run_cmd(
                uv(
                    "scripts/sft/11_train_sft.py",
                    "--config",
                    f"configs/sft/{size}/from_instruct_v1.yaml",
                    *smoke,
                    *train_extra,
                )
            )
        elif stage == "sft_merge_cpt":
            adapter = ROOT / f"runs/sft/{size}_from_cpt_v1" / "adapter"
            if args.smoke:
                adapter = (
                    ROOT
                    / "results"
                    / "smoke"
                    / f"qwen35_{size}_sft_from_cpt_v1-sft"
                    / "adapter"
                )
            merged = ROOT / f"models/qwen35-{size}-instruct-from-cpt-merged"
            base = ROOT / f"models/qwen35-{size}-cpt-v1-merged"
            run_cmd(
                uv(
                    "scripts/clm/05_merge_adapter.py",
                    "--base",
                    str(base),
                    "--adapter",
                    str(adapter),
                    "--output",
                    str(merged),
                )
            )
        elif stage == "dpo":
            run_cmd(
                uv(
                    "scripts/sft/14_train_dpo.py",
                    "--config",
                    f"configs/sft/{size}/dpo_v1.yaml",
                    *smoke,
                )
            )
        else:
            raise SystemExit(f"unknown stage {stage}")

        mark(state, key)
        print(f"STAGE_OK {key}", flush=True)

        if not args.skip_publish:
            which = STAGE_UPLOADS.get(stage, ["state"])
            try:
                publish_now(size, which)
                mark_upload(state, key, which)
                print(f"UPLOAD_OK {key} {which}", flush=True)
            except subprocess.CalledProcessError as exc:
                # Never block training on a Hub hiccup; surface and continue.
                print(f"UPLOAD_WARN {key}: {exc}", flush=True)
                atomic_json_write(
                    ROOT / "results" / "full_pipeline" / "upload_failures.jsonl",
                    {"stage": key, "error": str(exc), "at": utc_now(), "which": which},
                )
            if stage in {"stock_bpb", "cpt_bpb", "sft_from_cpt", "sft_from_instruct", "sft_merge_cpt"}:
                try:
                    publish_metrics_now(size)
                    print(f"METRICS_OK {key}", flush=True)
                except subprocess.CalledProcessError as exc:
                    print(f"METRICS_WARN {key}: {exc}", flush=True)

    if not args.skip_publish:
        try:
            publish_now(size, ["all"])
            print("UPLOAD_OK final all", flush=True)
        except subprocess.CalledProcessError as exc:
            print(f"UPLOAD_WARN final: {exc}", flush=True)
        try:
            publish_metrics_now(size)
            print("METRICS_OK final", flush=True)
        except subprocess.CalledProcessError as exc:
            print(f"METRICS_WARN final: {exc}", flush=True)

    print("FAMILY_PIPELINE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
