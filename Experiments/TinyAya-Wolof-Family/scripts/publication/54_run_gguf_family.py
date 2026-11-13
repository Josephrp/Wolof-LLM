#!/usr/bin/env python3
"""Durable Tiny Aya GGUF family runner: convert → imatrix → quant → verify → publish."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Tiny Aya GGUF conversion family")
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
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "gguf_report.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Filter like CPT, InstructEarth, DPO-std, or file_stem substring",
    )
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--skip-imatrix", action="store_true")
    parser.add_argument("--cleanup-after-publish", action="store_true")
    return parser


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "publication" / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_state(path: Path, resume: bool) -> dict:
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "pending",
        "updated_at": utc_now(),
        "models": {},
    }


def model_key(model: dict) -> str:
    return f"{model['stage']}:{model.get('ortho', 'ns')}:{model['file_stem']}"


def matches_only(model: dict, only: str | None) -> bool:
    if not only:
        return True
    needle = only.lower()
    hay = " ".join(
        [
            model.get("stage", ""),
            model.get("ortho", ""),
            f"{model.get('stage', '')}-{model.get('ortho', '')}",
            model.get("file_stem", ""),
            model.get("gguf_repo_name", ""),
            model.get("base_repo", ""),
        ]
    ).lower()
    return needle in hay


def resolve_calib_csv(cfg: dict, model: dict) -> Path:
    calib = cfg.get("calib") or {}
    eval_root = Path(calib.get("eval_root") or "/mnt/volume_w2l8nkp/Wolof-LLM-main/Evaluation")
    ortho = model.get("ortho", "ns")
    rel = calib.get("corpus_std" if ortho == "std" else "corpus_ns")
    if not rel:
        raise ValueError(f"missing calib corpus for ortho={ortho}")
    path = Path(rel)
    if not path.is_absolute():
        path = eval_root / path
    return path


def stage_slug(stage: str) -> str:
    return stage.lower().replace("_", "-")


def cleanup_local_ggufs(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(output_dir.glob("*.gguf")):
        # Keep imatrix for local requant unless publishing already uploaded it.
        if "imatrix" in path.name.lower():
            continue
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def process_model(
    model: dict,
    *,
    cfg: dict,
    state: dict,
    skip_publish: bool,
    skip_imatrix: bool,
    cleanup_after_publish: bool,
) -> dict:
    key = model_key(model)
    record = state["models"].setdefault(key, {"status": "pending"})
    output_dir = ROOT / cfg["output_root"] / model["size"] / stage_slug(model["stage"]) / model.get(
        "ortho", "ns"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if record.get("status") == "complete" and (output_dir / "gguf_meta.json").is_file():
        if cleanup_after_publish and not skip_publish:
            removed = cleanup_local_ggufs(output_dir)
            if removed:
                record["cleaned_ggufs"] = removed
        print(f"skip complete {key}", flush=True)
        return record

    record["status"] = "running"
    record["started_at"] = utc_now()
    try:
        artifact = ROOT / model["artifact"]
        if not artifact.is_dir():
            raise FileNotFoundError(f"artifact missing: {artifact}")
        quants = ",".join(cfg["quant_set"])
        imatrix_quants = ",".join(cfg.get("imatrix_quants") or [])
        calib = cfg.get("calib") or {}
        convert_args = [
            "51_convert_gguf.py",
            "--artifact",
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--file-stem",
            model["file_stem"],
            "--llama-root",
            str(cfg["llamacpp_root"]),
            "--quants",
            quants,
            "--imatrix-quants",
            imatrix_quants,
            "--calib-csv",
            str(resolve_calib_csv(cfg, model)),
            "--calib-text-column",
            str(calib.get("text_column", "text")),
            "--calib-n-texts",
            str(int(calib.get("n_texts", 512))),
            "--calib-chunks",
            str(int(calib.get("chunks", 128))),
            "--calib-ctx",
            str(int(calib.get("ctx_size", 1024))),
            "--n-gpu-layers",
            str(int(calib.get("n_gpu_layers", 99))),
            "--eval-root",
            str(calib.get("eval_root") or "/mnt/volume_w2l8nkp/Wolof-LLM-main/Evaluation"),
        ]
        if calib.get("apply_verbalization", True):
            convert_args.append("--apply-verbalization")
        else:
            convert_args.append("--no-apply-verbalization")
        if skip_imatrix:
            convert_args.append("--skip-imatrix")
        run_py(*convert_args)

        verify_quant = cfg.get("verify_quant", "Q4_K_M")
        gguf_path = output_dir / f"{model['file_stem']}-{verify_quant}.gguf"
        run_py(
            "52_verify_gguf.py",
            "--gguf",
            str(gguf_path),
            "--llama-root",
            str(cfg["llamacpp_root"]),
            "--prompt",
            cfg.get("smoke_prompt", "Salaam aleekum"),
            "--n-predict",
            str(cfg.get("smoke_n_predict", 48)),
            "--out",
            str(output_dir / "verify.json"),
        )
        if not skip_publish:
            publish_args = [
                "53_publish_gguf.py",
                "--gguf-dir",
                str(output_dir),
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
            if cfg.get("publication_manifest"):
                publish_args.extend(["--manifest", str(ROOT / cfg["publication_manifest"])])
            run_py(*publish_args)
            if cleanup_after_publish:
                removed = cleanup_local_ggufs(output_dir)
                record["cleaned_ggufs"] = removed
                print(f"cleaned after publish {key}: {removed}", flush=True)
        record["status"] = "complete"
        record["completed_at"] = utc_now()
        record["output_dir"] = str(output_dir.relative_to(ROOT))
        record["verify_quant"] = verify_quant
        record["hub_repo"] = f"{cfg.get('namespace_preferred', 'Tonic')}/{model['gguf_repo_name']}"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = str(exc)
        record["failed_at"] = utc_now()
        raise
    return record


def main() -> int:
    args = build_parser().parse_args()
    load_trackio_env()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    state = load_state(state_path, args.resume)
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_json_write(state_path, state)

    cleanup = bool(args.cleanup_after_publish or cfg.get("cleanup_after_publish"))
    models = [m for m in cfg.get("models", []) if matches_only(m, args.only)]
    if not models:
        raise SystemExit(f"--only matched nothing: {args.only}")

    try:
        for model in models:
            key = model_key(model)
            print(f"==== {key} ====", flush=True)
            process_model(
                model,
                cfg=cfg,
                state=state,
                skip_publish=args.skip_publish,
                skip_imatrix=args.skip_imatrix,
                cleanup_after_publish=cleanup,
            )
            state["updated_at"] = utc_now()
            atomic_json_write(state_path, state)

        state["status"] = "complete"
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        report = {
            "updated_at": utc_now(),
            "status": "complete",
            "n_models": len(models),
            "models": {
                k: {
                    "status": v.get("status"),
                    "output_dir": v.get("output_dir"),
                    "hub_repo": v.get("hub_repo"),
                    "error": v.get("error"),
                }
                for k, v in state.get("models", {}).items()
            },
        }
        report_path = args.report if args.report.is_absolute() else ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2)[:4000])
        print("TINYAYA_GGUF_FAMILY_OK", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        state["status"] = "failed"
        state["error"] = str(exc)
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
