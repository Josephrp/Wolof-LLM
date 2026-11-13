#!/usr/bin/env python3
"""Durable gated family evaluation (HF CLM + SFT + GGUF) with Hub metrics upload."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from huggingface_hub import HfApi, get_token


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gated LFM family evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "publication" / "eval_family.yaml",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "eval_state.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "eval_report.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only-stage",
        choices=("clm-stock", "clm-cpt", "clm-instruct", "sft-instruct", "gguf-q4", "attach-upload"),
        default=None,
    )
    parser.add_argument("--smoke", action="store_true", help="Run 230m base + 230m instruct GGUF only")
    parser.add_argument("--skip-upload", action="store_true")
    return parser


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


def load_state(path: Path, resume: bool) -> dict:
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "pending",
        "updated_at": utc_now(),
        "stages": {},
        "jobs": {},
    }


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, state)


def job_key(stage: str, size: str, model_stage: str) -> str:
    return f"{stage}:{size}:{model_stage}"


def redact(value):
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"/mnt/(?:volume|data)[^/\s]*/[^\s\"']*", "<REDACTED_REMOTE_PATH>", value)
        value = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "<REDACTED_TOKEN>", value)
        return value
    return value


def run_cmd(cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=cwd, check=True, env=merged)


def patch_yaml_model_id(config_path: Path, model_id: str) -> Path:
    """Write a temp YAML with base_model_id/gguf_path overridden to a local path."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "base_model_id" in payload:
        payload["base_model_id"] = model_id
        payload["subfolder"] = None
        payload["tokenizer_subfolder"] = None
    if "gguf_path" in payload:
        payload["gguf_path"] = model_id
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".yaml", delete=False, dir=str(config_path.parent)
    )
    yaml.safe_dump(payload, handle, sort_keys=False)
    handle.close()
    return Path(handle.name)


def resolve_output_json(eval_root: Path, config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    out = Path(cfg["output_json"])
    if out.is_absolute():
        return out
    stripped = out.as_posix()
    while stripped.startswith("../"):
        stripped = stripped[3:]
    return (eval_root / stripped).resolve()


def eval_python(lfm_root: Path) -> str:
    candidate = lfm_root / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def run_clm_job(
    entry: dict,
    *,
    eval_root: Path,
    lfm_root: Path,
    prefer_local: bool,
    state: dict,
    stage_name: str,
) -> dict:
    key = job_key(stage_name, entry["size"], entry["stage"])
    record = state["jobs"].setdefault(key, {"status": "pending"})
    if record.get("status") == "complete":
        print(f"skip complete {key}", flush=True)
        return record
    config_path = eval_root / entry["config"]
    model_id = entry.get("hub_repo") or (
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ).get("base_model_id")
    temp_cfg = None
    if prefer_local and entry.get("local_artifact"):
        local = lfm_root / entry["local_artifact"]
        if local.is_dir():
            model_id = str(local)
            temp_cfg = patch_yaml_model_id(config_path, model_id)
            config_path = temp_cfg
    record["status"] = "running"
    record["started_at"] = utc_now()
    record["model_id"] = model_id
    try:
        env = {
            "PYTHONPATH": f"{eval_root}:{eval_root / 'scripts'}",
            "TRACKIO_DISABLED": "1",
        }
        run_cmd(
            [
                eval_python(lfm_root),
                str(eval_root / "scripts" / "evaluate_clm.py"),
                "--config",
                str(config_path),
            ],
            cwd=eval_root,
            env=env,
        )
        out = resolve_output_json(eval_root, eval_root / entry["config"])
        record["metrics_path"] = str(out)
        record["metrics"] = json.loads(out.read_text(encoding="utf-8"))
        record["status"] = "complete"
        record["completed_at"] = utc_now()
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = str(exc)
        record["failed_at"] = utc_now()
        raise
    finally:
        if temp_cfg is not None:
            temp_cfg.unlink(missing_ok=True)
    return record


def run_sft_job(entry: dict, *, lfm_root: Path, state: dict) -> dict:
    key = job_key("sft-instruct", entry["size"], entry["stage"])
    record = state["jobs"].setdefault(key, {"status": "pending"})
    if record.get("status") == "complete":
        print(f"skip complete {key}", flush=True)
        return record
    model = lfm_root / entry["local_artifact"]
    data = lfm_root / entry["sft_data"]
    output = lfm_root / "results" / "full_pipeline" / "eval" / entry["size"]
    output.mkdir(parents=True, exist_ok=True)
    record["status"] = "running"
    record["started_at"] = utc_now()
    try:
        run_cmd(
            [
                sys.executable,
                str(lfm_root / "scripts" / "sft" / "12_evaluate_sft.py"),
                "--model",
                str(model),
                "--data",
                str(data),
                "--output",
                str(output),
            ],
            cwd=lfm_root,
            env={"TRACKIO_DISABLED": "1"},
        )
        eval_json = output / "evaluation.json"
        record["metrics_path"] = str(eval_json)
        record["metrics"] = json.loads(eval_json.read_text(encoding="utf-8"))
        record["status"] = "complete"
        record["completed_at"] = utc_now()
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = str(exc)
        record["failed_at"] = utc_now()
        raise
    return record


def run_gguf_job(
    entry: dict,
    *,
    eval_root: Path,
    lfm_root: Path,
    prefer_local: bool,
    state: dict,
) -> dict:
    key = job_key("gguf-q4", entry["size"], entry["stage"])
    record = state["jobs"].setdefault(key, {"status": "pending"})
    if record.get("status") == "complete":
        print(f"skip complete {key}", flush=True)
        return record
    config_path = eval_root / entry["config"]
    gguf_path = entry.get("local_gguf")
    temp_cfg = None
    if prefer_local and gguf_path:
        local = lfm_root / gguf_path
        if local.is_file():
            temp_cfg = patch_yaml_model_id(config_path, str(local))
            config_path = temp_cfg
    record["status"] = "running"
    record["started_at"] = utc_now()
    try:
        env = {
            "PYTHONPATH": f"{eval_root}:{eval_root / 'scripts'}",
            "TRACKIO_DISABLED": "1",
        }
        run_cmd(
            [
                eval_python(lfm_root),
                str(eval_root / "scripts" / "evaluate_gguf.py"),
                "--config",
                str(config_path),
            ],
            cwd=eval_root,
            env=env,
        )
        out = resolve_output_json(eval_root, eval_root / entry["config"])
        record["metrics_path"] = str(out)
        record["metrics"] = json.loads(out.read_text(encoding="utf-8"))
        record["status"] = "complete"
        record["completed_at"] = utc_now()
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = str(exc)
        record["failed_at"] = utc_now()
        raise
    finally:
        if temp_cfg is not None:
            temp_cfg.unlink(missing_ok=True)
    return record


def compose_metrics(
    *,
    clm: dict | None,
    sft: dict | None,
    gguf: dict | None,
    hub_repo: str,
    size: str,
    stage: str,
) -> dict:
    payload = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "hub_repo": hub_repo,
        "size": size,
        "stage": stage,
    }
    if clm:
        payload["clm"] = {
            "BPB": clm.get("BPB"),
            "perplexity": clm.get("perplexity"),
            "BLEU": clm.get("BLEU"),
            "chrF": clm.get("chrF"),
            "n_texts_scored": clm.get("n_texts_scored"),
            "n_bleu_pairs": clm.get("n_bleu_pairs"),
            "vocab_size": clm.get("vocab_size"),
            "run": clm.get("run"),
            "device": clm.get("device"),
        }
    if sft:
        payload["sft"] = {
            "per_task": sft.get("per_task"),
            "qualitative_count": sft.get("qualitative_count"),
            "translation_chrf": sft.get("translation_chrf"),
            "clm_continuity": sft.get("clm_continuity"),
        }
    if gguf:
        payload["gguf"] = gguf
    return redact(payload)


def upload_metrics(repo_id: str, metrics: dict, *, local_copy: Path) -> str:
    local_copy.parent.mkdir(parents=True, exist_ok=True)
    local_copy.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("HF_TOKEN required for metrics upload")
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing to upload metrics to non-private repo: {repo_id}")
    commit = api.upload_file(
        path_or_fileobj=str(local_copy),
        path_in_repo="metrics.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add family evaluation metrics.json",
    )
    return commit.oid


def fill_family_eval(eval_root: Path, lfm_root: Path, state: dict) -> None:
    rows = []
    for key, rec in sorted(state.get("jobs", {}).items()):
        if not key.startswith("clm-") or rec.get("status") != "complete":
            continue
        m = rec.get("metrics") or {}
        rows.append(
            {
                "key": key,
                "BPB": m.get("BPB"),
                "BLEU": m.get("BLEU"),
                "chrF": m.get("chrF"),
                "PPL": m.get("perplexity"),
            }
        )
    gguf_rows = []
    for key, rec in sorted(state.get("jobs", {}).items()):
        if not key.startswith("gguf-") or rec.get("status") != "complete":
            continue
        m = rec.get("metrics") or {}
        gguf_rows.append(
            {
                "key": key,
                "PPL": m.get("perplexity"),
                "BLEU": m.get("BLEU"),
                "chrF": m.get("chrF"),
            }
        )
    path = lfm_root / "docs" / "FAMILY_EVAL.md"
    lines = [
        "# Family evaluation results",
        "",
        "Generated by `55_run_eval_family.py`. BPB is the cross-tokenizer gold metric.",
        "",
        "## HF CLM (evaluate_clm.py)",
        "",
        "| Job | BPB ↓ | BLEU | chrF | PPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['key']} | {row['BPB']} | {row['BLEU']} | {row['chrF']} | {row['PPL']} |"
        )
    lines.extend(
        [
            "",
            "## GGUF Q4_K_M (gguf-llama-cpp protocol; not HF-BPB-equivalent)",
            "",
            "| Job | PPL | BLEU | chrF |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in gguf_rows:
        lines.append(f"| {row['key']} | {row['PPL']} | {row['BLEU']} | {row['chrF']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    # also mirror metrics into Evaluation tree summary
    summary = eval_root / "metrics" / "family_eval_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps({"hf_clm": rows, "gguf": gguf_rows, "updated_at": utc_now()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    load_trackio_env()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    eval_root = Path(cfg["evaluation_root"])
    lfm_root = Path(cfg["lfm_root"])
    prefer_local = bool(cfg.get("prefer_local_artifacts", True))
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    state = load_state(state_path, args.resume)
    state["status"] = "running"
    save_state(state_path, state)

    stages = [
        "clm-stock",
        "clm-cpt",
        "clm-instruct",
        "sft-instruct",
        "gguf-q4",
        "attach-upload",
    ]
    if args.only_stage:
        stages = [args.only_stage]

    try:
        for stage in stages:
            print(f"==== stage {stage} ====", flush=True)
            if stage == "clm-stock":
                entries = cfg["clm_stock"]
                if args.smoke:
                    entries = [e for e in entries if e["size"] == "230m"]
                for entry in entries:
                    run_clm_job(
                        entry,
                        eval_root=eval_root,
                        lfm_root=lfm_root,
                        prefer_local=False,
                        state=state,
                        stage_name=stage,
                    )
                    save_state(state_path, state)
            elif stage == "clm-cpt":
                entries = cfg["clm_cpt"]
                if args.smoke:
                    entries = [e for e in entries if e["size"] == "230m"]
                for entry in entries:
                    run_clm_job(
                        entry,
                        eval_root=eval_root,
                        lfm_root=lfm_root,
                        prefer_local=prefer_local,
                        state=state,
                        stage_name=stage,
                    )
                    save_state(state_path, state)
            elif stage == "clm-instruct":
                entries = cfg["clm_instruct"]
                if args.smoke:
                    entries = [e for e in entries if e["size"] == "230m"]
                for entry in entries:
                    run_clm_job(
                        entry,
                        eval_root=eval_root,
                        lfm_root=lfm_root,
                        prefer_local=prefer_local,
                        state=state,
                        stage_name=stage,
                    )
                    save_state(state_path, state)
            elif stage == "sft-instruct":
                entries = cfg["clm_instruct"]
                if args.smoke:
                    entries = [e for e in entries if e["size"] == "230m"]
                for entry in entries:
                    run_sft_job(entry, lfm_root=lfm_root, state=state)
                    save_state(state_path, state)
            elif stage == "gguf-q4":
                entries = cfg["gguf_q4"]
                if args.smoke:
                    entries = [
                        e
                        for e in entries
                        if e["size"] == "230m" and e["stage"] == "Instruct"
                    ]
                for entry in entries:
                    run_gguf_job(
                        entry,
                        eval_root=eval_root,
                        lfm_root=lfm_root,
                        prefer_local=prefer_local,
                        state=state,
                    )
                    save_state(state_path, state)
            elif stage == "attach-upload":
                uploads = []
                # HF CPT + Instruct
                for entry in cfg["clm_cpt"] + cfg["clm_instruct"]:
                    clm_key = job_key(
                        "clm-cpt" if entry["stage"] == "CPT" else "clm-instruct",
                        entry["size"],
                        entry["stage"],
                    )
                    sft_key = job_key("sft-instruct", entry["size"], entry["stage"])
                    clm = (state["jobs"].get(clm_key) or {}).get("metrics")
                    sft = (state["jobs"].get(sft_key) or {}).get("metrics")
                    if not clm:
                        continue
                    metrics = compose_metrics(
                        clm=clm,
                        sft=sft if entry["stage"] == "Instruct" else None,
                        gguf=None,
                        hub_repo=entry["hub_repo"],
                        size=entry["size"],
                        stage=entry["stage"],
                    )
                    local_copy = (
                        lfm_root
                        / entry["local_artifact"]
                        / "metrics.json"
                    )
                    if args.skip_upload:
                        local_copy.parent.mkdir(parents=True, exist_ok=True)
                        local_copy.write_text(
                            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        rev = None
                    else:
                        rev = upload_metrics(entry["hub_repo"], metrics, local_copy=local_copy)
                    uploads.append({"repo": entry["hub_repo"], "revision": rev})
                # GGUF
                for entry in cfg["gguf_q4"]:
                    gkey = job_key("gguf-q4", entry["size"], entry["stage"])
                    gguf = (state["jobs"].get(gkey) or {}).get("metrics")
                    clm_key = job_key(
                        "clm-cpt" if entry["stage"] == "CPT" else "clm-instruct",
                        entry["size"],
                        entry["stage"],
                    )
                    clm = (state["jobs"].get(clm_key) or {}).get("metrics")
                    if not gguf:
                        continue
                    metrics = compose_metrics(
                        clm=clm,
                        sft=None,
                        gguf=gguf,
                        hub_repo=entry["hub_repo"],
                        size=entry["size"],
                        stage=entry["stage"],
                    )
                    local_copy = lfm_root / entry["local_gguf"]
                    local_copy = local_copy.parent / "metrics.json"
                    if args.skip_upload:
                        local_copy.write_text(
                            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        rev = None
                    else:
                        rev = upload_metrics(entry["hub_repo"], metrics, local_copy=local_copy)
                    uploads.append({"repo": entry["hub_repo"], "revision": rev})
                state["stages"]["attach-upload"] = {
                    "status": "pass",
                    "at": utc_now(),
                    "uploads": uploads,
                }
                fill_family_eval(eval_root, lfm_root, state)
            else:
                raise ValueError(f"unknown stage: {stage}")
            # Preserve richer payloads (e.g. attach-upload uploads list).
            stage_rec = state["stages"].get(stage) or {}
            stage_rec["status"] = "pass"
            stage_rec["at"] = utc_now()
            state["stages"][stage] = stage_rec
            save_state(state_path, state)

        state["status"] = "complete"
        save_state(state_path, state)
        report = {
            "updated_at": utc_now(),
            "status": "complete",
            "stages": state.get("stages"),
            "jobs": {
                key: {
                    "status": value.get("status"),
                    "metrics_path": value.get("metrics_path"),
                    "error": value.get("error"),
                    "BPB": (value.get("metrics") or {}).get("BPB"),
                    "perplexity": (value.get("metrics") or {}).get("perplexity"),
                    "BLEU": (value.get("metrics") or {}).get("BLEU"),
                    "chrF": (value.get("metrics") or {}).get("chrF"),
                }
                for key, value in state.get("jobs", {}).items()
            },
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2)[:4000])
        return 0
    except Exception as exc:  # noqa: BLE001
        state["status"] = "failed"
        state["error"] = str(exc)
        save_state(state_path, state)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
