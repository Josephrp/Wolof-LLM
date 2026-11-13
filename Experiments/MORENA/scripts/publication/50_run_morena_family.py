#!/usr/bin/env python3
"""Durable MORENA Wolof family continuum orchestrator.

Stages: env → stock-eval → pack-data → cpt-0.2b → cpt-0.5b → cpt-1.5b →
sft-1.5b → prune-distill → sft-0.5b → gguf → attach-upload
"""

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
from common.pipeline import atomic_json_write, utc_now  # noqa: E402
from common.tracking import TrackingRun, load_trackio_env  # noqa: E402

EVAL_ROOT = ROOT.parent.parent / "Evaluation"


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
    atomic_json_write(path, state)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd or ROOT), env=env)


def stage_done(state: dict, name: str) -> bool:
    return (state.get("stages") or {}).get(name, {}).get("status") == "pass"


def mark(state: dict, name: str) -> None:
    state.setdefault("stages", {})[name] = {"status": "pass", "at": utc_now()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "publication" / "morena_family.yaml",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "morena_state.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "eval_report.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only-stage",
        choices=(
            "env",
            "stock-eval",
            "pack-data",
            "cpt-0.2b",
            "cpt-0.5b",
            "cpt-1.5b",
            "sft-1.5b",
            "prune-distill",
            "sft-0.5b",
            "gguf",
            "attach-upload",
        ),
        default=None,
    )
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    load_trackio_env(ROOT)
    run_id = os.environ.get("MORENA_PIPELINE_RUN_ID") or datetime.now(UTC).strftime(
        "family-%Y%m%d-%H%M%S"
    )
    os.environ["MORENA_PIPELINE_RUN_ID"] = run_id

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = load_state(args.state, args.resume)
    state["status"] = "running"
    state["pipeline_run_id"] = run_id
    state["trackio"] = {
        "project": os.environ.get("TRACKIO_PROJECT", "morena-wolof"),
        "space_id": os.environ.get("TRACKIO_SPACE_ID"),
        "bucket_id": os.environ.get("TRACKIO_BUCKET_ID"),
        "dataset_id": os.environ.get("TRACKIO_DATASET_ID"),
        "dashboard_url": (
            f"https://huggingface.co/spaces/{os.environ['TRACKIO_SPACE_ID']}"
            if os.environ.get("TRACKIO_SPACE_ID")
            else None
        ),
    }
    save_state(args.state, state)

    pipeline_tracking = None
    if os.environ.get("TRACKIO_DISABLED") != "1":
        try:
            pipeline_tracking = TrackingRun(
                size="family",
                stage="orchestrator",
                tokenizer_family="65k",
                method="continuum",
                seed=1234,
                run_id=run_id,
                config={
                    "smoke": args.smoke,
                    "only_stage": args.only_stage,
                    "stages_planned": (
                        [args.only_stage]
                        if args.only_stage
                        else [
                            "env",
                            "stock-eval",
                            "pack-data",
                            "cpt-0.2b",
                            "cpt-0.5b",
                            "cpt-1.5b",
                            "sft-1.5b",
                            "prune-distill",
                            "sft-0.5b",
                            "gguf",
                            "attach-upload",
                        ]
                    ),
                },
                production=bool(os.environ.get("TRACKIO_SPACE_ID")),
                state_path=args.state.parent / "trackio_orchestrator.json",
            ).start()
            print(
                f"[trackio] orchestrator run={pipeline_tracking.name} "
                f"space={pipeline_tracking.space_id}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[trackio] orchestrator tracking unavailable: {exc}", flush=True)
            pipeline_tracking = None

    py = cfg.get("python") or sys.executable
    stages = [
        "env",
        "stock-eval",
        "pack-data",
        "cpt-0.2b",
        "cpt-0.5b",
        "cpt-1.5b",
        "sft-1.5b",
        "prune-distill",
        "sft-0.5b",
        "gguf",
        "attach-upload",
    ]
    if args.only_stage:
        stages = [args.only_stage]

    try:
        for stage in stages:
            if stage_done(state, stage) and args.resume and not args.only_stage:
                print(f"[skip] {stage}")
                continue
            print(f"[stage] {stage}", flush=True)
            if pipeline_tracking is not None:
                pipeline_tracking.log({f"stage_{stage.replace('-', '_')}": 1}, step=stages.index(stage))

            if stage == "env":
                run([py, str(ROOT / "scripts" / "env" / "00_verify_env.py")])

            elif stage == "stock-eval":
                run(
                    [
                        py,
                        str(EVAL_ROOT / "scripts" / "run_morena_family.py"),
                        "--manifest",
                        str(EVAL_ROOT / "configs" / "morena" / "family_stock.yaml"),
                        "--python",
                        py,
                    ],
                    cwd=EVAL_ROOT,
                )
                # GGUF stock: fetch + evaluate when llamacpp available
                for entry in cfg.get("stock_gguf") or []:
                    out_dir = ROOT / entry["gguf_dir"]
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "publication" / "51_convert_morena_gguf.py"),
                            "--hub-gguf-repo",
                            entry["hub_repo"],
                            "--gguf-filename",
                            entry["filename"],
                            "--out",
                            str(out_dir),
                        ]
                    )
                    gguf_path = out_dir / entry["filename"]
                    eval_cfg = EVAL_ROOT / entry["eval_config"]
                    # rewrite gguf_path in a temp sense via env not supported; patch file
                    text = eval_cfg.read_text(encoding="utf-8")
                    text = text.replace("PLACEHOLDER_LOCAL_OR_HUB", str(gguf_path).replace("\\", "/"))
                    tmp = eval_cfg.with_suffix(".runtime.yaml")
                    tmp.write_text(text, encoding="utf-8")
                    try:
                        run(
                            [py, str(EVAL_ROOT / "scripts" / "evaluate_gguf.py"), "--config", str(tmp)],
                            cwd=EVAL_ROOT,
                        )
                    except subprocess.CalledProcessError as exc:
                        print(f"[warn] gguf eval failed for {entry['name']}: {exc}", flush=True)
                    finally:
                        tmp.unlink(missing_ok=True)

            elif stage == "pack-data":
                corpus = Path(cfg["wolof_corpus"])
                if not corpus.is_absolute():
                    corpus = (ROOT / corpus).resolve()
                if not corpus.is_file():
                    # fall back to Evaluation CLM CSV
                    corpus = EVAL_ROOT / "data" / "corpus_test_clm.csv"
                clm_out = ROOT / "data" / "shards" / "wolof_clm"
                run(
                    [
                        py,
                        str(ROOT / "scripts" / "data" / "01_pack_morena_shards.py"),
                        "--input",
                        str(corpus),
                        "--out",
                        str(clm_out),
                        "--max-docs",
                        str(200 if args.smoke else cfg.get("pack_max_docs", 0) or 0),
                    ]
                )
                sft_src = Path(cfg.get("sft_source") or "")
                if sft_src and not sft_src.is_absolute():
                    sft_src = ROOT / sft_src
                if sft_src and sft_src.is_file():
                    jsonl = ROOT / "data" / "sft" / "wolof_sft.jsonl"
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "sft" / "10_prepare_morena_sft.py"),
                            "--input",
                            str(sft_src),
                            "--out",
                            str(jsonl),
                            "--max-rows",
                            str(100 if args.smoke else 0),
                        ]
                    )
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "data" / "01_pack_morena_shards.py"),
                            "--input",
                            str(jsonl),
                            "--out",
                            str(ROOT / "data" / "shards" / "wolof_sft"),
                            "--sft-jsonl",
                            "--max-docs",
                            str(100 if args.smoke else 0),
                        ]
                    )

            elif stage in {"cpt-0.2b", "cpt-0.5b", "cpt-1.5b"}:
                size = stage.split("-", 1)[1]
                entry = cfg["cpt"][size]
                cmd = [
                    py,
                    str(ROOT / "scripts" / "cpt" / "03_train_morena.py"),
                    "--config",
                    str(ROOT / entry["config"]),
                    "--mix",
                    str(ROOT / "configs" / "cpt" / "mix_wolof.json"),
                    "--data-root",
                    str(ROOT / "data" / "shards"),
                    "--out",
                    str(ROOT / entry["out"]),
                    "--hub-init",
                    entry["hub_init"],
                    "--nproc",
                    str(entry.get("nproc", 1)),
                    "--trackio-size",
                    size,
                    "--trackio-stage",
                    "cpt",
                ]
                if args.smoke:
                    cmd.append("--smoke")
                run(cmd)

            elif stage == "sft-1.5b":
                entry = cfg["sft"]["1.5b"]
                sft_shards = ROOT / "data" / "shards" / "wolof_sft"
                if not (sft_shards / "index.json").is_file():
                    print("[warn] wolof_sft shards missing; skipping sft-1.5b")
                else:
                    cmd = [
                        py,
                        str(ROOT / "scripts" / "cpt" / "03_train_morena.py"),
                        "--config",
                        str(ROOT / entry["config"]),
                        "--mix",
                        str(ROOT / "configs" / "sft" / "mix_wolof_sft.json"),
                        "--data-root",
                        str(ROOT / "data" / "shards"),
                        "--out",
                        str(ROOT / entry["out"]),
                        "--init-weights",
                        str(ROOT / entry["init_weights"]),
                        "--loss-mask",
                        "--nproc",
                        "1",
                        "--trackio-size",
                        "1.5b",
                        "--trackio-stage",
                        "sft",
                    ]
                    if args.smoke:
                        cmd.append("--smoke")
                    # init_weights may still be hub path — allow hub-init fallback
                    if not Path(entry["init_weights"]).is_file() and not (ROOT / entry["init_weights"]).is_file():
                        cmd = [c for c in cmd if c != "--init-weights" and c != str(ROOT / entry["init_weights"])]
                        cmd.extend(["--hub-init", entry.get("hub_init", "vamboai/morena-1.5b-base")])
                    run(cmd)

            elif stage == "prune-distill":
                teacher_w = ROOT / cfg["distill"]["teacher_weights"]
                teacher_c = ROOT / cfg["distill"]["teacher_config"]
                if not teacher_w.is_file():
                    # use hub 1.5b base as teacher for scaffold smoke
                    from huggingface_hub import hf_hub_download

                    teacher_w = Path(hf_hub_download("vamboai/morena-1.5b-base", "model.safetensors"))
                    teacher_c = Path(hf_hub_download("vamboai/morena-1.5b-base", "config.json"))
                for target in ("0.5b-mini", "0.2b-nano"):
                    pruned = ROOT / "models" / "pruned" / target
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "distill" / "40_prune_student.py"),
                            "--teacher-weights",
                            str(teacher_w),
                            "--teacher-config",
                            str(teacher_c),
                            "--target",
                            target,
                            "--out",
                            str(pruned),
                        ]
                    )
                    distilled = ROOT / "models" / "distilled" / target
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "distill" / "41_train_distill.py"),
                            "--teacher-config",
                            str(teacher_c),
                            "--teacher-weights",
                            str(teacher_w),
                            "--student-config",
                            str(pruned / "config.json"),
                            "--student-weights",
                            str(pruned / "model.safetensors"),
                            "--data",
                            str(ROOT / "data" / "shards" / "wolof_clm"),
                            "--out",
                            str(distilled),
                            "--trackio-size",
                            target,
                            *(["--smoke"] if args.smoke else ["--steps", str(cfg["distill"].get("steps", 1000))]),
                        ]
                    )

            elif stage == "sft-0.5b":
                entry = cfg["sft"]["0.5b"]
                sft_shards = ROOT / "data" / "shards" / "wolof_sft"
                if not (sft_shards / "index.json").is_file():
                    print("[warn] wolof_sft shards missing; skipping sft-0.5b")
                else:
                    init_w = ROOT / entry["init_weights"]
                    cmd = [
                        py,
                        str(ROOT / "scripts" / "cpt" / "03_train_morena.py"),
                        "--config",
                        str(ROOT / entry["config"]),
                        "--mix",
                        str(ROOT / "configs" / "sft" / "mix_wolof_sft.json"),
                        "--data-root",
                        str(ROOT / "data" / "shards"),
                        "--out",
                        str(ROOT / entry["out"]),
                        "--loss-mask",
                        "--nproc",
                        "1",
                        "--trackio-size",
                        "0.5b",
                        "--trackio-stage",
                        "sft",
                    ]
                    if init_w.is_file():
                        cmd.extend(["--init-weights", str(init_w)])
                    else:
                        cmd.extend(["--hub-init", "vamboai/morena-0.5b-mini"])
                    if args.smoke:
                        cmd.append("--smoke")
                    run(cmd)

            elif stage == "gguf":
                for entry in cfg.get("gguf_targets") or []:
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "publication" / "51_convert_morena_gguf.py"),
                            *(
                                ["--hub-gguf-repo", entry["hub_repo"], "--gguf-filename", entry["filename"]]
                                if entry.get("hub_repo")
                                else ["--model-dir", str(ROOT / entry["model_dir"])]
                            ),
                            "--out",
                            str(ROOT / entry["out"]),
                        ]
                    )

            elif stage == "attach-upload":
                # Fill FAMILY_EVAL from available metrics
                metrics_dir = EVAL_ROOT / "metrics" / "morena"
                rows = []
                if metrics_dir.is_dir():
                    for path in sorted(metrics_dir.glob("*.json")):
                        data = json.loads(path.read_text(encoding="utf-8"))
                        rows.append(
                            {
                                "file": path.name,
                                "BPB": data.get("BPB"),
                                "BLEU": data.get("BLEU"),
                                "chrF": data.get("chrF"),
                                "PPL": data.get("perplexity"),
                            }
                        )
                lines = [
                    "# MORENA family evaluation results",
                    "",
                    "Generated by `50_run_morena_family.py`. Protocol: `morena-native`. BPB is gold.",
                    "",
                    "| File | BPB ↓ | BLEU | chrF | PPL |",
                    "|---|---:|---:|---:|---:|",
                ]
                for row in rows:
                    lines.append(
                        f"| {row['file']} | {row['BPB']} | {row['BLEU']} | {row['chrF']} | {row['PPL']} |"
                    )
                lines.append("")
                (ROOT / "docs" / "FAMILY_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
                if not args.skip_upload and (cfg.get("upload_manifest")):
                    manifest = ROOT / cfg["upload_manifest"]
                    if manifest.is_file():
                        run(
                            [
                                py,
                                str(ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"),
                                "--manifest",
                                str(manifest),
                            ]
                        )
                    else:
                        print(f"[warn] upload manifest missing: {manifest}")
                else:
                    print("[skip] hub upload")

            else:
                raise ValueError(stage)

            mark(state, stage)
            save_state(args.state, state)

        state["status"] = "complete"
        save_state(args.state, state)
        report = {
            "updated_at": utc_now(),
            "status": "complete",
            "stages": state.get("stages"),
            "smoke": args.smoke,
            "pipeline_run_id": run_id,
            "trackio": state.get("trackio"),
        }
        atomic_json_write(args.report, report)
        if pipeline_tracking is not None:
            pipeline_tracking.log({"pipeline_complete": 1}, step=len(stages))
            pipeline_tracking.finish(success=True)
        print("MORENA_FAMILY_OK")
        print(json.dumps(report, indent=2)[:3000])
        return 0
    except Exception as exc:  # noqa: BLE001
        state["status"] = "failed"
        state["error"] = str(exc)
        save_state(args.state, state)
        if pipeline_tracking is not None:
            pipeline_tracking.finish(success=False, error=exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
