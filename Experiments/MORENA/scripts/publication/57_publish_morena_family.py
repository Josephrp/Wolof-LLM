#!/usr/bin/env python3
"""Export, evaluate, and publish the MORENA Wolof family to private Tonic Hub repos.

Artifacts:
  - Tokenizer: Tonic/MORENA-65k-Tokenizer
  - CPT:       Tonic/MORENA-{0.2B-Nano,0.5B-Mini,1.5B}-Wolof-CPT
  - Instruct:  Tonic/MORENA-{0.5B-Mini,1.5B}-Wolof-Instruct
  - Distill:   Tonic/MORENA-{0.2B-Nano,0.5B-Mini}-Wolof-Distill (optional)

Each model repo gets model.safetensors + tokenizer.json + modeling_morena.py +
metrics.json + README model-index (HF-compliant).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT.parent.parent / "Evaluation"
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now  # noqa: E402
from common.tracking import TrackingRun, load_trackio_env  # noqa: E402

FAMILY = [
    {
        "key": "cpt-0.2b",
        "repo": "Tonic/MORENA-0.2B-Nano-Wolof-CPT",
        "size": "0.2b",
        "stage": "CPT",
        "config": "configs/cpt/0.2b_wolof_cpt.json",
        "run_dir": "runs/cpt/0.2b_wolof",
        "base_hub": "vamboai/morena-0.2b-nano",
        "artifact": "models/publish/cpt_0.2b",
        "metrics_out": "metrics/morena/wolof_cpt_0.2b_nano.json",
        "eval_kind": "dcp",
    },
    {
        "key": "cpt-0.5b",
        "repo": "Tonic/MORENA-0.5B-Mini-Wolof-CPT",
        "size": "0.5b",
        "stage": "CPT",
        "config": "configs/cpt/0.5b_wolof_cpt.json",
        "run_dir": "runs/cpt/0.5b_wolof",
        "base_hub": "vamboai/morena-0.5b-mini",
        "artifact": "models/publish/cpt_0.5b",
        "metrics_out": "metrics/morena/wolof_cpt_0.5b_mini.json",
        "eval_kind": "dcp",
    },
    {
        "key": "cpt-1.5b",
        "repo": "Tonic/MORENA-1.5B-Wolof-CPT",
        "size": "1.5b",
        "stage": "CPT",
        "config": "configs/cpt/1.5b_wolof_cpt.json",
        "run_dir": "runs/cpt/1.5b_wolof",
        "base_hub": "vamboai/morena-1.5b-base",
        "artifact": "models/publish/cpt_1.5b",
        "metrics_out": "metrics/morena/wolof_cpt_1.5b.json",
        "eval_kind": "dcp",
    },
    {
        "key": "sft-1.5b",
        "repo": "Tonic/MORENA-1.5B-Wolof-Instruct",
        "size": "1.5b",
        "stage": "Instruct",
        "config": "configs/sft/1.5b_wolof_instruct.json",
        "run_dir": "runs/sft/1.5b_wolof",
        "base_hub": "vamboai/morena-1.5b-base",
        "artifact": "models/publish/sft_1.5b",
        "metrics_out": "metrics/morena/wolof_sft_1.5b.json",
        "eval_kind": "dcp",
    },
    {
        "key": "sft-0.5b",
        "repo": "Tonic/MORENA-0.5B-Mini-Wolof-Instruct",
        "size": "0.5b",
        "stage": "Instruct",
        "config": "configs/sft/0.5b_wolof_instruct.json",
        "run_dir": "runs/sft/0.5b_wolof",
        "base_hub": "vamboai/morena-0.5b-mini",
        "artifact": "models/publish/sft_0.5b",
        "metrics_out": "metrics/morena/wolof_sft_0.5b_mini.json",
        "eval_kind": "dcp",
    },
    {
        "key": "distill-0.5b",
        "repo": "Tonic/MORENA-0.5B-Mini-Wolof-Distill",
        "size": "0.5b",
        "stage": "Distill",
        "config": "models/distilled/0.5b-mini/config.json",
        "weights": "models/distilled/0.5b-mini/model.safetensors",
        "base_hub": "vamboai/morena-0.5b-mini",
        "artifact": "models/publish/distill_0.5b",
        "metrics_out": "metrics/morena/wolof_distill_0.5b_mini.json",
        "eval_kind": "weights",
    },
    {
        "key": "distill-0.2b",
        "repo": "Tonic/MORENA-0.2B-Nano-Wolof-Distill",
        "size": "0.2b",
        "stage": "Distill",
        "config": "models/distilled/0.2b-nano/config.json",
        "weights": "models/distilled/0.2b-nano/model.safetensors",
        "base_hub": "vamboai/morena-0.2b-nano",
        "artifact": "models/publish/distill_0.2b",
        "metrics_out": "metrics/morena/wolof_distill_0.2b_nano.json",
        "eval_kind": "weights",
    },
]

TOKENIZER_REPO = "Tonic/MORENA-65k-Tokenizer"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd or ROOT), env=env)


def export_one(py: str, entry: dict) -> Path:
    out = ROOT / entry["artifact"]
    cmd = [
        py,
        str(ROOT / "scripts" / "cpt" / "05_export_dcp.py"),
        "--config",
        str(ROOT / entry["config"]),
        "--out",
        str(out),
        "--size",
        entry["size"],
        "--stage",
        entry["stage"],
        "--base-hub",
        entry["base_hub"],
        "--title",
        entry["repo"].split("/")[-1].replace("-", " "),
    ]
    if entry.get("weights"):
        cmd.extend(["--weights", str(ROOT / entry["weights"]), "--run-dir", str(ROOT)])
    else:
        cmd.extend(["--run-dir", str(ROOT / entry["run_dir"])])
    run(cmd)
    return out


def eval_artifact(py: str, entry: dict, artifact: Path, *, smoke: bool) -> Path:
    metrics_rel = entry["metrics_out"]
    metrics_path = EVAL_ROOT / metrics_rel
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "base_model_id": str(artifact).replace("\\", "/"),
        "corpus_csv": "../data/corpus_test_clm.csv",
        "text_column": "text",
        "n_texts": "all",
        "apply_verbalization": True,
        "bpb_max_len": 1024,
        "compute_bleu": True,
        "bleu_n_texts": 20 if smoke else 100,
        "bleu_seed": 0,
        "bleu_prompt_ratio": 0.5,
        "bleu_length_ratio": 1.3,
        "attn_mode": "sdpa",
        "allow_cpu": False,
        "output_json": metrics_rel,
    }
    tmp = EVAL_ROOT / "configs" / "morena" / f"_runtime_{entry['key']}.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    try:
        run([py, str(EVAL_ROOT / "scripts" / "evaluate_morena.py"), "--config", str(tmp)], cwd=EVAL_ROOT)
    finally:
        tmp.unlink(missing_ok=True)
    # copy metrics into artifact for Hub upload
    if metrics_path.is_file():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        data["hub_repo"] = entry["repo"]
        data["protocol"] = "morena-native"
        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        (artifact / "metrics.json").write_text(payload, encoding="utf-8")
        metrics_path.write_text(payload, encoding="utf-8")
    return metrics_path


def publish_tokenizer(py: str, source_tok: Path) -> None:
    dest = ROOT / "models" / "publish" / "tokenizer_65k"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_tok, dest / "tokenizer.json")
    readme = """---
language:
- wo
tags:
- morena
- wolof
- tokenizer
library_name: tokenizers
---

# MORENA 65k Tokenizer (Wolof continuum)

Frozen MORENA 65k `tokenizer.json` used across the GalsenAI MORENA Wolof family.
Source upstream: `vamboai/morena-1.5b-base`.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")
    # publish via upload_folder directly (no model.safetensors)
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    api.create_repo(TOKENIZER_REPO, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(TOKENIZER_REPO, repo_type="model")
    if info.private is not True:
        raise RuntimeError("tokenizer repo is not private")
    commit = api.upload_folder(
        repo_id=TOKENIZER_REPO,
        repo_type="model",
        folder_path=str(dest),
        commit_message="Publish MORENA 65k tokenizer",
        allow_patterns=["tokenizer.json", "README.md"],
    )
    print(json.dumps({"repo": TOKENIZER_REPO, "revision": commit.oid, "status": "ok"}))


def write_manifest(entries: list[dict], metrics_ok: dict[str, Path]) -> Path:
    uploads = []
    for entry in entries:
        metrics = metrics_ok.get(entry["key"])
        if metrics and metrics.is_file():
            uploads.append({"hub_repo": entry["repo"], "metrics": str(metrics)})
    path = ROOT / "configs" / "publication" / "hub_uploads.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tokenizer_repo": TOKENIZER_REPO,
                "uploads": uploads,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def refresh_family_eval() -> None:
    metrics_dir = EVAL_ROOT / "metrics" / "morena"
    rows = []
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob("*.json")):
            if path.name.endswith("_smoke.json"):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "file": path.name,
                    "BPB": data.get("BPB"),
                    "BLEU": data.get("BLEU"),
                    "chrF": data.get("chrF"),
                    "PPL": data.get("perplexity"),
                    "hub": data.get("hub_repo", ""),
                }
            )
    lines = [
        "# MORENA family evaluation results",
        "",
        "Generated by `57_publish_morena_family.py`. Protocol: `morena-native`. BPB is gold.",
        "",
        "| File | Hub | BPB ↓ | BLEU | chrF | PPL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['file']} | {row['hub']} | {row['BPB']} | {row['BLEU']} | {row['chrF']} | {row['PPL']} |"
        )
    lines.append("")
    (ROOT / "docs" / "FAMILY_EVAL.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--smoke-eval", action="store_true", help="Faster eval subset")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated keys, e.g. cpt-1.5b,sft-1.5b")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    load_trackio_env(ROOT)
    py = args.python
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    entries = [e for e in FAMILY if not only or e["key"] in only]

    tracking = None
    try:
        tracking = TrackingRun(
            size="family",
            stage="hub-publish",
            tokenizer_family="65k",
            method="private-hub",
            seed=42,
            run_id=os.environ.get("MORENA_PIPELINE_RUN_ID", f"hub-{utc_now()[:19].replace(':', '')}"),
            config={"n_models": len(entries), "smoke_eval": args.smoke_eval},
            production=bool(os.environ.get("TRACKIO_SPACE_ID")),
            state_path=ROOT / "results" / "full_pipeline" / "trackio_hub_publish.json",
        ).start()
    except Exception as exc:  # noqa: BLE001
        print(f"[trackio] unavailable: {exc}", flush=True)

    report = {
        "updated_at": utc_now(),
        "status": "running",
        "exports": [],
        "evals": [],
        "uploads": [],
        "failures": [],
    }
    metrics_ok: dict[str, Path] = {}

    try:
        if not args.skip_tokenizer:
            tok_src = ROOT / "models" / "hub_cache" / "vamboai__morena-1.5b-base" / "tokenizer.json"
            if not tok_src.is_file():
                from huggingface_hub import hf_hub_download

                tok_src = Path(hf_hub_download("vamboai/morena-1.5b-base", "tokenizer.json"))
            if not args.skip_upload:
                publish_tokenizer(py, tok_src)
            else:
                print(f"[skip-upload] tokenizer source={tok_src}")

        for i, entry in enumerate(entries):
            print(f"[export] {entry['key']} -> {entry['repo']}", flush=True)
            try:
                artifact = export_one(py, entry)
                report["exports"].append({"key": entry["key"], "artifact": str(artifact), "status": "ok"})
            except Exception as exc:  # noqa: BLE001
                report["failures"].append({"key": entry["key"], "phase": "export", "error": str(exc)})
                print(f"ERROR export {entry['key']}: {exc}", file=sys.stderr)
                continue

            if not args.skip_eval:
                print(f"[eval] {entry['key']}", flush=True)
                try:
                    metrics = eval_artifact(py, entry, artifact, smoke=args.smoke_eval)
                    metrics_ok[entry["key"]] = metrics
                    report["evals"].append({"key": entry["key"], "metrics": str(metrics), "status": "ok"})
                except Exception as exc:  # noqa: BLE001
                    report["failures"].append({"key": entry["key"], "phase": "eval", "error": str(exc)})
                    print(f"ERROR eval {entry['key']}: {exc}", file=sys.stderr)

            if not args.skip_upload:
                print(f"[upload] {entry['repo']}", flush=True)
                try:
                    run(
                        [
                            py,
                            str(ROOT / "scripts" / "publication" / "42_publish_private.py"),
                            "--artifact",
                            str(artifact),
                            "--repo-id",
                            entry["repo"],
                            "--size",
                            entry["size"],
                            "--stage",
                            entry["stage"].lower(),
                            "--report",
                            str(
                                ROOT
                                / "results"
                                / "full_pipeline"
                                / f"hub_publish_{entry['key']}.json"
                            ),
                        ]
                    )
                    # attach HF-compliant metrics/README model-index if metrics exist
                    if entry["key"] in metrics_ok:
                        manifest = {
                            "uploads": [
                                {
                                    "hub_repo": entry["repo"],
                                    "metrics": str(metrics_ok[entry["key"]]),
                                }
                            ]
                        }
                        man_path = ROOT / "results" / "full_pipeline" / f"hub_uploads_{entry['key']}.yaml"
                        man_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
                        run(
                            [
                                py,
                                str(ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"),
                                "--manifest",
                                str(man_path),
                            ]
                        )
                    report["uploads"].append({"key": entry["key"], "repo": entry["repo"], "status": "ok"})
                    if tracking is not None:
                        tracking.log({f"uploaded_{entry['key'].replace('-', '_')}": 1}, step=i)
                except Exception as exc:  # noqa: BLE001
                    report["failures"].append({"key": entry["key"], "phase": "upload", "error": str(exc)})
                    print(f"ERROR upload {entry['key']}: {exc}", file=sys.stderr)

        manifest = write_manifest(entries, metrics_ok)
        refresh_family_eval()
        report["manifest"] = str(manifest)
        report["tokenizer_repo"] = TOKENIZER_REPO
        report["status"] = "complete" if not report["failures"] else "partial"
        atomic_json_write(ROOT / "results" / "full_pipeline" / "hub_family_publish.json", report)
        if tracking is not None:
            tracking.log(
                {
                    "n_exports": len(report["exports"]),
                    "n_uploads": len(report["uploads"]),
                    "n_failures": len(report["failures"]),
                },
                step=len(entries),
            )
            tracking.finish(success=not report["failures"])
        print("MORENA_HUB_PUBLISH_OK" if not report["failures"] else "MORENA_HUB_PUBLISH_PARTIAL")
        print(json.dumps(report, indent=2)[:4000])
        return 0 if not report["failures"] else 1
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        atomic_json_write(ROOT / "results" / "full_pipeline" / "hub_family_publish.json", report)
        if tracking is not None:
            tracking.finish(success=False, error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
