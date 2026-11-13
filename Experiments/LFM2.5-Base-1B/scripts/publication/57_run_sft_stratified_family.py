#!/usr/bin/env python3
"""Durable stratified-SFT family: prepare → train → merge → eval → publish v2 Instruct."""

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

FAMILY = [
    {
        "size": "230m",
        "config": "configs/sft/230m/full_stratified_v1.yaml",
        "cpt_merged": "models/v2_230m_ext_lora-merged",
        "adapter": "runs/sft/230m_full_stratified_v1/adapter",
        "merged": "models/lfm25-230m-wolof-instruct-stratified-merged",
        "eos": "7",
        "tok_family": "65k-ext",
        "sft_data": "data/sft/prepared_stratified_v1",
        "base_model": "Tonic/LFM2.5-230M-Wolof-CPT",
        "hub_repo": "Tonic/LFM2.5-230M-Wolof-Instruct-v2",
        "clm_config": "configs/clm/lfm_230m_instruct.yaml",
        "clm_metrics": "metrics/clm/lfm_230m_instruct_v2.json",
        "eval_dir": "results/full_pipeline/eval_stratified/230m",
    },
    {
        "size": "350m",
        "config": "configs/sft/350m/full_stratified_v1.yaml",
        "cpt_merged": "models/v2_350m_ext4000_lora-merged",
        "adapter": "runs/sft/350m_full_stratified_v1/adapter",
        "merged": "models/lfm25-350m-wolof-instruct-stratified-merged",
        "eos": "7",
        "tok_family": "65k-ext",
        "sft_data": "data/sft/prepared_stratified_v1",
        "base_model": "Tonic/LFM2.5-350M-Wolof-CPT",
        "hub_repo": "Tonic/LFM2.5-350M-Wolof-Instruct-v2",
        "clm_config": "configs/clm/lfm_350m_instruct.yaml",
        "clm_metrics": "metrics/clm/lfm_350m_instruct_v2.json",
        "eval_dir": "results/full_pipeline/eval_stratified/350m",
    },
    {
        "size": "1.2b",
        "config": "configs/sft/1.2b/full_stratified_v1.yaml",
        "cpt_merged": "models/v2_1.2b_ext4000_lora-merged",
        "adapter": "runs/sft/1.2b_full_stratified_v1/adapter",
        "merged": "models/lfm25-1.2b-wolof-instruct-stratified-merged",
        "eos": "7",
        "tok_family": "65k-ext",
        "sft_data": "data/sft/prepared_stratified_v1",
        "base_model": "Tonic/LFM2.5-1.2B-Wolof-CPT",
        "hub_repo": "Tonic/LFM2.5-1.2B-Wolof-Instruct-v2",
        "clm_config": "configs/clm/lfm_1.2b_instruct.yaml",
        "clm_metrics": "metrics/clm/lfm_1.2b_instruct_v2.json",
        "eval_dir": "results/full_pipeline/eval_stratified/1.2b",
    },
    {
        "size": "2.6b",
        "config": "configs/sft/2.6b/full_stratified_v1.yaml",
        "cpt_merged": "models/lfm_2.6b_v2_ext4000_lora-merged",
        "adapter": "runs/sft/2.6b_full_stratified_v1/adapter",
        "merged": "models/lfm25-2.6b-wolof-instruct-stratified-merged",
        "eos": "128900",
        "tok_family": "128k-ext",
        "sft_data": "data/sft/prepared_stratified_v1_128k",
        "base_model": "Tonic/LFM2.5-2.6B-Wolof-CPT",
        "hub_repo": "Tonic/LFM2.5-2.6B-Wolof-Instruct-v2",
        "clm_config": "configs/clm/lfm_2.6b_instruct.yaml",
        "clm_metrics": "metrics/clm/lfm_2.6b_instruct_v2.json",
        "eval_dir": "results/full_pipeline/eval_stratified/2.6b",
    },
    {
        "size": "8b-a1b",
        "config": "configs/sft/8b/full_stratified_v1.yaml",
        "cpt_merged": "models/lfm_8b_a1b_v2_moe_lora-merged",
        "adapter": "runs/sft/8b_a1b_full_stratified_v1/adapter",
        "merged": "models/lfm25-8b-a1b-wolof-instruct-stratified-merged",
        "eos": "128900",
        "tok_family": "128k-ext",
        "sft_data": "data/sft/prepared_stratified_v1_128k",
        "base_model": "Tonic/LFM2.5-8B-A1B-Wolof-CPT",
        "hub_repo": "Tonic/LFM2.5-8B-A1B-Wolof-Instruct-v2",
        "clm_config": "configs/clm/lfm_8b_instruct.yaml",
        "clm_metrics": "metrics/clm/lfm_8b_instruct_v2.json",
        "eval_dir": "results/full_pipeline/eval_stratified/8b-a1b",
    },
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def stage_done(state: dict, name: str) -> bool:
    return (state.get("stages") or {}).get(name, {}).get("status") == "pass"


def mark_pass(state: dict, name: str, **extra) -> None:
    rec = {"status": "pass", "at": utc_now(), **extra}
    state.setdefault("stages", {})[name] = rec


def python_bin() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def prepare_data(state: dict, state_path: Path, *, smoke: bool) -> None:
    if not stage_done(state, "materialize-stratified"):
        run_cmd(
            [
                python_bin(),
                "scripts/data/22_materialize_stratified_sft.py",
                "--output",
                "data/sft/stratified_v1",
            ]
        )
        mark_pass(state, "materialize-stratified")
        atomic_json_write(state_path, state)

    if not stage_done(state, "prepare-65k"):
        cmd = [
            python_bin(),
            "scripts/sft/10_prepare_sft.py",
            "--input",
            "data/sft/stratified_v1",
            "--output",
            "data/sft/prepared_stratified_v1",
            "--tokenizer-model",
            "models/lfm25-wolof-ext4000-init",
        ]
        if smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
        mark_pass(state, "prepare-65k")
        atomic_json_write(state_path, state)

    if not stage_done(state, "prepare-128k"):
        cmd = [
            python_bin(),
            "scripts/sft/10_prepare_sft.py",
            "--input",
            "data/sft/stratified_v1",
            "--output",
            "data/sft/prepared_stratified_v1_128k",
            "--template-model",
            "LiquidAI/LFM2.5-2.6B",
            "--tokenizer-model",
            "models/lfm25-2.6b-wolof-ext4000-init",
        ]
        if smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
        mark_pass(state, "prepare-128k")
        atomic_json_write(state_path, state)


def publish_datasets(state: dict, state_path: Path) -> None:
    if stage_done(state, "publish-datasets"):
        return
    run_cmd(
        [
            python_bin(),
            "scripts/publication/47_publish_training_datasets.py",
            "--namespace",
            "Tonic",
            "--include-stratified-prepared",
        ]
    )
    mark_pass(state, "publish-datasets")
    atomic_json_write(state_path, state)


def train_one(entry: dict, *, smoke: bool, resume: bool) -> None:
    cmd = [python_bin(), "scripts/sft/11_train_sft.py", "--config", entry["config"]]
    if resume:
        cmd.append("--resume")
    if smoke:
        cmd.append("--smoke")
    run_cmd(cmd)


def merge_one(entry: dict) -> None:
    run_cmd(
        [
            python_bin(),
            "scripts/sft/13_merge_sft.py",
            "--base",
            entry["cpt_merged"],
            "--adapter",
            entry["adapter"] if not entry["adapter"].endswith("/adapter") else entry["adapter"],
            "--output",
            entry["merged"],
            "--expected-eos",
            entry["eos"],
        ]
    )


def evaluate_sft(entry: dict) -> None:
    out = ROOT / entry["eval_dir"]
    out.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_bin(),
            "scripts/sft/12_evaluate_sft.py",
            "--model",
            entry["merged"],
            "--data",
            entry["sft_data"],
            "--output",
            entry["eval_dir"],
        ]
    )


def evaluate_clm(entry: dict, eval_root: Path) -> None:
    cfg_path = eval_root / entry["clm_config"]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["base_model_id"] = str((ROOT / entry["merged"]).resolve())
    cfg.pop("subfolder", None)
    cfg["placeholder"] = False
    cfg["output_json"] = entry["clm_metrics"]
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temp = handle.name
    try:
        run_cmd(
            [
                python_bin(),
                str(eval_root / "scripts" / "evaluate_clm.py"),
                "--config",
                temp,
            ],
            cwd=eval_root,
        )
    finally:
        Path(temp).unlink(missing_ok=True)


def publish_one(entry: dict) -> None:
    metrics = ROOT / entry["eval_dir"] / "evaluation.json"
    cfg = yaml.safe_load((ROOT / entry["config"]).read_text(encoding="utf-8"))
    journal = ROOT / cfg["output_dir"] / "journal_sft.json"
    if not journal.is_file():
        candidates = list((ROOT / "results" / "smoke").glob("*stratified*sft/journal_sft.json"))
        if candidates:
            journal = candidates[-1]
    if not journal.is_file():
        raise FileNotFoundError(f"missing SFT journal for {entry['size']}: {journal}")
    if not metrics.is_file():
        raise FileNotFoundError(f"missing SFT evaluation.json for {entry['size']}: {metrics}")
    run_cmd(
        [
            python_bin(),
            "scripts/publication/43_prepare_artifact.py",
            "--artifact",
            entry["merged"],
            "--metrics",
            str(metrics.relative_to(ROOT)) if metrics.is_relative_to(ROOT) else str(metrics),
            "--run-manifest",
            str(journal.relative_to(ROOT)) if journal.is_relative_to(ROOT) else str(journal),
            "--model-name",
            entry["hub_repo"].split("/")[-1],
            "--base-model",
            entry["base_model"],
            "--tokenizer-family",
            entry["tok_family"],
            "--training-stage",
            "Instruct",
        ]
    )
    # Upload via HfApi directly to v2 repo (avoid 42's galsenai default + exist_ok=False issues)
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    repo_id = entry["hub_repo"]
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo {repo_id}")
    artifact = ROOT / entry["merged"]
    commit = api.upload_folder(
        folder_path=str(artifact),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish stratified-SFT Instruct-v2 artifact",
    )
    # also upload CLM metrics if present
    eval_root = Path(os.environ.get("EVALUATION_ROOT", "/mnt/volume_w2l8nkp/Wolof-LLM-main/Evaluation"))
    clm_path = eval_root / entry["clm_metrics"]
    if clm_path.is_file():
        api.upload_file(
            path_or_fileobj=str(clm_path),
            path_in_repo="metrics_clm.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add CLM family metrics for Instruct-v2",
        )
    # Compose HF model-index + Evaluation section onto the model card.
    run_cmd(
        [
            python_bin(),
            "scripts/publication/56_publish_hub_metrics.py",
            "--only-repo",
            repo_id,
        ]
    )
    print(json.dumps({"published": repo_id, "revision": commit.oid}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "sft_stratified_state.json",
    )
    parser.add_argument(
        "--only-stage",
        choices=(
            "publish-used-datasets",
            "prepare",
            "publish-all-datasets",
            "train",
            "merge",
            "eval",
            "publish-models",
        ),
        default=None,
    )
    parser.add_argument("--only-size", action="append", default=[])
    args = parser.parse_args()

    load_trackio_env()
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema_version": 1, "status": "running", "stages": {}, "jobs": {}}
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_json_write(state_path, state)

    entries = FAMILY
    if args.only_size:
        wanted = set(args.only_size)
        entries = [e for e in entries if e["size"] in wanted]

    eval_root = Path(
        os.environ.get(
            "EVALUATION_ROOT",
            "/mnt/volume_w2l8nkp/Wolof-LLM-main/Evaluation",
        )
    )

    try:
        stages = []
        if args.only_stage is None:
            stages = [
                "publish-used-datasets",
                "prepare",
                "publish-all-datasets",
                "train",
                "merge",
                "eval",
                "publish-models",
            ]
        else:
            stages = [args.only_stage]

        for stage in stages:
            if stage == "publish-used-datasets":
                if not stage_done(state, "publish-used-datasets"):
                    run_cmd(
                        [
                            python_bin(),
                            "scripts/publication/47_publish_training_datasets.py",
                            "--namespace",
                            "Tonic",
                        ]
                    )
                    mark_pass(state, "publish-used-datasets")
                    atomic_json_write(state_path, state)
            elif stage == "prepare":
                prepare_data(state, state_path, smoke=args.smoke)
            elif stage == "publish-all-datasets":
                publish_datasets(state, state_path)
            elif stage == "train":
                for entry in entries:
                    key = f"train:{entry['size']}"
                    if (state.get("jobs") or {}).get(key, {}).get("status") == "complete":
                        continue
                    if not (ROOT / entry["cpt_merged"]).exists():
                        raise FileNotFoundError(f"missing CPT merged model: {entry['cpt_merged']}")
                    train_one(entry, smoke=args.smoke, resume=args.resume)
                    state.setdefault("jobs", {})[key] = {
                        "status": "complete",
                        "at": utc_now(),
                    }
                    atomic_json_write(state_path, state)
                mark_pass(state, "train")
                atomic_json_write(state_path, state)
            elif stage == "merge":
                for entry in entries:
                    key = f"merge:{entry['size']}"
                    if (state.get("jobs") or {}).get(key, {}).get("status") == "complete":
                        continue
                    adapter = ROOT / entry["adapter"]
                    if args.smoke and not adapter.exists():
                        # smoke saves under results/smoke/<run>-sft/adapter
                        smoke_dirs = sorted((ROOT / "results" / "smoke").glob("*stratified*sft"))
                        if smoke_dirs:
                            entry = dict(entry)
                            entry["adapter"] = str(
                                (smoke_dirs[-1] / "adapter").relative_to(ROOT)
                            )
                    merge_one(entry)
                    state.setdefault("jobs", {})[key] = {
                        "status": "complete",
                        "at": utc_now(),
                        "merged": entry["merged"],
                    }
                    atomic_json_write(state_path, state)
                mark_pass(state, "merge")
                atomic_json_write(state_path, state)
            elif stage == "eval":
                for entry in entries:
                    key = f"eval:{entry['size']}"
                    if (state.get("jobs") or {}).get(key, {}).get("status") == "complete":
                        continue
                    evaluate_sft(entry)
                    if not args.smoke:
                        evaluate_clm(entry, eval_root)
                    state.setdefault("jobs", {})[key] = {
                        "status": "complete",
                        "at": utc_now(),
                    }
                    atomic_json_write(state_path, state)
                mark_pass(state, "eval")
                atomic_json_write(state_path, state)
            elif stage == "publish-models":
                for entry in entries:
                    key = f"publish:{entry['size']}"
                    if (state.get("jobs") or {}).get(key, {}).get("status") == "complete":
                        continue
                    publish_one(entry)
                    state.setdefault("jobs", {})[key] = {
                        "status": "complete",
                        "at": utc_now(),
                        "hub_repo": entry["hub_repo"],
                    }
                    atomic_json_write(state_path, state)
                mark_pass(state, "publish-models")
                atomic_json_write(state_path, state)
            else:
                raise ValueError(stage)

        state["status"] = "complete"
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        print("SFT_STRATIFIED_FAMILY_OK")
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
