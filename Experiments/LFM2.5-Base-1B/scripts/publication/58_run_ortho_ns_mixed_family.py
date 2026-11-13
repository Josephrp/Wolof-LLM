#!/usr/bin/env python3
"""Durable ortho NS+mixed v3 family: datasets → tok → CPT×5 → SFT×5 → eval → Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write

CPT_FAMILY = [
    {
        "size": "230m",
        "config": "configs/clm/230m/v3_ortho_ns_mixed_lora.yaml",
        "run": "v3_230m_ortho_ns_mixed_lora",
        "run_dir": "runs/clm/230m/v3_ortho_ns_mixed_lora",
        "merged": "models/v3_230m_ortho_ns_mixed_lora-merged",
        "init": "models/lfm25-230m-wolof-ext4000-ortho-ns-mixed-init",
        "eos": "7",
        "tok_family": "65k-ext",
        "base_stock": "LiquidAI/LFM2.5-230M-Base",
        "hub_repo": "Tonic/LFM2.5-230M-Wolof-CPT-v3",
        "clm_metrics": "metrics/clm/lfm_230m_cpt_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_230m_cpt_v3_original_test.json",
    },
    {
        "size": "350m",
        "config": "configs/clm/350m/v3_ortho_ns_mixed_lora.yaml",
        "run": "v3_350m_ortho_ns_mixed_lora",
        "run_dir": "runs/clm/350m/v3_ortho_ns_mixed_lora",
        "merged": "models/v3_350m_ortho_ns_mixed_lora-merged",
        "init": "models/lfm25-350m-wolof-ext4000-ortho-ns-mixed-init",
        "eos": "7",
        "tok_family": "65k-ext",
        "base_stock": "LiquidAI/LFM2.5-350M-Base",
        "hub_repo": "Tonic/LFM2.5-350M-Wolof-CPT-v3",
        "clm_metrics": "metrics/clm/lfm_350m_cpt_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_350m_cpt_v3_original_test.json",
    },
    {
        "size": "1.2b",
        "config": "configs/clm/1.2b/v3_ortho_ns_mixed_lora.yaml",
        "run": "v3_1.2b_ortho_ns_mixed_lora",
        "run_dir": "runs/clm/1.2b/v3_ortho_ns_mixed_lora",
        "merged": "models/v3_1.2b_ortho_ns_mixed_lora-merged",
        "init": "models/lfm25-wolof-ext4000-ortho-ns-mixed-init",
        "eos": "7",
        "tok_family": "65k-ext",
        "base_stock": "LiquidAI/LFM2.5-1.2B-Base",
        "hub_repo": "Tonic/LFM2.5-1.2B-Wolof-CPT-v3",
        "clm_metrics": "metrics/clm/lfm_1.2b_cpt_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_1.2b_cpt_v3_original_test.json",
    },
    {
        "size": "2.6b",
        "config": "configs/clm/2.6b/v3_ortho_ns_mixed_lora.yaml",
        "run": "v3_2.6b_ortho_ns_mixed_lora",
        "run_dir": "runs/clm/2.6b/v3_ortho_ns_mixed_lora",
        "merged": "models/v3_2.6b_ortho_ns_mixed_lora-merged",
        "init": "models/lfm25-2.6b-wolof-ext4000-ortho-ns-mixed-init",
        "eos": "128900",
        "tok_family": "128k-ext",
        "base_stock": "LiquidAI/LFM2.5-2.6B-Base",
        "hub_repo": "Tonic/LFM2.5-2.6B-Wolof-CPT-v3",
        "clm_metrics": "metrics/clm/lfm_2.6b_cpt_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_2.6b_cpt_v3_original_test.json",
    },
    {
        "size": "8b-a1b",
        "config": "configs/clm/8b/v3_ortho_ns_mixed_lora.yaml",
        "run": "v3_8b_a1b_ortho_ns_mixed_lora",
        # Config size label is 8b-a1b but output_root uses runs/clm/8b/.
        "run_dir": "runs/clm/8b/v3_ortho_ns_mixed_lora",
        "merged": "models/v3_8b_a1b_ortho_ns_mixed_lora-merged",
        "init": "models/lfm25-8b-a1b-wolof-ext4000-ortho-ns-mixed-init",
        "eos": "128900",
        "tok_family": "128k-ext",
        "base_stock": "LiquidAI/LFM2.5-8B-A1B-Base",
        "hub_repo": "Tonic/LFM2.5-8B-A1B-Wolof-CPT-v3",
        "clm_metrics": "metrics/clm/lfm_8b_cpt_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_8b_cpt_v3_original_test.json",
    },
]

SFT_FAMILY = [
    {
        "size": "230m",
        "config": "configs/sft/230m/full_ortho_ns_mixed_v1.yaml",
        "cpt_merged": "models/v3_230m_ortho_ns_mixed_lora-merged",
        "adapter": "runs/sft/230m_full_ortho_ns_mixed_v1/adapter",
        "merged": "models/lfm25-230m-wolof-instruct-ortho-ns-mixed-merged",
        "eos": "7",
        "tok_family": "65k-ext",
        "sft_data": "data/sft/prepared_ortho_ns_mixed_v1",
        "base_model": "Tonic/LFM2.5-230M-Wolof-CPT-v3",
        "hub_repo": "Tonic/LFM2.5-230M-Wolof-Instruct-v3",
        "clm_metrics": "metrics/clm/lfm_230m_instruct_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_230m_instruct_v3_original_test.json",
        "eval_dir": "results/full_pipeline/eval_ortho_ns_mixed/230m",
    },
    {
        "size": "350m",
        "config": "configs/sft/350m/full_ortho_ns_mixed_v1.yaml",
        "cpt_merged": "models/v3_350m_ortho_ns_mixed_lora-merged",
        "adapter": "runs/sft/350m_full_ortho_ns_mixed_v1/adapter",
        "merged": "models/lfm25-350m-wolof-instruct-ortho-ns-mixed-merged",
        "eos": "7",
        "tok_family": "65k-ext",
        "sft_data": "data/sft/prepared_ortho_ns_mixed_v1",
        "base_model": "Tonic/LFM2.5-350M-Wolof-CPT-v3",
        "hub_repo": "Tonic/LFM2.5-350M-Wolof-Instruct-v3",
        "clm_metrics": "metrics/clm/lfm_350m_instruct_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_350m_instruct_v3_original_test.json",
        "eval_dir": "results/full_pipeline/eval_ortho_ns_mixed/350m",
    },
    {
        "size": "1.2b",
        "config": "configs/sft/1.2b/full_ortho_ns_mixed_v1.yaml",
        "cpt_merged": "models/v3_1.2b_ortho_ns_mixed_lora-merged",
        "adapter": "runs/sft/1.2b_full_ortho_ns_mixed_v1/adapter",
        "merged": "models/lfm25-1.2b-wolof-instruct-ortho-ns-mixed-merged",
        "eos": "7",
        "tok_family": "65k-ext",
        "sft_data": "data/sft/prepared_ortho_ns_mixed_v1",
        "base_model": "Tonic/LFM2.5-1.2B-Wolof-CPT-v3",
        "hub_repo": "Tonic/LFM2.5-1.2B-Wolof-Instruct-v3",
        "clm_metrics": "metrics/clm/lfm_1.2b_instruct_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_1.2b_instruct_v3_original_test.json",
        "eval_dir": "results/full_pipeline/eval_ortho_ns_mixed/1.2b",
    },
    {
        "size": "2.6b",
        "config": "configs/sft/2.6b/full_ortho_ns_mixed_v1.yaml",
        "cpt_merged": "models/v3_2.6b_ortho_ns_mixed_lora-merged",
        "adapter": "runs/sft/2.6b_full_ortho_ns_mixed_v1/adapter",
        "merged": "models/lfm25-2.6b-wolof-instruct-ortho-ns-mixed-merged",
        "eos": "128900",
        "tok_family": "128k-ext",
        "sft_data": "data/sft/prepared_ortho_ns_mixed_v1_128k",
        "base_model": "Tonic/LFM2.5-2.6B-Wolof-CPT-v3",
        "hub_repo": "Tonic/LFM2.5-2.6B-Wolof-Instruct-v3",
        "clm_metrics": "metrics/clm/lfm_2.6b_instruct_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_2.6b_instruct_v3_original_test.json",
        "eval_dir": "results/full_pipeline/eval_ortho_ns_mixed/2.6b",
    },
    {
        "size": "8b-a1b",
        "config": "configs/sft/8b/full_ortho_ns_mixed_v1.yaml",
        "cpt_merged": "models/v3_8b_a1b_ortho_ns_mixed_lora-merged",
        "adapter": "runs/sft/8b_a1b_full_ortho_ns_mixed_v1/adapter",
        "merged": "models/lfm25-8b-a1b-wolof-instruct-ortho-ns-mixed-merged",
        "eos": "128900",
        "tok_family": "128k-ext",
        "sft_data": "data/sft/prepared_ortho_ns_mixed_v1_128k",
        "base_model": "Tonic/LFM2.5-8B-A1B-Wolof-CPT-v3",
        "hub_repo": "Tonic/LFM2.5-8B-A1B-Wolof-Instruct-v3",
        "clm_metrics": "metrics/clm/lfm_8b_instruct_v3.json",
        "clm_metrics_secondary": "metrics/clm/lfm_8b_instruct_v3_original_test.json",
        "eval_dir": "results/full_pipeline/eval_ortho_ns_mixed/8b-a1b",
    },
]

TOK_65K = "tokenizers/lfm25-wolof-ext4000-ortho-ns-mixed-v1"
TOK_65K_230M = "tokenizers/lfm25_230m_wolof_ext_4000_ortho_ns_mixed_v1"
TOK_128K = "tokenizers/lfm25-wolof-ext4000-ortho-ns-mixed-v1-128k"


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


def job_done(state: dict, name: str) -> bool:
    return (state.get("jobs") or {}).get(name, {}).get("status") == "complete"


def mark_pass(state: dict, name: str, **extra) -> None:
    state.setdefault("stages", {})[name] = {"status": "pass", "at": utc_now(), **extra}


def mark_job(state: dict, name: str, **extra) -> None:
    state.setdefault("jobs", {})[name] = {"status": "complete", "at": utc_now(), **extra}


def python_bin() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def extension_start(tokenizer_dir: Path, base_id: str) -> int:
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(base_id)
    ext = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return len(base)


def smoke_forward(model_dir: Path, expected_eos: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.eos_token_id != expected_eos:
        raise RuntimeError(
            f"eos mismatch for {model_dir}: got {tok.eos_token_id}, expected {expected_eos}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    ids = tok("naka nga def", return_tensors="pt")
    if torch.cuda.is_available():
        ids = {k: v.cuda() for k, v in ids.items()}
    with torch.no_grad():
        out = model(**ids)
    if not torch.isfinite(out.logits).all():
        raise RuntimeError(f"non-finite logits for {model_dir}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_datasets(state: dict, state_path: Path, *, smoke: bool) -> None:
    if stage_done(state, "datasets"):
        return
    cmd = [
        python_bin(),
        "scripts/data/26_build_ortho_ns_mixed_datasets.py",
        "--namespace",
        "Tonic",
    ]
    if smoke:
        cmd.append("--smoke")
    run_cmd(cmd)
    mark_pass(state, "datasets")
    atomic_json_write(state_path, state)


def stage_tokenizer(state: dict, state_path: Path, *, smoke: bool) -> None:
    if not stage_done(state, "tokenizer-65k"):
        cmd = [
            python_bin(),
            "scripts/clm/02_build_tokenizer.py",
            "--config",
            "configs/tokenizer/tokenizer-ext4000-65k-ortho-ns-mixed.yaml",
        ]
        if smoke:
            cmd.extend(["--max-rows", "2000", "--n-echantillon", "2000"])
        run_cmd(cmd)
        # Prefer local parquet when Hub path is flaky: rebuild already used Hub via token.
        smoke_forward(ROOT / "models/lfm25-wolof-ext4000-ortho-ns-mixed-init", 7)
        mark_pass(state, "tokenizer-65k")
        atomic_json_write(state_path, state)

    if not stage_done(state, "tokenizer-port-230m"):
        src = ROOT / TOK_65K
        start = extension_start(src, "LiquidAI/LFM2.5-1.2B-Base")
        run_cmd(
            [
                python_bin(),
                "scripts/clm/02c_port_ext_tokenizer_to_base.py",
                "--base-id",
                "LiquidAI/LFM2.5-230M-Base",
                "--source-tokenizer",
                TOK_65K,
                "--source-append-start",
                str(start),
                "--output-dir",
                TOK_65K_230M,
            ]
        )
        mark_pass(state, "tokenizer-port-230m", append_start=start)
        atomic_json_write(state_path, state)

    if not stage_done(state, "tokenizer-128k"):
        cmd = [
            python_bin(),
            "scripts/clm/02_build_tokenizer.py",
            "--config",
            "configs/tokenizer-ext4000-128k-ortho-ns-mixed.yaml",
        ]
        if smoke:
            cmd.extend(["--max-rows", "2000", "--n-echantillon", "2000"])
        run_cmd(cmd)
        smoke_forward(ROOT / "models/lfm25-2.6b-wolof-ext4000-ortho-ns-mixed-init", 128900)
        mark_pass(state, "tokenizer-128k")
        atomic_json_write(state_path, state)


def stage_resize(state: dict, state_path: Path) -> None:
    if stage_done(state, "resize"):
        return
    # 230M from ported tokenizer
    if not (ROOT / "models/lfm25-230m-wolof-ext4000-ortho-ns-mixed-init").exists():
        run_cmd(
            [
                python_bin(),
                "scripts/clm/02b_resize_ext_to_size.py",
                "--base-id",
                "LiquidAI/LFM2.5-230M-Base",
                "--source-tokenizer",
                TOK_65K_230M,
                "--output-dir",
                "models/lfm25-230m-wolof-ext4000-ortho-ns-mixed-init",
                "--expected-eos",
                "7",
            ]
        )
        smoke_forward(ROOT / "models/lfm25-230m-wolof-ext4000-ortho-ns-mixed-init", 7)
    # 350M from shared 65k tokenizer
    if not (ROOT / "models/lfm25-350m-wolof-ext4000-ortho-ns-mixed-init").exists():
        run_cmd(
            [
                python_bin(),
                "scripts/clm/02b_resize_ext_to_size.py",
                "--base-id",
                "LiquidAI/LFM2.5-350M-Base",
                "--source-tokenizer",
                TOK_65K,
                "--output-dir",
                "models/lfm25-350m-wolof-ext4000-ortho-ns-mixed-init",
                "--expected-eos",
                "7",
            ]
        )
        smoke_forward(ROOT / "models/lfm25-350m-wolof-ext4000-ortho-ns-mixed-init", 7)
    # 8B from 128k tokenizer (2.6b init already created by tokenizer build)
    if not (ROOT / "models/lfm25-8b-a1b-wolof-ext4000-ortho-ns-mixed-init").exists():
        run_cmd(
            [
                python_bin(),
                "scripts/clm/02b_resize_ext_to_size.py",
                "--base-id",
                "LiquidAI/LFM2.5-8B-A1B-Base",
                "--source-tokenizer",
                TOK_128K,
                "--output-dir",
                "models/lfm25-8b-a1b-wolof-ext4000-ortho-ns-mixed-init",
                "--expected-eos",
                "128900",
            ]
        )
        smoke_forward(ROOT / "models/lfm25-8b-a1b-wolof-ext4000-ortho-ns-mixed-init", 128900)
    mark_pass(state, "resize")
    atomic_json_write(state_path, state)


def stage_prepare_sft(state: dict, state_path: Path, *, smoke: bool) -> None:
    if not stage_done(state, "prepare-65k"):
        cmd = [
            python_bin(),
            "scripts/sft/10_prepare_sft.py",
            "--input",
            "data/sft/ortho_ns_mixed_stratified_v1",
            "--output",
            "data/sft/prepared_ortho_ns_mixed_v1",
            "--tokenizer-model",
            "models/lfm25-wolof-ext4000-ortho-ns-mixed-init",
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
            "data/sft/ortho_ns_mixed_stratified_v1",
            "--output",
            "data/sft/prepared_ortho_ns_mixed_v1_128k",
            "--template-model",
            "LiquidAI/LFM2.5-2.6B",
            "--tokenizer-model",
            "models/lfm25-2.6b-wolof-ext4000-ortho-ns-mixed-init",
        ]
        if smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
        mark_pass(state, "prepare-128k")
        atomic_json_write(state_path, state)
    if not stage_done(state, "publish-prepared"):
        run_cmd(
            [
                python_bin(),
                "scripts/publication/47_publish_training_datasets.py",
                "--namespace",
                "Tonic",
                "--skip-cpt",
                "--skip-sft-used",
                "--include-ortho-ns-mixed",
            ]
        )
        mark_pass(state, "publish-prepared")
        atomic_json_write(state_path, state)


def _stage_cpt_train(
    state: dict,
    state_path: Path,
    family: list[dict],
    *,
    smoke: bool,
    resume: bool,
) -> None:
    for entry in family:
        key = f"cpt-train:{entry['size']}"
        if job_done(state, key):
            continue
        cmd = [python_bin(), "scripts/clm/03_train.py", "--config", entry["config"]]
        if resume:
            cmd.append("--resume")
        if smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
        mark_job(state, key)
        atomic_json_write(state_path, state)
    mark_pass(state, "cpt-train")
    atomic_json_write(state_path, state)


def _stage_cpt_merge(
    state: dict, state_path: Path, family: list[dict], *, smoke: bool
) -> None:
    for entry in family:
        key = f"cpt-merge:{entry['size']}"
        if job_done(state, key):
            continue
        run_name = entry["run"] + ("-smoke" if smoke else "")
        run_cmd(
            [
                python_bin(),
                "scripts/clm/05_merge_adapter.py",
                "--run",
                run_name,
                "--dataset-id",
                "Tonic/LFM2.5-Wolof-CPT-ortho_ns_mixed_v1",
                "--data-root",
                "data/clm/ortho_ns_mixed_v1",
            ]
        )
        # Canonical alias without -lora suffix for docs/plan naming.
        merged = ROOT / entry["merged"]
        alias = ROOT / f"models/v3_{entry['size'].replace('-', '_')}_ortho_ns_mixed-merged"
        if merged.is_dir() and not alias.exists():
            try:
                alias.symlink_to(merged.resolve())
            except OSError:
                pass
        mark_job(state, key, merged=entry["merged"])
        atomic_json_write(state_path, state)
    mark_pass(state, "cpt-merge")
    atomic_json_write(state_path, state)


def stage_cpt_train(state: dict, state_path: Path, *, smoke: bool, resume: bool) -> None:
    _stage_cpt_train(state, state_path, CPT_FAMILY, smoke=smoke, resume=resume)


def stage_cpt_merge(state: dict, state_path: Path, *, smoke: bool) -> None:
    _stage_cpt_merge(state, state_path, CPT_FAMILY, smoke=smoke)


def publish_model(
    *,
    artifact: str,
    hub_repo: str,
    base_model: str,
    tok_family: str,
    stage: str,
    metrics_path: Path,
    journal_path: Path,
    clm_primary: Path | None,
    clm_secondary: Path | None,
    commit_message: str,
) -> None:
    from huggingface_hub import HfApi, get_token

    run_cmd(
        [
            python_bin(),
            "scripts/publication/43_prepare_artifact.py",
            "--artifact",
            artifact,
            "--metrics",
            str(metrics_path.relative_to(ROOT)) if metrics_path.is_relative_to(ROOT) else str(metrics_path),
            "--run-manifest",
            str(journal_path.relative_to(ROOT)) if journal_path.is_relative_to(ROOT) else str(journal_path),
            "--model-name",
            hub_repo.split("/")[-1],
            "--base-model",
            base_model,
            "--tokenizer-family",
            tok_family,
            "--training-stage",
            stage,
        ]
    )
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=hub_repo, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(hub_repo, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo {hub_repo}")
    api.upload_folder(
        folder_path=str(ROOT / artifact),
        repo_id=hub_repo,
        repo_type="model",
        commit_message=commit_message,
    )
    if clm_primary and clm_primary.is_file():
        api.upload_file(
            path_or_fileobj=str(clm_primary),
            path_in_repo="metrics_clm.json",
            repo_id=hub_repo,
            repo_type="model",
            commit_message="Add primary (ortho_ns_mixed) CLM metrics",
        )
    if clm_secondary and clm_secondary.is_file():
        api.upload_file(
            path_or_fileobj=str(clm_secondary),
            path_in_repo="metrics_clm_original_test.json",
            repo_id=hub_repo,
            repo_type="model",
            commit_message="Add secondary (original clean test) CLM metrics",
        )
    run_cmd(
        [
            python_bin(),
            "scripts/publication/56_publish_hub_metrics.py",
            "--only-repo",
            hub_repo,
        ]
    )


def evaluate_clm_dual(
    *,
    model_path: Path,
    primary_out: str,
    secondary_out: str,
    eval_root: Path,
    template_config: str,
) -> None:
    cfg = yaml.safe_load((eval_root / template_config).read_text(encoding="utf-8"))
    cfg["base_model_id"] = str(model_path.resolve())
    cfg.pop("subfolder", None)
    cfg["placeholder"] = False

    primary_cfg = dict(cfg)
    primary_cfg["corpus_csv"] = str(
        (eval_root / "data" / "corpus_test_clm_ortho_ns_mixed.csv").resolve()
    )
    primary_cfg["output_json"] = primary_out

    secondary_cfg = dict(cfg)
    secondary_cfg["corpus_csv"] = str((eval_root / "data" / "corpus_test_clm.csv").resolve())
    secondary_cfg["output_json"] = secondary_out

    for payload in (primary_cfg, secondary_cfg):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
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


def _stage_cpt_eval_publish(
    state: dict,
    state_path: Path,
    family: list[dict],
    *,
    smoke: bool,
    eval_root: Path,
) -> None:
    template = {
        "230m": "configs/clm/lfm_230m_clm.yaml",
        "350m": "configs/clm/lfm_350m_clm.yaml",
        "1.2b": "configs/clm/lfm_1.2b_clm.yaml",
        "2.6b": "configs/clm/lfm_2.6b_clm.yaml",
        "8b-a1b": "configs/clm/lfm_8b_clm.yaml",
    }
    for entry in family:
        key = f"cpt-eval-publish:{entry['size']}"
        if job_done(state, key):
            continue
        merged = ROOT / entry["merged"]
        if not smoke:
            evaluate_clm_dual(
                model_path=merged,
                primary_out=entry["clm_metrics"],
                secondary_out=entry["clm_metrics_secondary"],
                eval_root=eval_root,
                template_config=template[entry["size"]],
            )
        journal = None
        for candidate in (
            ROOT / entry.get("run_dir", f"runs/clm/{entry['size']}/v3_ortho_ns_mixed_lora") / "journal.json",
            ROOT / f"runs/clm/{entry['size']}/v3_ortho_ns_mixed_lora/journal.json",
            ROOT / f"runs/clm/8b/v3_ortho_ns_mixed_lora/journal.json",
            ROOT / f"results/smoke/{entry['run']}-smoke/journal.json",
        ):
            if candidate.is_file():
                journal = candidate
                break
        if journal is None:
            raise FileNotFoundError(f"missing CPT journal for {entry['size']}")
        metrics_src = eval_root / entry["clm_metrics"]
        if not metrics_src.is_file():
            # smoke: write a stub metrics file so 43_prepare can run
            stub = ROOT / "results" / "full_pipeline" / f"stub_{entry['size']}_cpt.json"
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_text(
                json.dumps({"bpb": None, "note": "smoke"}, indent=2) + "\n",
                encoding="utf-8",
            )
            metrics_src = stub
        # Copy primary metrics into artifact path expected by prepare.
        local_metrics = ROOT / "results" / "full_pipeline" / f"cpt_{entry['size']}_metrics.json"
        local_metrics.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_src, local_metrics)
        publish_model(
            artifact=entry["merged"],
            hub_repo=entry["hub_repo"],
            base_model=entry["base_stock"],
            tok_family=entry["tok_family"],
            stage="CPT",
            metrics_path=local_metrics,
            journal_path=journal,
            clm_primary=eval_root / entry["clm_metrics"] if (eval_root / entry["clm_metrics"]).is_file() else None,
            clm_secondary=(
                eval_root / entry["clm_metrics_secondary"]
                if (eval_root / entry["clm_metrics_secondary"]).is_file()
                else None
            ),
            commit_message="Publish LFM2.5 Wolof CPT-v3 (ortho NS+mixed)",
        )
        mark_job(state, key, hub_repo=entry["hub_repo"])
        atomic_json_write(state_path, state)
    mark_pass(state, "cpt-eval-publish")
    atomic_json_write(state_path, state)


def _stage_sft_train(
    state: dict,
    state_path: Path,
    family: list[dict],
    *,
    smoke: bool,
    resume: bool,
) -> None:
    for entry in family:
        key = f"sft-train:{entry['size']}"
        if job_done(state, key):
            continue
        if not (ROOT / entry["cpt_merged"]).exists():
            raise FileNotFoundError(entry["cpt_merged"])
        cmd = [python_bin(), "scripts/sft/11_train_sft.py", "--config", entry["config"]]
        if resume:
            cmd.append("--resume")
        if smoke:
            cmd.append("--smoke")
        run_cmd(cmd)
        mark_job(state, key)
        atomic_json_write(state_path, state)
    mark_pass(state, "sft-train")
    atomic_json_write(state_path, state)


def _stage_sft_merge(
    state: dict, state_path: Path, family: list[dict], *, smoke: bool
) -> None:
    for entry in family:
        key = f"sft-merge:{entry['size']}"
        if job_done(state, key):
            continue
        adapter = entry["adapter"]
        if smoke:
            smoke_dirs = sorted(
                (ROOT / "results" / "smoke").glob("*ortho_ns_mixed*sft")
            )
            if smoke_dirs:
                adapter = str((smoke_dirs[-1] / "adapter").relative_to(ROOT))
        run_cmd(
            [
                python_bin(),
                "scripts/sft/13_merge_sft.py",
                "--base",
                entry["cpt_merged"],
                "--adapter",
                adapter,
                "--output",
                entry["merged"],
                "--expected-eos",
                entry["eos"],
            ]
        )
        mark_job(state, key, merged=entry["merged"])
        atomic_json_write(state_path, state)
    mark_pass(state, "sft-merge")
    atomic_json_write(state_path, state)


def _stage_sft_eval_publish(
    state: dict,
    state_path: Path,
    family: list[dict],
    *,
    smoke: bool,
    eval_root: Path,
) -> None:
    template = {
        "230m": "configs/clm/lfm_230m_instruct.yaml",
        "350m": "configs/clm/lfm_350m_instruct.yaml",
        "1.2b": "configs/clm/lfm_1.2b_instruct.yaml",
        "2.6b": "configs/clm/lfm_2.6b_instruct.yaml",
        "8b-a1b": "configs/clm/lfm_8b_instruct.yaml",
    }
    for entry in family:
        key = f"sft-eval-publish:{entry['size']}"
        if job_done(state, key):
            continue
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
        if not smoke:
            evaluate_clm_dual(
                model_path=ROOT / entry["merged"],
                primary_out=entry["clm_metrics"],
                secondary_out=entry["clm_metrics_secondary"],
                eval_root=eval_root,
                template_config=template[entry["size"]],
            )
        cfg = yaml.safe_load((ROOT / entry["config"]).read_text(encoding="utf-8"))
        journal = ROOT / cfg["output_dir"] / "journal_sft.json"
        if not journal.is_file():
            candidates = list(
                (ROOT / "results" / "smoke").glob("*ortho_ns_mixed*sft/journal_sft.json")
            )
            if candidates:
                journal = candidates[-1]
        if not journal.is_file():
            raise FileNotFoundError(f"missing SFT journal for {entry['size']}")
        metrics = out / "evaluation.json"
        publish_model(
            artifact=entry["merged"],
            hub_repo=entry["hub_repo"],
            base_model=entry["base_model"],
            tok_family=entry["tok_family"],
            stage="Instruct",
            metrics_path=metrics,
            journal_path=journal,
            clm_primary=eval_root / entry["clm_metrics"] if (eval_root / entry["clm_metrics"]).is_file() else None,
            clm_secondary=(
                eval_root / entry["clm_metrics_secondary"]
                if (eval_root / entry["clm_metrics_secondary"]).is_file()
                else None
            ),
            commit_message="Publish LFM2.5 Wolof Instruct-v3 (ortho NS+mixed)",
        )
        mark_job(state, key, hub_repo=entry["hub_repo"])
        atomic_json_write(state_path, state)
    mark_pass(state, "sft-eval-publish")
    atomic_json_write(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "ortho_ns_mixed_state.json",
    )
    parser.add_argument(
        "--only-stage",
        choices=(
            "datasets",
            "tokenizer",
            "resize",
            "prepare_sft",
            "cpt_train",
            "cpt_merge",
            "cpt_eval_publish",
            "sft_train",
            "sft_merge",
            "sft_eval_publish",
        ),
        default=None,
    )
    parser.add_argument("--only-size", action="append", default=[])
    args = parser.parse_args()

    load_trackio_env()
    os.environ.setdefault("LFM_PRODUCTION", "1")
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema_version": 1, "status": "running", "stages": {}, "jobs": {}}
    state["status"] = "running"
    state["updated_at"] = utc_now()
    atomic_json_write(state_path, state)

    cpt_family = CPT_FAMILY
    sft_family = SFT_FAMILY
    if args.only_size:
        wanted = set(args.only_size)
        cpt_family = [e for e in CPT_FAMILY if e["size"] in wanted]
        sft_family = [e for e in SFT_FAMILY if e["size"] in wanted]

    eval_root = Path(
        os.environ.get(
            "EVALUATION_ROOT",
            "/mnt/volume_w2l8nkp/Wolof-LLM-main/Evaluation",
        )
    )

    stages = (
        [
            "datasets",
            "tokenizer",
            "resize",
            "cpt_train",
            "cpt_merge",
            "prepare_sft",
            "cpt_eval_publish",
            "sft_train",
            "sft_merge",
            "sft_eval_publish",
        ]
        if args.only_stage is None
        else [args.only_stage]
    )

    try:
        for stage in stages:
            if stage == "datasets":
                stage_datasets(state, state_path, smoke=args.smoke)
            elif stage == "tokenizer":
                stage_tokenizer(state, state_path, smoke=args.smoke)
            elif stage == "resize":
                stage_resize(state, state_path)
            elif stage == "prepare_sft":
                stage_prepare_sft(state, state_path, smoke=args.smoke)
            elif stage == "cpt_train":
                _stage_cpt_train(
                    state, state_path, cpt_family, smoke=args.smoke, resume=args.resume
                )
            elif stage == "cpt_merge":
                _stage_cpt_merge(state, state_path, cpt_family, smoke=args.smoke)
            elif stage == "cpt_eval_publish":
                _stage_cpt_eval_publish(
                    state,
                    state_path,
                    cpt_family,
                    smoke=args.smoke,
                    eval_root=eval_root,
                )
            elif stage == "sft_train":
                _stage_sft_train(
                    state, state_path, sft_family, smoke=args.smoke, resume=args.resume
                )
            elif stage == "sft_merge":
                _stage_sft_merge(state, state_path, sft_family, smoke=args.smoke)
            elif stage == "sft_eval_publish":
                _stage_sft_eval_publish(
                    state,
                    state_path,
                    sft_family,
                    smoke=args.smoke,
                    eval_root=eval_root,
                )
            else:
                raise ValueError(stage)

        state["status"] = "complete"
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        print("ORTHO_NS_MIXED_FAMILY_OK")
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
