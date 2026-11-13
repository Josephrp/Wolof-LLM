#!/usr/bin/env python3
"""Config-driven TRL SFT LoRA training from a merged CPT checkpoint."""

from __future__ import annotations

import argparse
import atexit
import importlib.metadata
import json
import os
import sys
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer
except ImportError:
    torch = None
    load_dataset = None
    LoraConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    DataCollatorForSeq2Seq = None
    EarlyStoppingCallback = None
    SFTConfig = None
    SFTTrainer = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.config import load_yaml, sha256_file, validate_sft_config
from common.tracking import TrackingRun, TrackioAlertCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Wolof SFT with TRL and LoRA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="10 steps, at most 1000 train rows")
    parser.add_argument("--resume", action="store_true")
    return parser


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> int:
    if any(dep is None for dep in (torch, load_dataset, LoraConfig, SFTConfig, SFTTrainer)):
        raise RuntimeError("train dependencies are missing; run uv sync --group train")
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    config = validate_sft_config(load_yaml(path), path)
    model_path = Path(config["model"])
    if not model_path.is_absolute() and not str(config["model"]).startswith("LiquidAI/"):
        model_path = ROOT / model_path
    if config["starts_from"] == "merged_cpt" and not model_path.exists():
        raise ValueError(f"merged CPT checkpoint does not exist: {model_path}")
    dataset_path = Path(config["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    train_file = dataset_path / "train.parquet"
    val_file = dataset_path / "val.parquet"
    if not train_file.is_file() or not val_file.is_file():
        raise ValueError(f"prepared SFT train/val parquet missing under {dataset_path}")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), token=token)
    expected_eos = config.get("eos_token_id")
    if expected_eos is not None and tokenizer.eos_token_id != expected_eos:
        raise ValueError(
            f"tokenizer eos_token_id={tokenizer.eos_token_id}, expected {expected_eos}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        token=token,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    datasets = load_dataset(
        "parquet", data_files={"train": str(train_file), "validation": str(val_file)}
    )
    if args.smoke:
        datasets["train"] = datasets["train"].select(range(min(1000, len(datasets["train"]))))
        datasets["validation"] = datasets["validation"].select(
            range(min(200, len(datasets["validation"])))
        )
    training = dict(config["training"])
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    if args.smoke:
        output = ROOT / "results" / "smoke" / f"{config['run']}-sft"
        training["max_steps"] = 10
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (output / "journal_sft.json").is_file():
        print(f"Run already complete, skipping: {output}")
        return 0
    tracking = TrackingRun(
        size=str(config.get("size", config["run"].split("_")[1])),
        stage="sft",
        tokenizer_family=str(config.get("tokenizer_family", "unknown")),
        method="lora",
        seed=int(training.get("seed", 42)),
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", config["run"]),
        config=config,
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)
    max_length = int(training.get("seq_len", 2048))

    def truncate(example):
        return {
            key: example[key][:max_length]
            for key in ("input_ids", "attention_mask", "labels")
        }

    for split in datasets:
        required_columns = {"input_ids", "attention_mask", "labels"}
        if not required_columns <= set(datasets[split].column_names):
            raise ValueError(
                f"{split} is missing explicit assistant-only columns: "
                f"{sorted(required_columns - set(datasets[split].column_names))}"
            )
        datasets[split] = datasets[split].map(
            truncate,
            remove_columns=[
                column
                for column in datasets[split].column_names
                if column not in required_columns
            ],
        )

    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        modules_to_save=lora.get("modules_to_save"),
        task_type="CAUSAL_LM",
        bias="none",
    )
    sft_config = SFTConfig(
        output_dir=str(output),
        max_steps=int(training["max_steps"]),
        per_device_train_batch_size=int(training.get("micro_bs", 1)),
        gradient_accumulation_steps=int(training.get("grad_acc", 8)),
        learning_rate=float(training.get("lr", 2e-4)),
        max_length=max_length,
        bf16=True,
        logging_steps=int(training.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(training.get("eval_steps", 50)),
        save_steps=int(training.get("save_steps", 50)),
        report_to="trackio" if not tracking.disabled else "none",
        trackio_space_id=tracking.space_id,
        trackio_bucket_id=tracking.bucket_id,
        run_name=tracking.name,
        seed=int(training.get("seed", 42)),
        assistant_only_loss=False,
        packing=bool(training.get("packing", False)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(training.get("patience", 5))
            ),
            TrackioAlertCallback(tracking),
        ],
    )
    resume_checkpoint = bool(args.resume and any(output.glob("checkpoint-*")))
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    journal = {
        "run": config["run"],
        "config": str(path),
        "model": str(model_path),
        "starts_from": config["starts_from"],
        "mixture_id": config.get("mixture_id"),
        "task_weights": config.get("task_weights"),
        "assistant_only_labels": "explicit_-100_mask",
        "smoke": args.smoke,
        "steps": trainer.state.global_step,
        "train_loss": result.metrics.get("train_loss"),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "versions": {name: _version(name) for name in ("torch", "transformers", "trl", "peft")},
    }
    (output / "journal_sft.json").write_text(
        json.dumps(journal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tracking.log(
        {
            "train_loss_final": float(result.metrics.get("train_loss", float("nan"))),
            "tokens_total": trainer.state.global_step
            * int(training.get("micro_bs", 1))
            * int(training.get("grad_acc", 8))
            * max_length,
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0
            ),
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
