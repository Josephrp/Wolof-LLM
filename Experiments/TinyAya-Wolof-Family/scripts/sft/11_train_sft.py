#!/usr/bin/env python3
"""TRL SFT LoRA for Tiny Aya Wolof — starts_from merged_cpt | stock_earth | stock_global."""

from __future__ import annotations

import argparse
import atexit
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.checkpointing import make_eval_history_callback, step_train_args
from common.config import load_yaml, sha256_file, validate_sft_config
from common.kernels import (
    attention_context,
    load_causal_lm,
    prefer_sft_packing,
    resolve_attn_implementation,
)
from common.pipeline import atomic_json_write, utc_now
from common.tracking import TrackingRun, TrackioAlertCallback

HUB_PREFIXES = ("CohereLabs/", "Tonic/", "galsenai/", "LiquidAI/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Tiny Aya Wolof SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_model(path_or_id: str) -> str:
    if path_or_id.startswith(HUB_PREFIXES):
        return path_or_id
    p = Path(path_or_id)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer

    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    config = validate_sft_config(load_yaml(path), path)
    model_path = resolve_model(config["model"])
    dataset_path = Path(config["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    train_file = dataset_path / "train.parquet"
    val_file = dataset_path / "val.parquet"
    if not train_file.is_file() or not val_file.is_file():
        raise SystemExit(f"prepared SFT missing under {dataset_path}")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training = dict(config["training"])
    quant = None
    if config.get("methode") == "qlora" or training.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    attn = resolve_attn_implementation(config.get("attn_implementation"))
    model, kernel_meta = load_causal_lm(
        model_path,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        quantization_config=quant,
        attn_implementation=attn,
        apply_cce=config.get("fused_ce", True) is not False,
    )
    if hasattr(model, "resize_token_embeddings"):
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    datasets = load_dataset(
        "parquet", data_files={"train": str(train_file), "validation": str(val_file)}
    )
    if args.smoke:
        datasets["train"] = datasets["train"].select(range(min(256, len(datasets["train"]))))
        datasets["validation"] = datasets["validation"].select(
            range(min(64, len(datasets["validation"])))
        )

    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    if args.smoke:
        output = ROOT / "results" / "smoke" / f"{config['run']}-sft"
        training["num_train_epochs"] = 1
        training["max_steps"] = 20
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and (output / "journal_sft.json").is_file():
        print(f"Run already complete, skipping: {output}")
        return 0

    max_length = int(training.get("seq_len", 2048))

    def truncate(example):
        return {
            key: example[key][:max_length]
            for key in ("input_ids", "attention_mask", "labels")
        }

    for split in datasets:
        required = {"input_ids", "attention_mask", "labels"}
        if not required <= set(datasets[split].column_names):
            raise SystemExit(f"{split} missing {required}")
        datasets[split] = datasets[split].map(
            truncate,
            remove_columns=[c for c in datasets[split].column_names if c not in required],
        )

    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        modules_to_save=lora.get("modules_to_save"),
        task_type="CAUSAL_LM",
        bias="none",
    )

    tracking = TrackingRun(
        size=str(config.get("size", "3b35")),
        stage="sft",
        tokenizer_family=str(config.get("tokenizer_family", "tinyaya-ext")),
        method=str(config.get("methode", "lora")),
        seed=int(training.get("seed", 42)),
        run_id=os.environ.get("TINYAYA_PIPELINE_RUN_ID", config["run"]),
        config=config,
        production=os.environ.get("TINYAYA_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    packing = prefer_sft_packing(
        configured=bool(training.get("packing", False)),
        smoke=bool(args.smoke),
    )
    sft_kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": int(training.get("micro_bs", 1)),
        "gradient_accumulation_steps": int(training.get("grad_acc", 8)),
        "learning_rate": float(training.get("lr", 2e-4)),
        "max_length": max_length,
        "bf16": True,
        "report_to": "trackio" if not tracking.disabled else "none",
        "seed": int(training.get("seed", 42)),
        "assistant_only_loss": False,
        "packing": packing,
        **step_train_args(training, smoke=bool(args.smoke), default_eval=250),
    }
    params = inspect.signature(SFTConfig.__init__).parameters
    for key, value in {
        "trackio_space_id": tracking.space_id,
        "trackio_bucket_id": tracking.bucket_id,
        "run_name": tracking.name,
    }.items():
        if key in params and value:
            sft_kwargs[key] = value
    if args.smoke or training.get("max_steps"):
        sft_kwargs["max_steps"] = int(training.get("max_steps", 20))
    else:
        sft_kwargs["num_train_epochs"] = float(training.get("num_train_epochs", 3))
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in params}

    callbacks = [
        TrackioAlertCallback(tracking),
        make_eval_history_callback(output),
    ]
    # patience<=0 disables early stop so production can finish full epochs.
    patience = int(training.get("patience", 0))
    if patience > 0:
        callbacks.insert(0, EarlyStoppingCallback(early_stopping_patience=patience))
    trainer_kwargs: dict = {
        "model": model,
        "args": SFTConfig(**sft_kwargs),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "processing_class": tokenizer,
        "peft_config": peft_config,
        "callbacks": callbacks,
    }
    # Custom collator is incompatible with TRL padding-free packing.
    if not packing:
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
        )
    trainer = SFTTrainer(**trainer_kwargs)
    resume = bool(args.resume and any(output.glob("checkpoint-*")))
    try:
        with attention_context(kernel_meta["attn_implementation"]):
            result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(str(output / "adapter"))
        journal = {
            "created_at": utc_now(),
            "run": config["run"],
            "model": model_path,
            "starts_from": config["starts_from"],
            "epochs": sft_kwargs.get("num_train_epochs"),
            "max_steps": sft_kwargs.get("max_steps"),
            "train_loss": result.metrics.get("train_loss"),
            "smoke": args.smoke,
            "kernels": kernel_meta,
            "packing": bool(training.get("packing", False)),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock")
            if (ROOT / "uv.lock").is_file()
            else None,
            "trackio": tracking.metadata(),
        }
        atomic_json_write(output / "journal_sft.json", journal)
        tracking.log({"train_loss": result.metrics.get("train_loss")}, step=None)
        tracking.finish(success=True)
        print(json.dumps(journal, indent=2), flush=True)
        return 0
    except Exception as exc:
        tracking.finish(success=False, error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
