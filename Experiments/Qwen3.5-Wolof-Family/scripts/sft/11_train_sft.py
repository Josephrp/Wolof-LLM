#!/usr/bin/env python3
"""TRL SFT LoRA for Qwen3.5 Wolof — starts_from merged_cpt | stock_instruct | stock_base."""

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
from common.checkpoint_pool import BestCheckpointPoolCallback
from common.config import load_yaml, sha256_file, validate_sft_config
from common.kernels import load_causal_lm, maybe_use_liger_training_arg, prefer_sft_packing
from common.pipeline import atomic_json_write, utc_now
from common.tracking import TrackingRun, TrackioAlertCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 Wolof SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_model(path_or_id: str) -> str:
    if path_or_id.startswith(("Qwen/", "Tonic/", "LiquidAI/", "galsenai/")):
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
    # Prefer the tokenizer snapshot used to build prepared_anchored_v1 so anchor
    # specials (<|lang_wo|>, <|ortho_ns|>) keep the same ids as in the parquet.
    prep_tok = dataset_path / "tokenizer_snapshot"
    tok_source = str(prep_tok) if prep_tok.is_dir() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source, token=token, trust_remote_code=True)
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
    model = load_causal_lm(
        model_path,
        token=token,
        quant=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=config.get("attn_implementation")
        or training.get("attn_implementation"),
    )
    embed_rows = model.get_input_embeddings().weight.shape[0]
    if embed_rows != len(tokenizer):
        print(
            f"resize_token_embeddings {embed_rows} -> {len(tokenizer)} "
            f"(tok_source={tok_source})",
            flush=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        # Newly added rows must be trainable when starting from stock weights.
        lora_cfg = dict(config.get("lora") or {})
        modules = list(lora_cfg.get("modules_to_save") or [])
        for name in ("embed_tokens", "lm_head"):
            if name not in modules:
                modules.append(name)
        config["lora"] = {**lora_cfg, "modules_to_save": modules}

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

    vocab_size = model.get_input_embeddings().weight.shape[0]

    def coerce_and_truncate(example):
        def as_ints(values):
            if values and isinstance(values[0], str):
                raise ValueError(
                    "input_ids/labels are strings — regenerate prepared_anchored_v1 "
                    "(apply_chat_template BatchEncoding was list()'d incorrectly)"
                )
            return [int(v) for v in values][:max_length]

        input_ids = as_ints(example["input_ids"])
        attention_mask = as_ints(example["attention_mask"])[: len(input_ids)]
        labels = as_ints(example["labels"])[: len(input_ids)]
        if len(input_ids) < 2:
            raise ValueError(f"example too short: {len(input_ids)} tokens")
        if max(input_ids) >= vocab_size:
            raise ValueError(
                f"token id {max(input_ids)} >= embedding rows {vocab_size}; "
                "tokenizer/model vocab mismatch"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    for split in datasets:
        required = {"input_ids", "attention_mask", "labels"}
        if not required <= set(datasets[split].column_names):
            raise SystemExit(f"{split} missing {required}")
        datasets[split] = datasets[split].map(
            coerce_and_truncate,
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

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable — install torch cu128 matching the host driver "
            f"(torch={torch.__version__} cuda={torch.version.cuda})"
        )

    tracking = TrackingRun(
        size=str(config.get("size", "08b")),
        stage="sft",
        tokenizer_family=str(config.get("tokenizer_family", "qwen35-ext")),
        method="lora",
        seed=int(training.get("seed", 42)),
        run_id=os.environ.get("QWEN_PIPELINE_RUN_ID", config["run"]),
        config=config,
        production=os.environ.get("QWEN_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    sft_kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": int(training.get("micro_bs", 1)),
        "gradient_accumulation_steps": int(training.get("grad_acc", 8)),
        "learning_rate": float(training.get("lr", 2e-4)),
        "max_length": max_length,
        "bf16": True,
        "logging_steps": int(training.get("logging_steps", 10)),
        "eval_strategy": "steps",
        "eval_steps": int(training.get("eval_steps", 50)),
        "save_steps": int(training.get("save_steps", 50)),
        "report_to": "trackio" if not tracking.disabled else "none",
        "trackio_space_id": tracking.space_id,
        "trackio_bucket_id": tracking.bucket_id,
        "run_name": tracking.name,
        "seed": int(training.get("seed", 42)),
        "assistant_only_loss": False,
        "packing": prefer_sft_packing(
            smoke=bool(args.smoke),
            config_packing=bool(training["packing"]) if "packing" in training else None,
        ),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    pool_keep = int(training.get("best_pool_keep", 15))
    if args.smoke or training.get("max_steps"):
        sft_kwargs["max_steps"] = int(training.get("max_steps", 20))
    else:
        sft_kwargs["num_train_epochs"] = float(training.get("num_train_epochs", 3))

    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in accepted or k == "output_dir"}
    sft_kwargs = maybe_use_liger_training_arg(sft_kwargs, accepted)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=int(training.get("patience", 5))),
            TrackioAlertCallback(tracking),
            BestCheckpointPoolCallback(output, keep=pool_keep, metric="eval_loss"),
        ],
    )
    resume = bool(args.resume and any(output.glob("checkpoint-*")))
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
        "kernels": getattr(model, "_qwen_kernel_meta", None),
        "packing": sft_kwargs.get("packing"),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "trackio": tracking.metadata(),
    }
    atomic_json_write(output / "journal_sft.json", journal)
    tracking.log(
        {
            "train_loss_final": float(result.metrics.get("train_loss", float("nan"))),
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    print(json.dumps(journal, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
