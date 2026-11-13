#!/usr/bin/env python3
"""Continual pretrain Qwen3.5 Base on ortho-filtered NTP (epoch-driven LoRA/QLoRA)."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.checkpoint_pool import BestCheckpointPoolCallback
from common.config import validate_clm_config
from common.kernels import load_causal_lm, maybe_use_liger_training_arg
from common.pipeline import atomic_json_write, utc_now
from common.tracking import TrackingRun, TrackioAlertCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5 Wolof CPT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = validate_clm_config(yaml.safe_load(cfg_path.read_text(encoding="utf-8")), cfg_path)
    e = dict(cfg["entrainement"])
    token = os.environ.get("HF_TOKEN")
    model_id = cfg.get("model") or cfg["base_id"]
    tok_id = cfg.get("tokenizer") or model_id
    if not str(model_id).startswith("Qwen/") and not Path(model_id).is_absolute():
        model_id = str(ROOT / model_id)
    if not str(tok_id).startswith("Qwen/") and not Path(tok_id).is_absolute():
        tok_id = str(ROOT / tok_id)

    data_root = Path(cfg.get("data_root") or ROOT / "data" / "ortho_ns_mixed_v1" / "cpt")
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    train_file = data_root / "train.parquet"
    val_file = data_root / "test.parquet"
    ds = load_dataset(
        "parquet",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(512, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(128, len(ds["validation"]))))

    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if cfg["methode"] == "qlora":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = load_causal_lm(
        model_id,
        token=token,
        quant=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=cfg.get("attn_implementation") or e.get("attn_implementation"),
    )
    if cfg.get("extended_tokenizer") and hasattr(model, "resize_token_embeddings"):
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    seq_len = int(e.get("seq_len", 2048))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=seq_len,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    max_eval_samples = e.get("max_eval_samples")
    if max_eval_samples:
        n_eval = min(int(max_eval_samples), len(tokenized["validation"]))
        tokenized["validation"] = tokenized["validation"].select(range(n_eval))
        print(f"eval capped to {n_eval} samples", flush=True)
    lora_cfg = cfg.get("lora") or {}
    if cfg["methode"] in {"lora", "qlora"}:
        peft = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            modules_to_save=lora_cfg.get("modules_to_save"),
            bias="none",
        )
        model = get_peft_model(model, peft)
        model.print_trainable_parameters()

    out = Path(cfg.get("output_root") or f"runs/clm/{cfg['size']}/{cfg['run']}")
    if not out.is_absolute():
        out = ROOT / out
    if args.smoke:
        out = ROOT / "results" / "smoke" / f"{cfg['run']}-cpt"
        e["num_train_epochs"] = 1
        e["max_steps"] = 20
    out.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise SystemExit(
            "CUDA unavailable — install torch cu128 matching the host driver "
            f"(torch={torch.__version__} cuda={torch.version.cuda})"
        )

    tracking = TrackingRun(
        size=str(cfg.get("size", "08b")),
        stage="cpt",
        tokenizer_family="qwen35-ext",
        method=str(cfg.get("methode", "lora")),
        seed=int(e.get("seed", 42)),
        run_id=os.environ.get("QWEN_PIPELINE_RUN_ID", cfg["run"]),
        config=cfg,
        production=os.environ.get("QWEN_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=out / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    # Prefer epochs (user request: 3); smoke overrides with max_steps.
    train_args = {
        "output_dir": str(out),
        "per_device_train_batch_size": int(e.get("micro_bs", 1)),
        "per_device_eval_batch_size": int(e.get("eval_micro_bs", e.get("micro_bs", 1))),
        "gradient_accumulation_steps": int(e.get("grad_acc", 8)),
        "learning_rate": float(e.get("lr", 2e-4)),
        "logging_steps": int(e.get("logging_steps", 10)),
        "eval_strategy": "steps",
        "eval_steps": int(e.get("eval_steps", 100)),
        "save_steps": int(e.get("save_steps", 100)),
        "bf16": True,
        "gradient_checkpointing": bool(e.get("gradient_checkpointing", True)),
        "dataloader_num_workers": int(e.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": True,
        "report_to": "trackio" if not tracking.disabled else "none",
        "trackio_space_id": tracking.space_id,
        "trackio_bucket_id": tracking.bucket_id,
        "run_name": tracking.name,
        "seed": int(e.get("seed", 42)),
        # Rolling window stays small; BestCheckpointPoolCallback archives top-N.
        "save_total_limit": int(e.get("save_total_limit", 3)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    pool_keep = int(e.get("best_pool_keep", 15))
    if args.smoke or e.get("max_steps"):
        train_args["max_steps"] = int(e.get("max_steps", 20))
    else:
        train_args["num_train_epochs"] = float(e.get("num_train_epochs", 3))

    # Drop unsupported TrainingArguments keys for older/newer transformers.
    import inspect

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    train_args = {k: v for k, v in train_args.items() if k in accepted or k == "output_dir"}
    train_args = maybe_use_liger_training_arg(train_args, accepted)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**train_args),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            TrackioAlertCallback(tracking),
            BestCheckpointPoolCallback(out, keep=pool_keep, metric="eval_loss"),
        ],
    )
    resume = bool(args.resume and any(out.glob("checkpoint-*")))
    result = trainer.train(resume_from_checkpoint=resume)
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    journal = {
        "created_at": utc_now(),
        "run": cfg["run"],
        "base_id": cfg["base_id"],
        "model": str(model_id),
        "epochs": train_args.get("num_train_epochs"),
        "max_steps": train_args.get("max_steps"),
        "train_loss": result.metrics.get("train_loss"),
        "smoke": args.smoke,
        "kernels": getattr(model, "_qwen_kernel_meta", None),
        "trackio": tracking.metadata(),
    }
    atomic_json_write(out / "journal_cpt.json", journal)
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
