#!/usr/bin/env python3
"""Continual pretrain Tiny Aya Base on ortho-filtered NTP (epoch-driven LoRA/QLoRA)."""

from __future__ import annotations

import argparse
import atexit
import inspect
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.checkpointing import make_eval_history_callback, step_train_args
from common.config import validate_clm_config
from common.kernels import attention_context, load_causal_lm, resolve_attn_implementation
from common.pipeline import atomic_json_write, utc_now
from common.tracking import TrackingRun, TrackioAlertCallback

HUB_PREFIXES = ("CohereLabs/", "Tonic/", "galsenai/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny Aya Wolof CPT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_local(path_or_id: str) -> str:
    if str(path_or_id).startswith(HUB_PREFIXES):
        return path_or_id
    p = Path(path_or_id)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
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
    model_id = resolve_local(cfg.get("model") or cfg["base_id"])
    tok_id = resolve_local(cfg.get("tokenizer") or model_id)

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
    attn = resolve_attn_implementation(cfg.get("attn_implementation"))
    model, kernel_meta = load_causal_lm(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        quantization_config=quant,
        attn_implementation=attn,
        apply_cce=cfg.get("fused_ce", True) is not False,
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
    lora_cfg = cfg.get("lora") or {}
    if cfg["methode"] in {"lora", "qlora"}:
        peft = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 32)),
            lora_alpha=int(lora_cfg.get("alpha", 64)),
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

    tracking = TrackingRun(
        size=str(cfg.get("size", "3b35")),
        stage=str(cfg.get("stage", "cpt")),
        tokenizer_family=str(cfg.get("tokenizer_family", "tinyaya-ext")),
        method=str(cfg.get("methode", "lora")),
        seed=int(e.get("seed", 42)),
        run_id=os.environ.get("TINYAYA_PIPELINE_RUN_ID", cfg["run"]),
        config=cfg,
        production=os.environ.get("TINYAYA_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=out / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    train_args = {
        "output_dir": str(out),
        "per_device_train_batch_size": int(e.get("micro_bs", 1)),
        "gradient_accumulation_steps": int(e.get("grad_acc", 8)),
        "learning_rate": float(e.get("lr", 2e-4)),
        "bf16": True,
        "gradient_checkpointing": bool(e.get("gradient_checkpointing", True)),
        "report_to": "trackio" if not tracking.disabled else "none",
        "seed": int(e.get("seed", 42)),
        **step_train_args(e, smoke=bool(args.smoke), default_eval=250),
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    for key, value in {
        "trackio_space_id": tracking.space_id,
        "trackio_bucket_id": tracking.bucket_id,
        "run_name": tracking.name,
    }.items():
        if key in params and value:
            train_args[key] = value
    if args.smoke or e.get("max_steps"):
        train_args["max_steps"] = int(e.get("max_steps", 20))
    else:
        train_args["num_train_epochs"] = float(e.get("num_train_epochs", 3))
    train_args = {k: v for k, v in train_args.items() if k in params}

    callbacks = [
        TrackioAlertCallback(tracking),
        make_eval_history_callback(out),
    ]
    # patience<=0 disables early stop so production can finish full epochs.
    patience = int(e.get("patience", 0))
    if patience > 0:
        callbacks.insert(0, EarlyStoppingCallback(early_stopping_patience=patience))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**train_args),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )
    resume = bool(args.resume and any(out.glob("checkpoint-*")))
    try:
        with attention_context(kernel_meta["attn_implementation"]):
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
            "kernels": kernel_meta,
            "trackio": tracking.metadata(),
        }
        atomic_json_write(out / "journal_cpt.json", journal)
        tracking.log({"train_loss": result.metrics.get("train_loss")}, step=None)
        tracking.finish(success=True)
        print(json.dumps(journal, indent=2), flush=True)
        return 0
    except Exception as exc:
        tracking.finish(success=False, error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
