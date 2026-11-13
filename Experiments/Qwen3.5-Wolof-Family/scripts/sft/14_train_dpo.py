#!/usr/bin/env python3
"""Optional DPO on validated Wolof preference pairs (LFM pattern)."""

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
from common.kernels import load_causal_lm
from common.pipeline import atomic_json_write, utc_now
from common.tracking import TrackingRun, TrackioAlertCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 Wolof DPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_preferences(ds) -> None:
    required = {"prompt", "chosen", "rejected"}
    if not required <= set(ds.column_names):
        raise ValueError(f"prefs missing columns {required}")
    for row in ds.select(range(min(20, len(ds)))):
        if not row["prompt"] or not row["chosen"] or not row["rejected"]:
            raise ValueError("empty preference field")
        if row["chosen"] == row["rejected"]:
            raise ValueError("chosen == rejected")


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("model", "dataset", "output_dir", "lora", "training"):
        if key not in cfg:
            raise SystemExit(f"missing DPO config key: {key}")

    token = os.environ.get("HF_TOKEN")
    model_path = cfg["model"]
    if not str(model_path).startswith(("Qwen/", "Tonic/")):
        model_path = str(ROOT / model_path)
    dataset_path = Path(cfg["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    ds = load_dataset("parquet", data_files=str(dataset_path), split="train")
    if args.smoke:
        ds = ds.select(range(min(64, len(ds))))
    validate_preferences(ds)

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm(
        model_path,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=cfg.get("attn_implementation"),
    )
    lora = cfg["lora"]
    peft = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type="CAUSAL_LM",
        bias="none",
    )
    training = cfg["training"]
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        out = ROOT / out
    if args.smoke:
        out = ROOT / "results" / "smoke" / "dpo"
        training = dict(training)
        training["max_steps"] = 10
    out.mkdir(parents=True, exist_ok=True)

    tracking = TrackingRun(
        size=str(cfg.get("size", "08b")),
        stage="dpo",
        tokenizer_family="qwen35-ext",
        method="lora",
        seed=int(training.get("seed", 42)),
        run_id=os.environ.get("QWEN_PIPELINE_RUN_ID", "dpo"),
        config=cfg,
        production=os.environ.get("QWEN_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=out / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    dpo_kwargs = {
        "output_dir": str(out),
        "per_device_train_batch_size": int(training.get("micro_bs", 1)),
        "gradient_accumulation_steps": int(training.get("grad_acc", 8)),
        "learning_rate": float(training.get("lr", 5e-7)),
        "max_steps": int(training.get("max_steps", 1000)),
        "beta": float(training.get("beta", 0.1)),
        "bf16": True,
        "logging_steps": 10,
        "report_to": "trackio" if not tracking.disabled else "none",
        "trackio_space_id": tracking.space_id,
        "trackio_bucket_id": tracking.bucket_id,
        "run_name": tracking.name,
        "seed": int(training.get("seed", 42)),
    }
    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in accepted or k == "output_dir"}

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**dpo_kwargs),
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft,
        callbacks=[TrackioAlertCallback(tracking)],
    )
    result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    journal = {
        "created_at": utc_now(),
        "model": model_path,
        "dataset": str(dataset_path),
        "train_loss": result.metrics.get("train_loss"),
        "smoke": args.smoke,
        "trackio": tracking.metadata(),
    }
    atomic_json_write(out / "journal_dpo.json", journal)
    tracking.log(
        {
            "train_loss_final": float(result.metrics.get("train_loss", float("nan"))),
            "preference_rows": len(ds),
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    print(json.dumps(journal, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
