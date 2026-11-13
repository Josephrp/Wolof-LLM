#!/usr/bin/env python3
"""Optional, gated DPO training for validated Wolof preference pairs."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

import yaml

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
except ImportError:
    torch = None
    load_dataset = None
    LoraConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    DPOConfig = None
    DPOTrainer = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.tracking import TrackingRun, TrackioAlertCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train gated Wolof DPO adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_preferences(frame) -> None:
    required = {"prompt", "chosen", "rejected"}
    if not required <= set(frame.column_names):
        raise ValueError(f"preference dataset must contain {sorted(required)}")
    for index, row in enumerate(frame):
        if not all(isinstance(row[key], str) and row[key].strip() for key in required):
            raise ValueError(f"row {index} has empty/non-string preference fields")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"row {index} has identical chosen/rejected responses")


def run(args: argparse.Namespace) -> int:
    if DPOTrainer is None:
        raise RuntimeError("TRL DPO dependencies are missing")
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"model", "dataset", "output_dir", "training", "lora"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing DPO config keys: {missing}")
    dataset_path = Path(config["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    data = load_dataset("parquet", data_files=str(dataset_path))["train"]
    validate_preferences(data)
    if args.smoke:
        data = data.select(range(min(100, len(data))))
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM",
    )
    training = config["training"]
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    tracking = TrackingRun(
        size=str(config.get("size", "1.2b")),
        stage="dpo",
        tokenizer_family=str(config.get("tokenizer_family", "65k-ext")),
        method="lora",
        seed=int(training.get("seed", 42)),
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", "dpo"),
        config=config,
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        peft_config=peft_config,
        args=DPOConfig(
            output_dir=str(output),
            beta=float(training.get("beta", 0.1)),
            max_steps=10 if args.smoke else int(training["max_steps"]),
            per_device_train_batch_size=int(training.get("micro_bs", 1)),
            gradient_accumulation_steps=int(training.get("grad_acc", 8)),
            learning_rate=float(training.get("lr", 5e-7)),
            bf16=True,
            report_to="trackio" if not tracking.disabled else "none",
            trackio_space_id=tracking.space_id,
            trackio_bucket_id=tracking.bucket_id,
            run_name=tracking.name,
        ),
        callbacks=[TrackioAlertCallback(tracking)],
    )
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    (output / "journal_dpo.json").write_text(
        json.dumps({"config": str(path), "rows": len(data), "smoke": args.smoke}, indent=2) + "\n",
        encoding="utf-8",
    )
    tracking.log(
        {
            "train_loss_final": float(result.metrics.get("train_loss", float("nan"))),
            "preference_rows": len(data),
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
