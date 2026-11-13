#!/usr/bin/env python3
"""Optional DPO on validated Wolof preference pairs (after best Instruct SFT)."""

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
from common.checkpointing import make_eval_history_callback, resolve_cadence
from common.kernels import attention_context, load_causal_lm, resolve_attn_implementation
from common.pipeline import atomic_json_write, utc_now
from common.tracking import TrackingRun, TrackioAlertCallback

HUB_PREFIXES = ("CohereLabs/", "Tonic/", "galsenai/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Tiny Aya Wolof DPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--model",
        default=None,
        help="Override config model path (use pick_best merged Instruct weights)",
    )
    return parser


def resolve_model_dir(path_or_id: str) -> str:
    """Resolve Hub id, local weights dir, or best_pointer.json stub folder."""
    if str(path_or_id).startswith(HUB_PREFIXES):
        return path_or_id
    p = Path(path_or_id)
    if not p.is_absolute():
        p = ROOT / p
    pointer = p / "best_pointer.json"
    if pointer.is_file() and not (p / "config.json").is_file():
        best = json.loads(pointer.read_text(encoding="utf-8")).get("best") or {}
        target = best.get("path")
        if not target:
            raise SystemExit(f"best_pointer.json missing path under {p}")
        return resolve_model_dir(str(target))
    return str(p)


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
    model_path = resolve_model_dir(args.model or cfg["model"])
    dataset_path = Path(cfg["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    training = dict(cfg["training"])
    ds = load_dataset("parquet", data_files=str(dataset_path), split="train")
    if args.smoke:
        ds = ds.select(range(min(64, len(ds))))
    validate_preferences(ds)
    # Hold out a small eval slice so DPO evaluates + checkpoints on a step grid.
    default_eval = max(8, min(256, int(len(ds) * 0.05))) if len(ds) > 32 else max(1, len(ds) // 8)
    n_eval = int(training.get("eval_size", default_eval))
    n_eval = max(1, min(n_eval, max(1, len(ds) // 4)))
    split = ds.train_test_split(test_size=n_eval, seed=int(training.get("seed", 42)))
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn = resolve_attn_implementation(cfg.get("attn_implementation"))
    model, kernel_meta = load_causal_lm(
        model_path,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=attn,
        apply_cce=cfg.get("fused_ce", True) is not False,
    )
    lora = cfg["lora"]
    peft = LoraConfig(
        r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type="CAUSAL_LM",
        bias="none",
    )
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        out = ROOT / out
    if args.smoke:
        out = ROOT / "results" / "smoke" / "dpo"
        training["max_steps"] = 10
        training.setdefault("eval_steps", 5)
        training.setdefault("save_steps", 5)
    out.mkdir(parents=True, exist_ok=True)

    tracking = TrackingRun(
        size=str(cfg.get("size", "3b35")),
        stage="dpo",
        tokenizer_family=str(cfg.get("tokenizer_family", "tinyaya-ext")),
        method=str(cfg.get("methode", "lora")),
        seed=int(training.get("seed", 42)),
        run_id=os.environ.get("TINYAYA_PIPELINE_RUN_ID", cfg.get("run", "dpo")),
        config=cfg,
        production=os.environ.get("TINYAYA_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=out / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    cadence = resolve_cadence(training, smoke=bool(args.smoke), default_eval=200)
    dpo_kwargs = {
        "output_dir": str(out),
        "per_device_train_batch_size": int(training.get("micro_bs", 1)),
        "gradient_accumulation_steps": int(training.get("grad_acc", 8)),
        "learning_rate": float(training.get("lr", 5e-7)),
        "beta": float(training.get("beta", 0.1)),
        "bf16": True,
        "report_to": "trackio" if not tracking.disabled else "none",
        "seed": int(training.get("seed", 42)),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "eval_steps": cadence["eval_steps"],
        "save_steps": cadence["save_steps"],
        "logging_steps": cadence["logging_steps"],
        "save_total_limit": cadence["save_total_limit"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    # Smoke / explicit max_steps only; production defaults to full-split epochs.
    if args.smoke or training.get("max_steps"):
        dpo_kwargs["max_steps"] = int(training.get("max_steps", 10 if args.smoke else 1000))
    else:
        dpo_kwargs["num_train_epochs"] = float(training.get("num_train_epochs", 3))
    params = inspect.signature(DPOConfig.__init__).parameters
    for key, value in {
        "trackio_space_id": tracking.space_id,
        "trackio_bucket_id": tracking.bucket_id,
        "run_name": tracking.name,
        "save_safetensors": True,
    }.items():
        if key in params and value:
            dpo_kwargs[key] = value
    # Drop any keys this TRL DPOConfig build does not accept.
    dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in params or k == "output_dir"}

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**dpo_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft,
        callbacks=[TrackioAlertCallback(tracking), make_eval_history_callback(out)],
    )
    try:
        with attention_context(kernel_meta["attn_implementation"]):
            result = trainer.train()
        trainer.save_model(str(out / "adapter"))
        tokenizer.save_pretrained(str(out / "adapter"))
        journal = {
            "created_at": utc_now(),
            "model": model_path,
            "dataset": str(dataset_path),
            "epochs": dpo_kwargs.get("num_train_epochs"),
            "max_steps": dpo_kwargs.get("max_steps"),
            "n_train": len(train_ds),
            "n_eval": len(eval_ds),
            "train_loss": result.metrics.get("train_loss"),
            "smoke": args.smoke,
            "kernels": kernel_meta,
            "trackio": tracking.metadata(),
        }
        atomic_json_write(out / "journal_dpo.json", journal)
        tracking.log({"train_loss": result.metrics.get("train_loss")}, step=None)
        tracking.finish(success=True)
        print(json.dumps(journal, indent=2), flush=True)
        return 0
    except Exception as exc:
        tracking.finish(success=False, error=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
