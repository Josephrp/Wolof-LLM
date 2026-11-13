"""Regular eval + checkpoint cadence for production Tiny Aya runs.

Ensures Trainer/SFT/DPO:
- eval and save on the same step grid (required for load_best_model_at_end)
- keep enough mid-run checkpoints for resume after crashes
- append each eval to a durable local JSONL (survives Trackio blips)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common.pipeline import utc_now


def production_mode() -> bool:
    return os.environ.get("TINYAYA_PRODUCTION") == "1"


def resolve_cadence(
    training: dict[str, Any],
    *,
    smoke: bool,
    default_eval: int = 250,
    default_save_limit: int = 3,
) -> dict[str, int]:
    """Pick eval/save/logging cadence. Prefer sparse evals on long production runs."""
    eval_steps = int(training.get("eval_steps", default_eval))
    save_steps = int(training.get("save_steps", eval_steps))
    # Keep eval and save aligned so load_best_model_at_end is valid.
    if save_steps != eval_steps:
        save_steps = eval_steps
    logging_steps = int(training.get("logging_steps", max(1, min(25, eval_steps // 10))))
    if smoke:
        save_total_limit = int(training.get("save_total_limit", 1))
    else:
        save_total_limit = int(training.get("save_total_limit", default_save_limit))
        save_total_limit = max(save_total_limit, 2)
    return {
        "eval_steps": max(1, eval_steps),
        "save_steps": max(1, save_steps),
        "logging_steps": max(1, logging_steps),
        "save_total_limit": save_total_limit,
    }


def step_train_args(
    training: dict[str, Any],
    *,
    smoke: bool,
    default_eval: int = 250,
    with_best_model: bool = True,
) -> dict[str, Any]:
    """Common HF Trainer kwargs for regular step-based eval/checkpointing."""
    cadence = resolve_cadence(training, smoke=smoke, default_eval=default_eval)
    args: dict[str, Any] = {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "eval_steps": cadence["eval_steps"],
        "save_steps": cadence["save_steps"],
        "logging_steps": cadence["logging_steps"],
        "save_total_limit": cadence["save_total_limit"],
    }
    if with_best_model:
        args.update(
            {
                "load_best_model_at_end": True,
                "metric_for_best_model": str(
                    training.get("metric_for_best_model", "eval_loss")
                ),
                "greater_is_better": bool(training.get("greater_is_better", False)),
            }
        )
    return args


def make_eval_history_callback(output_dir: Path | str):
    """TrainerCallback: append evals to eval_history.jsonl; write latest_checkpoint.json."""
    from transformers import TrainerCallback

    out = Path(output_dir)
    history = out / "eval_history.jsonl"
    pointer = out / "latest_checkpoint.json"

    class EvalHistoryCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            out.mkdir(parents=True, exist_ok=True)
            row = {
                "created_at": utc_now(),
                "global_step": int(getattr(state, "global_step", 0) or 0),
                "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                "metrics": {
                    k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()
                },
            }
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return control

        def on_save(self, args, state, control, **kwargs):
            ckpt = out / f"checkpoint-{int(state.global_step)}"
            payload = {
                "created_at": utc_now(),
                "global_step": int(state.global_step),
                "checkpoint": str(ckpt) if ckpt.is_dir() else None,
                "best_metric": getattr(state, "best_metric", None),
                "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
            }
            out.mkdir(parents=True, exist_ok=True)
            pointer.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return control

    return EvalHistoryCallback()
