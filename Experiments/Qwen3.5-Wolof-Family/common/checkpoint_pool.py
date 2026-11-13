"""Retain top-N training checkpoints by eval_loss (hall-of-fame pool).

HuggingFace ``save_total_limit`` only keeps a short rolling window. This callback
copies strong checkpoints into ``{output_dir}/best_pool/`` ranked by eval_loss
(lower is better) so golden-eval can later pick among a small elite set.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from common.pipeline import atomic_json_write, utc_now


def pool_dir(output_dir: Path | str) -> Path:
    return Path(output_dir) / "best_pool"


def pool_index_path(output_dir: Path | str) -> Path:
    return pool_dir(output_dir) / "index.json"


def load_pool_index(output_dir: Path | str) -> dict[str, Any]:
    path = pool_index_path(output_dir)
    if not path.is_file():
        return {"updated_at": None, "metric": "eval_loss", "greater_is_better": False, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_pool_index(output_dir: Path | str, index: dict[str, Any]) -> None:
    index["updated_at"] = utc_now()
    atomic_json_write(pool_index_path(output_dir), index)


def _entry_sort_key(entry: dict[str, Any], *, greater_is_better: bool) -> float:
    score = float(entry["score"])
    return -score if greater_is_better else score


def prune_pool(output_dir: Path | str, *, keep: int = 15) -> dict[str, Any]:
    """Keep only the top-``keep`` pool entries by score; delete the rest on disk."""
    out = Path(output_dir)
    index = load_pool_index(out)
    greater = bool(index.get("greater_is_better", False))
    entries = sorted(
        list(index.get("entries") or []),
        key=lambda e: _entry_sort_key(e, greater_is_better=greater),
    )
    keep_entries = entries[: max(1, int(keep))]
    keep_names = {e["name"] for e in keep_entries}
    root = pool_dir(out)
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name not in keep_names:
                shutil.rmtree(child, ignore_errors=True)
    index["entries"] = keep_entries
    save_pool_index(out, index)
    return index


# Resume-only blobs are not needed for golden eval / merge from elite copies.
_POOL_STRIP_NAMES = frozenset(
    {"optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"}
)


def _strip_resume_blobs(checkpoint_dir: Path) -> None:
    for name in _POOL_STRIP_NAMES:
        path = checkpoint_dir / name
        if path.is_file():
            path.unlink(missing_ok=True)


def promote_checkpoint(
    checkpoint: Path,
    output_dir: Path | str,
    *,
    score: float,
    step: int,
    keep: int = 15,
    metric: str = "eval_loss",
    greater_is_better: bool = False,
) -> dict[str, Any]:
    """Copy ``checkpoint`` into the pool and prune to top-``keep``."""
    out = Path(output_dir)
    root = pool_dir(out)
    root.mkdir(parents=True, exist_ok=True)
    name = checkpoint.name
    dest = root / name
    if dest.resolve() != checkpoint.resolve():
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(checkpoint, dest, dirs_exist_ok=False)
        _strip_resume_blobs(dest)
    index = load_pool_index(out)
    index["metric"] = metric
    index["greater_is_better"] = greater_is_better
    entries = [e for e in (index.get("entries") or []) if e.get("name") != name]
    entries.append(
        {
            "name": name,
            "step": int(step),
            "score": float(score),
            "path": str(dest),
            "promoted_at": utc_now(),
        }
    )
    index["entries"] = entries
    save_pool_index(out, index)
    return prune_pool(out, keep=keep)


class BestCheckpointPoolCallback(TrainerCallback):
    """After each save, promote into best_pool when eval_loss is competitive."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        keep: int = 15,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.keep = int(keep)
        self.metric = metric
        self.greater_is_better = greater_is_better

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        step = int(state.global_step)
        ckpt = self.output_dir / f"checkpoint-{step}"
        if not ckpt.is_dir():
            return control
        score = self._latest_metric(state)
        if score is None:
            return control
        index = load_pool_index(self.output_dir)
        entries = list(index.get("entries") or [])
        if len(entries) >= self.keep:
            worst = max(
                entries,
                key=lambda e: _entry_sort_key(e, greater_is_better=self.greater_is_better),
            )
            worst_score = float(worst["score"])
            better = score > worst_score if self.greater_is_better else score < worst_score
            already = any(e.get("step") == step for e in entries)
            if not better and not already:
                return control
        promote_checkpoint(
            ckpt,
            self.output_dir,
            score=score,
            step=step,
            keep=self.keep,
            metric=self.metric,
            greater_is_better=self.greater_is_better,
        )
        print(
            f"BEST_POOL promote checkpoint-{step} {self.metric}={score:.6f} keep={self.keep}",
            flush=True,
        )
        return control

    def _latest_metric(self, state: TrainerState) -> float | None:
        for log in reversed(list(state.log_history or [])):
            if self.metric in log and log[self.metric] is not None:
                try:
                    return float(log[self.metric])
                except (TypeError, ValueError):
                    return None
        return None
