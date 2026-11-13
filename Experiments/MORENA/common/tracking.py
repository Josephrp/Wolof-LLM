"""Secret-safe Trackio integration for the MORENA Wolof continuum.

Gradio dashboard is the Trackio Space (`TRACKIO_SPACE_ID`). Live runs persist to
the private Bucket (`TRACKIO_BUCKET_ID`).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Self

try:
    import trackio
except ImportError:
    trackio = None

PROJECT = "morena-wolof"
SENSITIVE_KEY = re.compile(r"(?i)(token|password|passwd|secret|api.?key|credential|auth)")
TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\[^\\\s]+|/mnt/(?:volume|data)[^/\s]*)[^\s\"']*")


def load_trackio_env(root: Path | None = None) -> None:
    """Load Experiments/MORENA/.env.trackio into os.environ (setdefault)."""
    base = root or Path(__file__).resolve().parents[1]
    env_path = base / ".env.trackio"
    if not env_path.is_file():
        # Fall back to sibling LFM Trackio Space so Gradio is reachable immediately.
        sibling = base.parent / "LFM2.5-Base-1B" / ".env.trackio"
        env_path = sibling if sibling.is_file() else env_path
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def sanitize_config(value: Any, key: str = "") -> Any:
    if SENSITIVE_KEY.search(key):
        return "<REDACTED>"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_config(item_value, str(item_key))
            for item_key, item_value in value.items()
            if not SENSITIVE_KEY.search(str(item_key))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_config(item, key) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        value = TOKEN.sub("<REDACTED_TOKEN>", value)
        value = IP.sub("<REDACTED_IP>", value)
        value = PRIVATE_PATH.sub("<REDACTED_PRIVATE_PATH>", value)
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def deterministic_run_name(
    *,
    size: str,
    stage: str,
    tokenizer_family: str,
    method: str,
    seed: int,
    run_id: str,
) -> str:
    components = (size, stage, tokenizer_family, method, f"seed{seed}", run_id)
    safe = [re.sub(r"[^A-Za-z0-9_.-]+", "-", str(item)).strip("-") for item in components]
    if any(not item for item in safe):
        raise ValueError("tracking run-name components must be non-empty")
    return "__".join(safe).lower()


def classify_alert(metrics: dict[str, Any], step: int = 0) -> tuple[str, str] | None:
    for key in ("loss", "train_loss", "eval_loss", "bpb", "BPB"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return "error", f"non-finite {key} at step {step}"
    loss = metrics.get("loss", metrics.get("train_loss"))
    if isinstance(loss, (int, float)) and step >= 50 and float(loss) > 20:
        return "error", f"loss divergence at step {step}"
    return None


class TrackingRun:
    """Best-effort Trackio logging for MORENA native trainer / distill / eval."""

    def __init__(
        self,
        *,
        size: str,
        stage: str,
        tokenizer_family: str = "65k",
        method: str = "muon-fsdp2",
        seed: int = 1234,
        run_id: str,
        config: dict[str, Any] | None = None,
        production: bool = False,
        disabled: bool = False,
        state_path: Path | None = None,
        project: str | None = None,
    ):
        self.project = project or os.environ.get("TRACKIO_PROJECT", PROJECT)
        self.group = f"{size}/{stage}"
        self.name = deterministic_run_name(
            size=size,
            stage=stage,
            tokenizer_family=tokenizer_family,
            method=method,
            seed=seed,
            run_id=run_id,
        )
        self.space_id = os.environ.get("TRACKIO_SPACE_ID")
        self.bucket_id = os.environ.get("TRACKIO_BUCKET_ID")
        self.dataset_id = os.environ.get("TRACKIO_DATASET_ID")
        self.config = sanitize_config(config or {})
        self.production = production
        self.disabled = disabled or os.environ.get("TRACKIO_DISABLED") == "1"
        self.state_path = state_path
        self.status = "disabled" if self.disabled else "pending"
        self.error: str | None = None
        self.remote_run_id: str | None = None

    def metadata(self) -> dict[str, Any]:
        dashboard = (
            f"https://huggingface.co/spaces/{self.space_id}" if self.space_id else None
        )
        return {
            "project": self.project,
            "group": self.group,
            "run_name": self.name,
            "run_id": self.remote_run_id or self.name,
            "space_id": self.space_id,
            "bucket_id": self.bucket_id,
            "dataset_id": self.dataset_id,
            "sync_status": self.status,
            "dashboard_url": dashboard,
            "error": self.error,
        }

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def start(self) -> Self:
        if self.disabled:
            self._persist()
            return self
        if self.production and not self.space_id:
            raise RuntimeError("production tracking requires TRACKIO_SPACE_ID")
        if self.production and not self.bucket_id:
            raise RuntimeError("production tracking requires TRACKIO_BUCKET_ID")
        if trackio is None:
            raise RuntimeError("trackio is not installed")
        try:
            init_kwargs: dict[str, Any] = {
                "project": self.project,
                "name": self.name,
                "group": self.group,
                "config": self.config,
                "space_id": self.space_id,
            }
            if self.bucket_id:
                init_kwargs["bucket_id"] = self.bucket_id
            elif self.dataset_id:
                init_kwargs["dataset_id"] = self.dataset_id
            run = trackio.init(**init_kwargs)
            returned_id = getattr(run, "id", None)
            if returned_id:
                self.remote_run_id = str(returned_id)
            self.status = "active"
            self._persist()
            return self
        except Exception as exc:
            self.status = "initialization_failed"
            self.error = sanitize_config(str(exc))
            self._persist()
            raise RuntimeError("Trackio initialization failed") from exc

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.disabled or self.status not in {"active", "degraded"}:
            return
        safe_metrics = sanitize_config(metrics)
        try:
            trackio.log(safe_metrics, step=step)
            alert = classify_alert(safe_metrics, step or 0)
            if alert:
                self.alert("Training diagnostic", alert[1], level=alert[0])
        except Exception as exc:  # noqa: BLE001
            self.status = "degraded"
            self.error = sanitize_config(str(exc))
            self._persist()

    def alert(self, title: str, text: str, level: str = "error") -> None:
        if self.disabled or trackio is None:
            return
        levels = {
            "info": trackio.AlertLevel.INFO,
            "warn": trackio.AlertLevel.WARN,
            "error": trackio.AlertLevel.ERROR,
        }
        try:
            trackio.alert(
                title=sanitize_config(title),
                text=sanitize_config(text),
                level=levels[level],
            )
        except Exception as exc:  # noqa: BLE001
            self.status = "degraded"
            self.error = sanitize_config(str(exc))
            self._persist()

    def finish(self, *, success: bool, error: BaseException | None = None) -> None:
        if self.disabled or self.status in {"synced", "failed", "initialization_failed"}:
            return
        if error is not None:
            category = "oom" if "out of memory" in str(error).lower() else "stage failure"
            self.alert(category.upper(), str(error), level="error")
        try:
            trackio.finish()
            self.status = "synced" if success else "failed"
        except Exception as exc:  # noqa: BLE001
            already_finished = (
                self.remote_run_id is not None
                and isinstance(exc, RuntimeError)
                and str(exc) == "Call trackio.init() before trackio.finish()."
            )
            if already_finished:
                self.status = "synced" if success else "failed"
                self.error = None
            else:
                self.status = "degraded"
                self.error = sanitize_config(str(exc))
        self._persist()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback_value) -> bool:
        self.finish(success=exc_value is None, error=exc_value)
        return False
