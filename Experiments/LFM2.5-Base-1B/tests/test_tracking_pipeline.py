"""Offline tests for Trackio safety and durable production orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import common.tracking as tracking_module
from common.pipeline import PipelineRunner, atomic_json_write
from common.publication import scan_for_secrets
from common.tracking import (
    TrackingRun,
    classify_alert,
    deterministic_run_name,
    sanitize_config,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTrackio:
    class AlertLevel:
        INFO = "info"
        WARN = "warn"
        ERROR = "error"

    def __init__(self):
        self.inits = []
        self.logs = []
        self.alerts = []
        self.finished = 0

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return SimpleNamespace(id="fake-run-id")

    def log(self, metrics, step=None):
        self.logs.append((metrics, step))

    def alert(self, **kwargs):
        self.alerts.append(kwargs)

    def finish(self):
        self.finished += 1


def test_tracking_config_sanitization_and_name():
    sanitized = sanitize_config(
        {
            "token": "hf_" + "A" * 30,
            "endpoint": "192.0.2.10",
            "path": "/mnt/volume_private/project/model",
            "nested": {"password": "not-safe"},
        }
    )
    payload = json.dumps(sanitized)
    assert "hf_" not in payload
    assert "192.0.2" not in payload
    assert "/mnt/volume" not in payload
    assert "not-safe" not in payload
    first = deterministic_run_name(
        size="1.2b",
        stage="cpt",
        tokenizer_family="65k-ext",
        method="lora",
        seed=42,
        run_id="production-v1",
    )
    second = deterministic_run_name(
        size="1.2b",
        stage="cpt",
        tokenizer_family="65k-ext",
        method="lora",
        seed=42,
        run_id="production-v1",
    )
    assert first == second
    assert {"1.2b", "cpt", "65k-ext", "lora", "seed42"} <= set(first.split("__"))


def test_disabled_tracking_is_network_free(tmp_path: Path, monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking_module, "trackio", fake)
    run = TrackingRun(
        size="test",
        stage="unit",
        tokenizer_family="none",
        method="disabled",
        seed=42,
        run_id="test",
        disabled=True,
        state_path=tmp_path / "trackio.json",
    ).start()
    run.log({"loss": 1.0})
    run.finish(success=True)
    assert not fake.inits
    assert json.loads((tmp_path / "trackio.json").read_text())["sync_status"] == "disabled"


def test_finish_is_idempotent_after_transformers_finishes_run(
    tmp_path: Path,
    monkeypatch,
):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking_module, "trackio", fake)
    run = TrackingRun(
        size="350m",
        stage="v1",
        tokenizer_family="65k",
        method="lora",
        seed=42,
        run_id="production",
        state_path=tmp_path / "trackio.json",
    ).start()

    def already_finished():
        raise RuntimeError("Call trackio.init() before trackio.finish().")

    monkeypatch.setattr(fake, "finish", already_finished)
    run.finish(success=True)

    metadata = json.loads((tmp_path / "trackio.json").read_text())
    assert metadata["sync_status"] == "synced"
    assert metadata["error"] is None


def test_production_tracking_requires_space(monkeypatch):
    monkeypatch.delenv("TRACKIO_SPACE_ID", raising=False)
    with pytest.raises(RuntimeError, match="TRACKIO_SPACE_ID"):
        TrackingRun(
            size="1.2b",
            stage="cpt",
            tokenizer_family="65k-ext",
            method="lora",
            seed=42,
            run_id="production",
            production=True,
        ).start()


def test_trackio_space_gate_allows_paused_dashboard():
    module = load_script("scripts/publication/44_setup_trackio_space.py")
    assert (
        module.resolve_space_id(
            "owner",
            "new-space",
            "owner/preflight-space",
        )
        == "owner/preflight-space"
    )
    with pytest.raises(ValueError, match="namespace/name"):
        module.resolve_space_id("owner", "new-space", "invalid")

    class PausedApi:
        @staticmethod
        def get_space_runtime(_space_id):
            return SimpleNamespace(stage="PAUSED")

    status = module.inspect_dashboard(
        PausedApi(),
        "owner/space",
        "paused_quota",
    )
    assert status == {
        "stage": "PAUSED",
        "dashboard_status": "paused_quota",
    }


def test_remote_bucket_persistence_is_required(tmp_path: Path, monkeypatch):
    module = load_script("scripts/publication/44_setup_trackio_space.py")
    run_name = "smoke-run"
    run_id = "run-id"
    record = {
        "v": 1,
        "kind": "metric",
        "project": "lfm25-wolof",
        "run": run_name,
        "run_id": run_id,
        "metrics": {"sync_smoke": 1, "gate_status": 1},
        "step": 0,
        "log_id": "log-id",
    }
    item = SimpleNamespace(
        type="file",
        path="trackio/inbox/writer/00000000.jsonl",
        xet_hash="remote-revision",
    )
    monkeypatch.setattr(
        module,
        "bucket_info",
        lambda *_args, **_kwargs: SimpleNamespace(
            private=True,
            size=100,
            total_files=1,
        ),
    )
    monkeypatch.setattr(module, "list_bucket_tree", lambda *_args, **_kwargs: [item])

    def download(_bucket_id, files, **_kwargs):
        files[0][1].write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(module, "download_bucket_files", download)
    result = module.verify_bucket_persistence(
        bucket_id="owner/bucket",
        project="lfm25-wolof",
        run_name=run_name,
        run_id=run_id,
        token="redacted",
        timeout=0,
    )
    assert result["status"] == "verified"
    assert result["fragment_revision"] == "remote-revision"


def test_tracking_alert_thresholds():
    assert classify_alert({"loss": float("nan")})[0] == "error"
    assert classify_alert({"eval_loss": float("inf")})[0] == "error"
    assert classify_alert({"loss": 11.0}, step=100)[0] == "error"
    assert classify_alert({"loss": 1.0}, step=100) is None


def test_mocked_trackio_metadata_and_alert(tmp_path: Path, monkeypatch):
    fake = FakeTrackio()
    monkeypatch.setattr(tracking_module, "trackio", fake)
    monkeypatch.setenv("TRACKIO_SPACE_ID", "owner/private-space")
    monkeypatch.setenv("TRACKIO_BUCKET_ID", "owner/private-bucket")
    run = TrackingRun(
        size="350m",
        stage="smoke",
        tokenizer_family="65k-ext",
        method="lora",
        seed=42,
        run_id="v1",
        production=True,
        state_path=tmp_path / "trackio.json",
    ).start()
    run.log({"loss": float("nan")}, step=2)
    run.finish(success=False)
    metadata = json.loads((tmp_path / "trackio.json").read_text())
    assert metadata["project"] == "lfm25-wolof"
    assert metadata["space_id"] == "owner/private-space"
    assert metadata["bucket_id"] == "owner/private-bucket"
    assert metadata["run_id"] == "fake-run-id"
    assert fake.alerts


def test_secret_scanner_rejects_trackio_config(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"token": "hf_" + "Z" * 30}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="secret"):
        scan_for_secrets(tmp_path)


def test_pipeline_dry_run_and_resume_state(tmp_path: Path):
    manifest_path = tmp_path / "pipeline.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "unit",
                "stages": [
                    {
                        "name": "one",
                        "size": "common",
                        "tracking": False,
                        "command": ["python", "--version"],
                        "outputs": [],
                        "resources": {"min_disk_gb": 0},
                    },
                    {
                        "name": "two",
                        "size": "350m",
                        "tracking": False,
                        "requires": ["one"],
                        "command": ["python", "--version"],
                        "outputs": [],
                        "resources": {"min_disk_gb": 0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "state.json"
    runner = PipelineRunner(
        root=tmp_path,
        manifest_path=manifest_path,
        state_path=state_path,
        logs_dir=tmp_path / "logs",
        resume=False,
        dry_run=True,
    )
    assert runner.run("350m") == 0
    state = json.loads(state_path.read_text())
    assert state["status"] == "dry_run_complete"
    assert set(state["stages"]) == {"one", "two"}
    atomic_json_write(tmp_path / "atomic.json", {"ok": True})
    assert json.loads((tmp_path / "atomic.json").read_text()) == {"ok": True}
