"""Tests for private checkpoint and tokenizer asset discovery."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/publication/45_publish_training_assets.py"


def _module():
    spec = importlib.util.spec_from_file_location("publish_training_assets", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_discover_checkpoints_requires_resumable_state(tmp_path):
    valid = tmp_path / "1.2b" / "run" / "checkpoint-100"
    invalid = tmp_path / "1.2b" / "run" / "checkpoint-200"
    valid.mkdir(parents=True)
    invalid.mkdir()
    (valid / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    assert _module().discover_checkpoints(tmp_path) == [valid]


def test_discover_tokenizers_requires_config(tmp_path):
    valid = tmp_path / "ext4000"
    invalid = tmp_path / "unfinished"
    valid.mkdir()
    invalid.mkdir()
    (valid / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    assert _module().discover_tokenizers(tmp_path) == [valid]


def test_asset_repo_slug_is_stable():
    assert _module()._slug("lfm25_wolof/ext 4000") == "lfm25-wolof-ext-4000"
