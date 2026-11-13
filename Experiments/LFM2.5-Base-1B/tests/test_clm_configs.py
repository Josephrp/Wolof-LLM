"""Static schema checks for CLM recipes and the family manifest."""

import importlib.util
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"run", "size", "base_id", "methode", "entrainement"}


def test_train_group_includes_runtime_plot_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "matplotlib" in project["dependency-groups"]["train"]


def test_all_clm_configs_have_training_schema():
    paths = list((ROOT / "configs" / "clm").rglob("*.yaml"))
    assert paths
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert not REQUIRED - config.keys(), path
        training = config["entrainement"]
        assert {"max_steps", "seq_len", "micro_bs", "grad_acc", "lr", "seed"} <= (
            training.keys()
        ), path
        if "ext" in path.name and config["methode"] == "lora":
            assert config["lora"].get("modules_to_save"), path


def test_family_manifest_references_existing_configs():
    manifest_path = ROOT / "configs" / "family_runs.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["sizes"].values():
        for config in entry["configs"]:
            assert (ROOT / config).is_file(), config


def test_350m_lora_targets_match_validated_probe():
    expected = ["q_proj", "k_proj", "v_proj"]
    paths = [
        ROOT / "configs/scaling/smoke_350m_default_lora.yaml",
        ROOT / "configs/scaling/smoke_350m_ext_lora.yaml",
        ROOT / "configs/clm/350m/v2_ext4000_lora.yaml",
    ]
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["lora"]["target_modules"] == expected, path


def test_1_2b_lora_targets_match_validated_probe():
    path = ROOT / "configs/clm/1.2b/v2_ext4000_lora.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj"]


def test_dataloader_training_args_preserve_safe_worker_options():
    path = ROOT / "scripts/clm/03_train.py"
    spec = importlib.util.spec_from_file_location("clm_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    accepted = {
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
    }
    assert module._dataloader_training_args(
        {
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 2,
        },
        accepted,
    ) == {
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 2,
    }


def test_remaining_1_2b_ext_runs_enable_dataloader_workers():
    for name in ("v2_ext4000_lora.yaml", "v2_ext4000_full.yaml"):
        config = yaml.safe_load(
            (ROOT / "configs/clm/1.2b" / name).read_text(encoding="utf-8")
        )
        training = config["entrainement"]
        assert training["dataloader_num_workers"] == 4
        assert training["dataloader_pin_memory"] is True
        assert training["dataloader_persistent_workers"] is True
        assert training["dataloader_prefetch_factor"] == 2
