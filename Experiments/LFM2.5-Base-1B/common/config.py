"""Fail-fast configuration validation for LFM2.5 Wolof pipelines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

FAMILY_SPECIALS = {
    "65k": {"pad": 0, "bos": 1, "eos": 7},
    "65k-ext": {"pad": 0, "bos": 1, "eos": 7},
    "128k": {"pad": 124893, "bos": 124894, "eos": 124900},
    # insert-before-specials (+4000) for Wolof ext4000 on the 128k family
    "128k-ext": {"pad": 128893, "bos": 128894, "eos": 128900},
}
ALLOWED_METHODS = {"lora", "full"}
SAFE_CPT_DATASET = "galsenai/cleaned_data_for_clm"
# Curated subsets of the cleaned CLM source (same cleaning lineage).
SAFE_CPT_DATASETS = frozenset(
    {
        SAFE_CPT_DATASET,
        "Tonic/LFM2.5-Wolof-CPT-ortho_ns_mixed_v1",
    }
)


class ConfigError(ValueError):
    """Raised when a recipe violates a safety invariant."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: root must be a mapping")
    return data


def validate_clm_config(config: dict[str, Any], path: str | Path = "<config>") -> dict[str, Any]:
    label = str(path)
    required = {"run", "size", "base_id", "methode", "entrainement"}
    missing = sorted(required - config.keys())
    if missing:
        raise ConfigError(f"{label}: missing keys: {', '.join(missing)}")
    method = config["methode"]
    if method not in ALLOWED_METHODS:
        raise ConfigError(f"{label}: methode must be one of {sorted(ALLOWED_METHODS)}")
    training = config["entrainement"]
    if not isinstance(training, dict):
        raise ConfigError(f"{label}: entrainement must be a mapping")
    train_required = {"max_steps", "seq_len", "micro_bs", "grad_acc", "lr", "seed"}
    train_missing = sorted(train_required - training.keys())
    if train_missing:
        raise ConfigError(f"{label}: entrainement missing: {', '.join(train_missing)}")

    family = str(config.get("tokenizer_family") or config.get("family") or "65k")
    if family not in FAMILY_SPECIALS:
        raise ConfigError(f"{label}: unknown tokenizer family {family!r}")
    expected_eos = FAMILY_SPECIALS[family]["eos"]
    eos = config.get("eos_token_id", expected_eos)
    if eos != expected_eos:
        raise ConfigError(
            f"{label}: eos_token_id={eos} is unsafe for {family}; expected {expected_eos}"
        )

    extended = bool(config.get("tokenizer")) and "ext" in str(config["tokenizer"]).lower()
    if extended and method == "lora":
        modules = set((config.get("lora") or {}).get("modules_to_save") or [])
        if "embed_tokens" not in modules:
            raise ConfigError(
                f"{label}: extended LoRA requires modules_to_save including embed_tokens"
            )

    dataset_id = config.get("dataset_id", SAFE_CPT_DATASET)
    if dataset_id not in SAFE_CPT_DATASETS and not config.get(
        "allow_unsafe_cpt_source", False
    ):
        raise ConfigError(
            f"{label}: CPT dataset {dataset_id!r} is not an allowed cleaned source; "
            "set allow_unsafe_cpt_source only for an explicitly reviewed ablation"
        )
    if "central" in str(dataset_id).lower() and not config.get("allow_unsafe_cpt_source", False):
        raise ConfigError(f"{label}: raw centralized corpus is forbidden for CPT")
    return config


def validate_sft_config(config: dict[str, Any], path: str | Path = "<config>") -> dict[str, Any]:
    label = str(path)
    required = {"run", "model", "dataset", "output_dir", "training", "lora"}
    missing = sorted(required - config.keys())
    if missing:
        raise ConfigError(f"{label}: missing keys: {', '.join(missing)}")
    if config.get("starts_from") not in {"merged_cpt", "stock_instruct"}:
        raise ConfigError(f"{label}: starts_from must be merged_cpt or stock_instruct")
    extended = bool(config.get("extended_tokenizer", True))
    modules = set((config.get("lora") or {}).get("modules_to_save") or [])
    if extended and "embed_tokens" not in modules:
        raise ConfigError(f"{label}: extended SFT LoRA must save embed_tokens")
    family = config.get("tokenizer_family")
    if family is not None:
        if family not in FAMILY_SPECIALS:
            raise ConfigError(f"{label}: unknown tokenizer family {family!r}")
        expected_eos = FAMILY_SPECIALS[family]["eos"]
        if config.get("eos_token_id") != expected_eos:
            raise ConfigError(f"{label}: eos_token_id must be {expected_eos} for {family}")
        dataset = str(config["dataset"])
        if family.startswith("128k") and "128k" not in dataset:
            raise ConfigError(f"{label}: 128k SFT must use separately prepared 128k data")
        if family.startswith("65k") and "128k" in dataset:
            raise ConfigError(f"{label}: 65k SFT cannot use prepared 128k data")
    return config


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
