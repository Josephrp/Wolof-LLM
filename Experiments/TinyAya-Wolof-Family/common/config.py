"""YAML validation for Tiny Aya Wolof Family configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected mapping")
    return data


def sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_clm_config(config: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    label = str(path) if path else "clm"
    for key in ("run", "size", "base_id", "methode", "entrainement"):
        if key not in config:
            raise ConfigError(f"{label}: missing {key}")
    if config["methode"] not in {"lora", "full", "qlora"}:
        raise ConfigError(f"{label}: methode must be lora|full|qlora")
    if config.get("extended_tokenizer"):
        modules = set((config.get("lora") or {}).get("modules_to_save") or [])
        if "embed_tokens" not in modules:
            raise ConfigError(f"{label}: extended LoRA requires modules_to_save embed_tokens")
    return config


def validate_sft_config(config: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    label = str(path) if path else "sft"
    for key in ("run", "model", "dataset", "output_dir", "training", "lora"):
        if key not in config:
            raise ConfigError(f"{label}: missing {key}")
    allowed = {"merged_cpt", "stock_earth", "stock_global"}
    if config.get("starts_from") not in allowed:
        raise ConfigError(f"{label}: starts_from must be one of {sorted(allowed)}")
    if config.get("extended_tokenizer"):
        modules = set((config.get("lora") or {}).get("modules_to_save") or [])
        if "embed_tokens" not in modules:
            raise ConfigError(f"{label}: extended LoRA requires modules_to_save embed_tokens")
    return config
