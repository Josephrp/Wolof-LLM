#!/usr/bin/env python3
"""Require configured LoRA targets to be proven by a model module probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LoRA targets against probe JSON")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    return parser


def validate(probe: dict, config: dict, label: str) -> None:
    discovered = set(probe.get("linear_suffixes", [])) | set(probe.get("conv_suffixes", []))
    targets = set((config.get("lora") or {}).get("target_modules") or [])
    if not targets:
        raise ValueError(f"{label}: no LoRA target_modules configured")
    invalid = sorted(targets - discovered)
    if invalid:
        raise ValueError(f"{label}: targets absent from probe: {invalid}")
    architecture = probe.get("architecture")
    size = str(config.get("size", ""))
    if architecture == "moe" or "8b" in size:
        if architecture != "moe":
            raise ValueError(f"{label}: 8B recipe requires a MoE model probe")
        expert_targets = set(probe.get("expert_suffixes", []))
        # LFM2.5-8B-A1B packs expert weights as non-Linear params (gate_up_proj /
        # down_proj), so expert_suffixes may be empty. Attention-only LoRA is the
        # gated starting recipe in that case; require expert overlap only when
        # PEFT-addressable expert Linear suffixes were probed.
        if expert_targets and not (targets & expert_targets):
            raise ValueError(f"{label}: MoE recipe does not target any probed expert module")
        if not expert_targets and not (targets & {"q_proj", "k_proj", "v_proj", "out_proj", "o_proj"}):
            raise ValueError(
                f"{label}: MoE probe has no Linear expert suffixes; "
                "attention LoRA targets are required"
            )


def run(args: argparse.Namespace) -> int:
    probe_path = args.probe if args.probe.is_absolute() else ROOT / args.probe
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    for path in args.config:
        config_path = path if path.is_absolute() else ROOT / path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        validate(probe, config, str(config_path))
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
