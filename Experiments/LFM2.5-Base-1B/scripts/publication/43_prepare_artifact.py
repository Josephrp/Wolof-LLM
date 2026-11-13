#!/usr/bin/env python3
"""Attach factual metadata and a conservative model card to a merged artifact."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.publication import publishable_files, scan_for_secrets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a private production model artifact")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-family", choices=("65k-ext", "128k-ext"), required=True)
    parser.add_argument("--training-stage", choices=("CPT", "Instruct"), required=True)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _require_artifact_files(artifact: Path) -> None:
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (artifact / name).is_file()]
    weights = list(artifact.glob("*.safetensors"))
    if missing or not weights:
        raise ValueError(f"incomplete merged artifact: missing={missing}, safetensors={len(weights)}")
    if any(path.stat().st_size == 0 for path in weights):
        raise ValueError("merged artifact contains an empty safetensors file")


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"/mnt/(?:volume|data)[^/\s]*/[^\s\"']*", "<REDACTED_REMOTE_PATH>", value)
        value = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "<REDACTED_TOKEN>", value)
        value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<REDACTED_IP>", value)
        return value
    return value


def run(args: argparse.Namespace) -> int:
    artifact = _resolve(args.artifact)
    metrics = _resolve(args.metrics)
    run_manifest = _resolve(args.run_manifest)
    _require_artifact_files(artifact)
    loaded = {}
    for path in (metrics, run_manifest):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required factual metadata is missing or empty: {path}")
        loaded[path] = _redact(json.loads(path.read_text(encoding="utf-8")))
    (artifact / "metrics.json").write_text(
        json.dumps(loaded[metrics], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifact / "training_manifest.json").write_text(
        json.dumps(loaded[run_manifest], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    card = f"""---
base_model: {args.base_model}
pipeline_tag: text-generation
library_name: transformers
---

# {args.model_name}

> **Private production artifact** — ongoing/experimental and not a public release.

## Exact base

`{args.base_model}`

## Training stage

{args.training_stage}. This artifact passed the recorded automated gates in
`training_manifest.json`; no unrecorded quality claim is made.

## Tokenizer family

`{args.tokenizer_family}`. Tokenizer-family-specific BPB must not be compared as
perplexity across the 65k and 128k families.

## Data caveat

Continual pretraining uses the cleaned Wolof corpus protocol. Instruction data is
reweighted and excludes the source Hub test split, but benchmark-like examples may
have existed in upstream pretraining. Raw private rows are not included here.

## Intended use

Private research and evaluation of Wolof language adaptation.

## Limitations

This model is experimental. Wolof orthography, code-switching, factuality,
reasoning, long-context behavior, and safety have not been comprehensively
validated. Native-speaker review remains necessary before broader use.

## Metrics

Only measured outputs are included in `metrics.json`. Absence of a metric means it
was not measured; it must not be inferred.
"""
    (artifact / "README.md").write_text(card, encoding="utf-8")
    files = publishable_files(artifact)
    scan_for_secrets(artifact, files)
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
