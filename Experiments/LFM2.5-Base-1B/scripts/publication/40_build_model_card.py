#!/usr/bin/env python3
"""Render a model card only after license/leakage fields are explicitly completed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[2]


def validate_manifest(manifest: dict) -> None:
    required = {
        "model_name", "base_model", "license_reviewed_by", "license_compatible",
        "leakage_caveat", "training_data", "intended_use", "limitations",
    }
    missing = sorted(key for key in required if not manifest.get(key))
    if missing:
        raise ValueError(f"publication manifest incomplete: {missing}")
    if manifest["license_compatible"] is not True:
        raise ValueError("license_compatible must be explicitly true after human review")
    if not manifest.get("metrics"):
        raise ValueError("real metrics are required; model card generation refuses placeholders")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and render an LFM Wolof model card")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    environment = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    rendered = environment.get_template("model_card.md.j2").render(**manifest)
    Path(args.output).write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
