#!/usr/bin/env python3
"""Merge LoRA CPT/SFT adapters into a full CausalLM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now

HUB_PREFIXES = ("CohereLabs/", "Tonic/", "galsenai/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Tiny Aya Wolof LoRA adapter")
    parser.add_argument("--base", required=True, help="Base model id or local path")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    base = args.base
    adapter = Path(args.adapter)
    output = Path(args.output)
    if not Path(base).is_absolute() and not str(base).startswith(HUB_PREFIXES):
        maybe = ROOT / base
        if maybe.exists():
            base = str(maybe)
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter), token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    meta = {"created_at": utc_now(), "base": base, "adapter": str(adapter), "output": str(output)}
    atomic_json_write(output / "merge_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
