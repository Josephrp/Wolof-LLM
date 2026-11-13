#!/usr/bin/env python3
"""Merge an SFT LoRA adapter into its start checkpoint (CPT or resized Instruct)."""

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
EXPECTED_BOS = 2
EXPECTED_PAD = 0
# Base CPT uses eos=3 (<EOS_TOKEN>); Instruct earth/global use eos=6 (END_OF_TURN).
ALLOWED_EOS = (3, 6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Tiny Aya Wolof SFT adapter")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-bos", type=int, default=EXPECTED_BOS)
    parser.add_argument(
        "--expected-eos",
        type=int,
        default=None,
        help="If set, require this eos id; otherwise allow base(3) or Instruct(6)",
    )
    parser.add_argument("--expected-pad", type=int, default=EXPECTED_PAD)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def resolve(path_or_id: str) -> str:
    if path_or_id.startswith(HUB_PREFIXES):
        return path_or_id
    p = Path(path_or_id)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    base = resolve(args.base)
    adapter = Path(args.adapter)
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter), token=token, trust_remote_code=True)
    allowed_eos = (args.expected_eos,) if args.expected_eos is not None else ALLOWED_EOS
    if (
        tokenizer.bos_token_id != args.expected_bos
        or tokenizer.eos_token_id not in allowed_eos
        or tokenizer.pad_token_id != args.expected_pad
    ):
        raise SystemExit(
            f"tokenizer specials mismatch: bos={tokenizer.bos_token_id} "
            f"eos={tokenizer.eos_token_id} pad={tokenizer.pad_token_id} "
            f"(allowed eos={allowed_eos})"
        )

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
    # Preserve the adapter/tokenizer specials (Instruct eos=6 must not be forced to 3).
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    meta = {
        "created_at": utc_now(),
        "base": base,
        "adapter": str(adapter),
        "output": str(output),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    atomic_json_write(output / "merge_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
