#!/usr/bin/env python3
"""Lightweight per-language BPB on NTP ortho test (stock / CPT)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now

HUB_PREFIXES = ("CohereLabs/", "Tonic/", "galsenai/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BPB on NTP test")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "ortho_ns_mixed_v1" / "cpt" / "test.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-texts", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    model_id = args.model
    if not str(model_id).startswith(HUB_PREFIXES):
        maybe = ROOT / model_id
        if maybe.exists():
            model_id = str(maybe)
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    df = pd.read_parquet(data).head(args.max_texts)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    totals = defaultdict(lambda: {"nll_bits": 0.0, "bytes": 0, "n": 0})
    with torch.no_grad():
        for row in df.to_dict(orient="records"):
            text = str(row.get("text") or "")
            if len(text.encode("utf-8")) < 8:
                continue
            lang = str(row.get("lang") or "unk")
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_len,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            n_tokens = input_ids.size(1) - 1
            nll_nats = float(out.loss.item()) * n_tokens
            nll_bits = nll_nats / math.log(2)
            nbytes = len(text.encode("utf-8"))
            totals[lang]["nll_bits"] += nll_bits
            totals[lang]["bytes"] += nbytes
            totals[lang]["n"] += 1
            totals["all"]["nll_bits"] += nll_bits
            totals["all"]["bytes"] += nbytes
            totals["all"]["n"] += 1

    bpb = {
        lang: (vals["nll_bits"] / vals["bytes"] if vals["bytes"] else None)
        for lang, vals in totals.items()
    }
    result = {
        "created_at": utc_now(),
        "model": str(model_id),
        "data": str(data),
        "bpb": bpb,
        "counts": {k: v["n"] for k, v in totals.items()},
    }
    out = args.output
    if not out.is_absolute():
        out = ROOT / out
    atomic_json_write(out, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
