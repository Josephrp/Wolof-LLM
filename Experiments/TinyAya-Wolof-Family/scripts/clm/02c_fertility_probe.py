#!/usr/bin/env python3
"""Stock CohereTokenizer fertility probe on ortho ns+mixed Wolof CPT text.

Run before choosing n_tokens for append-only vocab extension (G2b).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now

MOTS_TEST = [
    "jefandikoo",
    "wooral",
    "cakuteef",
    "jewrinu",
    "reew",
    "waaw",
    "ndaxte",
    "buur",
    "xamxam",
    "jàmm",
    "jamm",
    "mbokk",
    "teranga",
    "ndank",
]

# Earth shares the gated base's 262k CohereTokenizer; prefer when base gate is blocked.
DEFAULT_BASE = "CohereLabs/tiny-aya-earth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny Aya stock tokenizer fertility probe")
    parser.add_argument("--base-id", default=DEFAULT_BASE)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "ortho_ns_mixed_v1" / "cpt" / "train.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "fertility" / "stock_tinyaya.json",
    )
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def fertility_corpus(tokenizer, texts: list[str]) -> dict:
    n_tokens = n_chars = n_unk = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for text in texts:
        text = "" if text is None else str(text)
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens += len(ids)
        n_chars += len(text)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
    return {
        "tokens_per_char": n_tokens / max(n_chars, 1),
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "unk_rate": n_unk / max(n_tokens, 1),
    }


def fertility_words(tokenizer, words: list[str]) -> dict:
    rows = []
    total = 0
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        total += len(ids)
        rows.append(
            {
                "word": word,
                "n_tok": len(ids),
                "pieces": tokenizer.convert_ids_to_tokens(ids),
            }
        )
    return {"mean_tokens": total / max(len(words), 1), "words": rows}


def main() -> int:
    args = build_parser().parse_args()
    import pandas as pd
    from transformers import AutoTokenizer

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    data = args.data if args.data.is_absolute() else ROOT / args.data
    df = pd.read_parquet(data).head(args.max_rows)
    if "lang" in df.columns:
        wo = df[df["lang"].astype(str) == "wo"]
        if len(wo) >= 50:
            df = wo
    if "ortho_label" in df.columns:
        keep = df[df["ortho_label"].astype(str).isin({"non_standard", "mixed"})]
        if len(keep) >= 20:
            df = keep
    texts = df["text"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.base_id, token=token, trust_remote_code=True)
    corpus = fertility_corpus(tokenizer, texts)
    words = fertility_words(tokenizer, MOTS_TEST)
    report = {
        "created_at": utc_now(),
        "base_id": args.base_id,
        "data": str(data),
        "n_texts": len(texts),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": len(tokenizer),
        "corpus": corpus,
        "probe_words": words,
        "note": "Tune configs/tokenizer/ext_wolof_anchors.yaml n_tokens from this probe",
    }
    out = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json_write(out, report)
    print(json.dumps(report, indent=2), flush=True)
    print("FERTILITY_PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
