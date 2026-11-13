#!/usr/bin/env python3
"""Evaluate an SFT checkpoint: val loss by task + qualitative Wolof prompts."""

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
from common.anchoring import SYSTEM_ANCHOR
from common.pipeline import atomic_json_write, utc_now

HUB_PREFIXES = ("CohereLabs/", "Tonic/", "galsenai/")

WOLOF_PROMPTS = [
    "Waxal ma ci sa bopp.",
    "Lan mooy njariñu jang?",
    "Bind ab bataaxal bu gatt.",
    "Naka lañuy aar àdduna?",
    "Leeralal ma ndimbalug dëkk.",
    "Joxe ñetti xelal ci liggéey.",
    "Tëral ab waxtaan ci wér-gu-yaram.",
    "Lan la njàngat mi tekki?",
    "Wax léeb bu gatt.",
    "Naka la nit man a jàng Wolof?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Tiny Aya Wolof SFT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/sft/prepared_anchored_v1")
    parser.add_argument("--output", default="results/sft/eval")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def resolve_model(path_or_id: str) -> str:
    if path_or_id.startswith(HUB_PREFIXES):
        return path_or_id
    p = Path(path_or_id)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def _row_loss(model, tokenizer, text: str) -> float:
    import torch

    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.no_grad():
        return float(model(**batch, labels=batch["input_ids"]).loss)


def main() -> int:
    args = build_parser().parse_args()
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    model_id = resolve_model(args.model)
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    frame = pd.read_parquet(data / "val.parquet").iloc[: args.max_rows]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    ).eval()

    losses = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        losses[str(row.get("task", "unknown"))].append(
            _row_loss(model, tokenizer, row["text"])
        )
    summaries = {
        task: {
            "n": len(values),
            "mean_loss": sum(values) / len(values),
            "ppl": math.exp(sum(values) / len(values)),
        }
        for task, values in losses.items()
        if values
    }
    overall = [v for vals in losses.values() for v in vals]
    mean_loss = sum(overall) / max(len(overall), 1)

    generations = []
    for prompt in WOLOF_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_ANCHOR},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=128, do_sample=False)
        answer = tokenizer.decode(
            generated[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        generations.append({"prompt": prompt, "response": answer})

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "qualitative.jsonl").open("w", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "created_at": utc_now(),
        "model": model_id,
        "mean_loss": mean_loss,
        "per_task": summaries,
        "qualitative_count": len(generations),
    }
    atomic_json_write(output / "evaluation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
