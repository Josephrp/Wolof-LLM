#!/usr/bin/env python3
"""Inject Qwen3.5 chat-template anchors and prepare assistant-only SFT parquet."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.anchoring import ANCHOR_SPECIALS, inject_anchors, normalize_messages
from common.pipeline import atomic_json_write, utc_now

DEFAULT_TEMPLATE_MODEL = "Qwen/Qwen3.5-0.8B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor + prepare Qwen Wolof SFT")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "ortho_ns_mixed_v1" / "sft_stratified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "sft" / "prepared_anchored_v1",
    )
    parser.add_argument("--template-model", default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def as_token_id_list(value) -> list[int]:
    """Normalize apply_chat_template outputs (list | tensor | BatchEncoding) to int ids.

    Newer transformers may return a BatchEncoding from tokenize=True; ``list(encoding)``
    yields dict keys like ``['input_ids', 'attention_mask']`` — never use bare list().
    """
    current = value
    for _ in range(4):
        if current is None:
            break
        if hasattr(current, "keys") and not isinstance(current, (str, bytes, list, tuple)):
            keys = set(current.keys())
            if "input_ids" in keys:
                current = current["input_ids"]
                continue
            break
        break
    if hasattr(current, "tolist"):
        current = current.tolist()
    if isinstance(current, list) and current and isinstance(current[0], list):
        current = current[0]
    if not isinstance(current, (list, tuple)):
        raise TypeError(f"expected token id sequence, got {type(value)!r} -> {type(current)!r}")
    if current and isinstance(current[0], str):
        raise TypeError(
            f"token ids resolved to strings {current[:5]!r}; apply_chat_template misuse"
        )
    return [int(token_id) for token_id in current]


def assistant_only_encoding(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Mask non-assistant tokens. Qwen3.5 templates lack {% generation %}, so use prefix diffs."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = as_token_id_list(encoded)
        assistant_mask = None
        if hasattr(encoded, "keys"):
            assistant_mask = encoded.get("assistant_masks")
            if assistant_mask is None:
                assistant_mask = encoded.get("assistant_tokens_mask")
        if assistant_mask is not None:
            if hasattr(assistant_mask, "tolist"):
                assistant_mask = assistant_mask.tolist()
            if isinstance(assistant_mask, list) and assistant_mask and isinstance(
                assistant_mask[0], list
            ):
                assistant_mask = assistant_mask[0]
        if assistant_mask is not None and len(assistant_mask) == len(input_ids):
            labels = [
                token_id if is_assistant else -100
                for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True)
            ]
            if not all(label == -100 for label in labels):
                return input_ids, labels
    except (TypeError, ValueError):
        pass

    input_ids = as_token_id_list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    labels = [-100] * len(input_ids)
    built: list[dict] = []
    for message in messages:
        built.append(message)
        if message["role"] != "assistant":
            continue
        cur = as_token_id_list(
            tokenizer.apply_chat_template(built, tokenize=True, add_generation_prompt=False)
        )
        prev = (
            as_token_id_list(
                tokenizer.apply_chat_template(
                    built[:-1], tokenize=True, add_generation_prompt=False
                )
            )
            if len(built) > 1
            else []
        )
        # Require prefix alignment; otherwise fall back to training the full turn suffix.
        if len(prev) <= len(cur) and cur[: len(prev)] == prev:
            start = len(prev)
        else:
            start = max(0, len(cur) - max(1, len(cur) // 2))
        end = min(len(cur), len(labels))
        for idx in range(start, end):
            labels[idx] = input_ids[idx] if idx < len(input_ids) else cur[idx]
    if all(label == -100 for label in labels):
        # Last-resort: supervise the whole sequence so smoke/training can proceed.
        labels = list(input_ids)
    return input_ids, [int(label) for label in labels]


def prepare_split(df: pd.DataFrame, tokenizer, smoke: bool) -> pd.DataFrame:
    rows = []
    limit = min(500, len(df)) if smoke else len(df)
    for row in df.head(limit).to_dict(orient="records"):
        messages = normalize_messages(row.get("messages"))
        task = str(row.get("task") or "")
        messages = inject_anchors(messages, task=task)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids, labels = assistant_only_encoding(messages, tokenizer)
        if len(input_ids) < 2:
            raise ValueError(f"prepared example too short ({len(input_ids)} ids)")
        attention_mask = [1] * len(input_ids)
        rows.append(
            {
                "messages": json.dumps(messages, ensure_ascii=False),
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "task": task,
                "source": row.get("source"),
                "ortho_label": row.get("ortho_label"),
            }
        )
    frame = pd.DataFrame(rows)
    # Force int lists so parquet/pyarrow does not stringify token ids.
    for column in ("input_ids", "attention_mask", "labels"):
        frame[column] = frame[column].map(lambda seq: [int(x) for x in seq])
    return frame


def main() -> int:
    args = build_parser().parse_args()
    if AutoTokenizer is None:
        raise SystemExit("transformers required")
    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    tok_id = args.tokenizer_model or args.template_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=token, trust_remote_code=True)
    # Ensure anchor specials exist even before full vocab extension.
    existing = set(tokenizer.get_vocab())
    to_add = [t for t in ANCHOR_SPECIALS if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    train = pd.read_parquet(args.input / "train.parquet")
    val_path = args.input / "val.parquet"
    if not val_path.is_file():
        val_path = args.input / "test.parquet"
    val = pd.read_parquet(val_path)

    args.output.mkdir(parents=True, exist_ok=True)
    train_p = prepare_split(train, tokenizer, args.smoke)
    val_p = prepare_split(val, tokenizer, args.smoke)
    train_p.to_parquet(args.output / "train.parquet", index=False)
    val_p.to_parquet(args.output / "val.parquet", index=False)
    tokenizer.save_pretrained(args.output / "tokenizer_snapshot")
    template = getattr(tokenizer, "chat_template", None)
    if template:
        (args.output / "chat_template.jinja").write_text(str(template), encoding="utf-8")

    meta = {
        "created_at": utc_now(),
        "template_model": args.template_model,
        "tokenizer_model": tok_id,
        "anchor_specials": list(ANCHOR_SPECIALS),
        "n_train": int(len(train_p)),
        "n_val": int(len(val_p)),
        "smoke": args.smoke,
        "added_specials_now": to_add,
    }
    atomic_json_write(args.output / "prepare_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
