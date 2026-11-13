#!/usr/bin/env python3
"""Prepare Wolof SFT rows with the LFM Instruct chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    load_dataset = None
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare templated LFM Wolof SFT parquet")
    parser.add_argument("--input", default="data/sft/mixture_v1")
    parser.add_argument("--output", default="data/sft/prepared_mixture_v1")
    parser.add_argument("--template-model", default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer to encode with; required for extended 65k/128k artifacts",
    )
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Limit each split to 1000 rows")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def normalize_messages(value) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    elif not isinstance(value, list) and hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError("messages must be a list or JSON list")
    result = []
    role_map = {"human": "user", "gpt": "assistant"}
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("each message must be a mapping")
        role = role_map.get(message.get("from"), message.get("role"))
        content = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant", "tool"} or content is None:
            raise ValueError(f"invalid message: {message!r}")
        result.append({"role": role, "content": str(content)})
    if not any(message["role"] == "assistant" for message in result):
        raise ValueError("conversation has no assistant turn")
    return result


def assistant_spans(messages: list[dict], tokenizer) -> list[list[int]]:
    """Return tokenized assistant payloads for masking-aware collators."""
    return [
        tokenizer.encode(message["content"], add_special_tokens=False)
        for message in messages
        if message["role"] == "assistant"
    ]


def assistant_only_encoding(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Tokenize one conversation and mask every non-assistant token."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = list(encoded["input_ids"])
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded.get("assistant_tokens_mask")
    if assistant_mask is None or len(assistant_mask) != len(input_ids):
        raise ValueError("chat template did not return a valid assistant token mask")
    labels = [
        token_id if is_assistant else -100
        for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True)
    ]
    if all(label == -100 for label in labels):
        raise ValueError("assistant-only encoding produced no trainable labels")
    return input_ids, labels


def _load_template(args) -> tuple[object, str]:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required")
    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model or args.template_model,
        token=token,
    )
    if args.chat_template:
        template = args.chat_template.read_text(encoding="utf-8")
    else:
        template_source = AutoTokenizer.from_pretrained(args.template_model, token=token)
        template = template_source.chat_template
    if not template:
        raise ValueError("template tokenizer has no chat_template")
    tokenizer.chat_template = template
    return tokenizer, template


def _read_split(source: str, split: str, token: str | None) -> pd.DataFrame:
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    local = path / f"{split}.parquet" if path.is_dir() else path
    if local.is_file():
        return pd.read_parquet(local)
    if load_dataset is None:
        raise RuntimeError("datasets is required for Hub input")
    return load_dataset(source, split=split, token=token).to_pandas()


def prepare_frame(frame: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, int]:
    rows = []
    rejected = 0
    for row in frame.to_dict(orient="records"):
        try:
            messages = normalize_messages(row["messages"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            row["messages"] = json.dumps(messages, ensure_ascii=False)
            row["text"] = text
            row["assistant_token_ids"] = json.dumps(assistant_spans(messages, tokenizer))
            input_ids, labels = assistant_only_encoding(messages, tokenizer)
            row["input_ids"] = input_ids
            row["attention_mask"] = [1] * len(input_ids)
            row["labels"] = labels
            rows.append(row)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            rejected += 1
    return pd.DataFrame(rows), rejected


def require_nonempty_prepared(split: str, input_count: int, output_count: int) -> None:
    if input_count <= 0:
        raise ValueError(f"SFT input split {split!r} is empty")
    if output_count <= 0:
        raise ValueError(f"all {input_count} rows in SFT split {split!r} were rejected")


def run(args: argparse.Namespace) -> int:
    tokenizer, template = _load_template(args)
    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": args.input,
        "template_model": args.template_model,
        "tokenizer_model": args.tokenizer_model or args.template_model,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "assistant_only_prepared": True,
        "format": "TRL-compatible input_ids with explicit -100 assistant-only labels",
        "smoke": args.smoke,
        "splits": {},
    }
    for split in ("train", "val"):
        frame = _read_split(args.input, split, token)
        if args.smoke:
            frame = frame.iloc[:1000].copy()
        prepared, rejected = prepare_frame(frame, tokenizer)
        require_nonempty_prepared(split, len(frame), len(prepared))
        prepared.to_parquet(output / f"{split}.parquet", index=False)
        metadata["splits"][split] = {"input": len(frame), "output": len(prepared), "rejected": rejected}
    (output / "prepare_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "chat_template.jinja").write_text(template, encoding="utf-8")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
