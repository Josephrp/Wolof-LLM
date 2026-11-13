#!/usr/bin/env python3
"""Inject Cohere Tiny Aya developer anchors and prepare assistant-only SFT parquet.

Label mask: supervise tokens strictly between <|START_RESPONSE|> … <|END_RESPONSE|>.
No Qwen <think> wrapping.
"""

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
from common.anchoring import (
    ANCHOR_SPECIALS,
    END_RESPONSE,
    START_RESPONSE,
    inject_anchors,
    normalize_messages,
)
from common.ortho_variants import get_variant, resolve_variants
from common.pipeline import atomic_json_write, utc_now

DEFAULT_TEMPLATE_MODEL = "CohereLabs/tiny-aya-earth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor + prepare Tiny Aya Wolof SFT")
    parser.add_argument(
        "--ortho",
        default="ns",
        help="ns | std | both — selects input/output roots and system anchor",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override SFT stratified input dir (single ortho only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override prepared output dir (single ortho only)",
    )
    parser.add_argument("--template-model", default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def _token_id(tokenizer, token_str: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(token_str)
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(tid)


def response_span_labels(input_ids: list[int], tokenizer) -> list[int]:
    """Mask labels to assistant body between START_RESPONSE and END_RESPONSE."""
    start_id = _token_id(tokenizer, START_RESPONSE)
    end_id = _token_id(tokenizer, END_RESPONSE)
    labels = [-100] * len(input_ids)
    if start_id is None or end_id is None:
        return labels

    i = 0
    while i < len(input_ids):
        if input_ids[i] != start_id:
            i += 1
            continue
        # Supervise content after START_RESPONSE, stop before END_RESPONSE.
        j = i + 1
        while j < len(input_ids) and input_ids[j] != end_id:
            labels[j] = input_ids[j]
            j += 1
        i = j + 1 if j < len(input_ids) else j
    return labels


def assistant_only_encoding(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Encode with Tiny Aya chat template; mask only START_RESPONSE…END_RESPONSE body."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        input_ids = list(encoded["input_ids"])
    except (TypeError, ValueError):
        input_ids = list(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        )

    labels = response_span_labels(input_ids, tokenizer)
    if not all(label == -100 for label in labels):
        return input_ids, labels

    # Fallback: TRL-style assistant mask if template supports it.
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = list(encoded["input_ids"])
        assistant_mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
        if assistant_mask is not None and len(assistant_mask) == len(input_ids):
            labels = [
                token_id if is_assistant else -100
                for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True)
            ]
            if not all(label == -100 for label in labels):
                return input_ids, labels
    except (TypeError, ValueError):
        pass

    # Last resort: supervise full sequence so smoke/training can proceed.
    return input_ids, list(input_ids)


def prepare_split(
    df: pd.DataFrame, tokenizer, smoke: bool, *, ortho: str
) -> pd.DataFrame:
    rows = []
    limit = min(500, len(df)) if smoke else len(df)
    for row in df.head(limit).to_dict(orient="records"):
        messages = normalize_messages(row.get("messages"))
        task = str(row.get("task") or "")
        messages = inject_anchors(messages, task=task, ortho=ortho)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids, labels = assistant_only_encoding(messages, tokenizer)
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
                "ortho_variant": ortho,
            }
        )
    return pd.DataFrame(rows)


def prepare_one(
    *,
    ortho: str,
    input_dir: Path,
    output_dir: Path,
    tokenizer,
    template_model: str,
    tok_id: str,
    to_add: list[str],
    smoke: bool,
) -> dict:
    train = pd.read_parquet(input_dir / "train.parquet")
    val_path = input_dir / "val.parquet"
    if not val_path.is_file():
        val_path = input_dir / "test.parquet"
    val = pd.read_parquet(val_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_p = prepare_split(train, tokenizer, smoke, ortho=ortho)
    val_p = prepare_split(val, tokenizer, smoke, ortho=ortho)
    train_p.to_parquet(output_dir / "train.parquet", index=False)
    val_p.to_parquet(output_dir / "val.parquet", index=False)
    tokenizer.save_pretrained(output_dir / "tokenizer_snapshot")
    template = getattr(tokenizer, "chat_template", None)
    if template:
        (output_dir / "chat_template.jinja").write_text(str(template), encoding="utf-8")

    meta = {
        "created_at": utc_now(),
        "ortho": ortho,
        "anchor_token": get_variant(ortho)["anchor_token"],
        "template_model": template_model,
        "tokenizer_model": tok_id,
        "anchor_specials": list(ANCHOR_SPECIALS),
        "label_mask": f"{START_RESPONSE}…{END_RESPONSE}",
        "think_wrap": False,
        "n_train": int(len(train_p)),
        "n_val": int(len(val_p)),
        "smoke": smoke,
        "added_specials_now": to_add,
        "input": str(input_dir),
        "output": str(output_dir),
    }
    atomic_json_write(output_dir / "prepare_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> int:
    args = build_parser().parse_args()
    if AutoTokenizer is None:
        raise SystemExit("transformers required")
    keys = resolve_variants(args.ortho)
    if (args.input is not None or args.output is not None) and len(keys) != 1:
        raise SystemExit("--input/--output require a single --ortho (ns or std)")

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    tok_id = args.tokenizer_model or args.template_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, token=token, trust_remote_code=True)
    # Base ext-anchors lacks Cohere chat_template; copy from Instruct template model.
    if not getattr(tokenizer, "chat_template", None):
        template_tok = AutoTokenizer.from_pretrained(
            args.template_model, token=token, trust_remote_code=True
        )
        if not getattr(template_tok, "chat_template", None):
            raise SystemExit(
                f"tokenizer {tok_id!r} has no chat_template and template-model "
                f"{args.template_model!r} also lacks one"
            )
        tokenizer.chat_template = template_tok.chat_template
        print(
            f"  copied chat_template from {args.template_model} -> {tok_id}",
            flush=True,
        )
    existing = set(tokenizer.get_vocab())
    to_add = [t for t in ANCHOR_SPECIALS if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    # Need response markers for assistant-only label masks.
    for marker in (START_RESPONSE, END_RESPONSE):
        tid = tokenizer.convert_tokens_to_ids(marker)
        if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
            raise SystemExit(
                f"tokenizer {tok_id!r} missing {marker!r} (unk). "
                "Use an Instruct-extended tokenizer (earth/global-ext-init), "
                "not base ext-anchors alone."
            )

    for key in keys:
        variant = get_variant(key)
        input_dir = args.input
        if input_dir is None:
            input_dir = ROOT / "data" / variant["data_dirname"] / "sft_stratified"
        elif not input_dir.is_absolute():
            input_dir = ROOT / input_dir
        output_dir = args.output
        if output_dir is None:
            output_dir = ROOT / "data" / "sft" / variant["prepared_dirname"]
        elif not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        prepare_one(
            ortho=key,
            input_dir=input_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            template_model=args.template_model,
            tok_id=tok_id,
            to_add=to_add,
            smoke=args.smoke,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
