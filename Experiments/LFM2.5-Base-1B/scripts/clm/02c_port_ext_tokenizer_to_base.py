#!/usr/bin/env python3
"""Port an append-only Wolof extension onto a different 65k LFM2.5 base.

Used when a sibling base (e.g. 230M) adds trailing specials such as <think>
that are absent from the 1.2B/350M bases the shared ext4000 tokenizer was
built on. The Wolof extension strings are kept in ID order and appended after
the target base vocab; existing target specials are preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EOS_65K = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-id",
        required=True,
        help="Target stock base, e.g. LiquidAI/LFM2.5-230M-Base",
    )
    parser.add_argument(
        "--source-tokenizer",
        required=True,
        help="Existing extended tokenizer (1.2B/350M family)",
    )
    parser.add_argument(
        "--source-append-start",
        type=int,
        default=64400,
        help="First extension ID in the source tokenizer",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-eos", type=int, default=EXPECTED_EOS_65K)
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source = AutoTokenizer.from_pretrained(args.source_tokenizer)
    target = AutoTokenizer.from_pretrained(args.base_id)
    if target.eos_token_id != args.expected_eos:
        raise ValueError(
            f"target eos_token_id={target.eos_token_id}, expected {args.expected_eos}"
        )

    source_vocab = source.get_vocab()
    extension = sorted(
        (
            (token_id, token)
            for token, token_id in source_vocab.items()
            if token_id >= args.source_append_start
        ),
        key=lambda item: item[0],
    )
    if not extension:
        raise ValueError("source tokenizer has no extension tokens to port")

    target_before = len(target)
    target_vocab = target.get_vocab()
    collisions = [
        token for _, token in extension if token in target_vocab
    ]
    if collisions:
        raise ValueError(
            "extension tokens already exist on target base: "
            + ", ".join(repr(token) for token in collisions[:5])
        )

    added = target.add_tokens([token for _, token in extension])
    if added != len(extension):
        raise ValueError(f"expected to add {len(extension)} tokens, added {added}")

    for offset, (_, token) in enumerate(extension):
        expected_id = target_before + offset
        actual_id = target.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(
                f"extension ID drift for {token!r}: expected {expected_id}, got {actual_id}"
            )

    target.save_pretrained(output_dir)
    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_id": args.base_id,
        "source_tokenizer": str(args.source_tokenizer),
        "source_append_start": args.source_append_start,
        "target_vocab_before": target_before,
        "target_vocab_after": len(target),
        "n_ported": len(extension),
        "eos_token_id": target.eos_token_id,
        "bos_token_id": target.bos_token_id,
        "pad_token_id": target.pad_token_id,
        "think_token_ids": {
            token: target_vocab.get(token)
            for token in ("<think>", "</think>")
            if token in target_vocab
        },
        "output_dir": str(output_dir.relative_to(ROOT)),
    }
    (output_dir / "port_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
