#!/usr/bin/env python3
"""Pack long documents and optionally inject deterministic Wolof needles."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.synthetic import read_jsonl, write_jsonl


def pack_documents(rows: list[dict], target_words: int) -> list[dict]:
    packs, buffer, sources = [], [], []
    for row in rows:
        words = str(row.get("text", "")).split()
        if not words:
            continue
        buffer.extend(words)
        sources.append(row.get("id"))
        while len(buffer) >= target_words:
            packs.append({"text": " ".join(buffer[:target_words]), "source_ids": list(sources)})
            buffer = buffer[target_words:]
            sources = []
    return packs


def add_needles(packs: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for index, pack in enumerate(packs):
        words = pack["text"].split()
        fact = f"Ndimbal bu nëbbu bi mooy LIM-{index:06d}."
        position = rng.randrange(len(words) + 1)
        words[position:position] = fact.split()
        result.append({
            **pack,
            "text": " ".join(words),
            "needle": fact,
            "question": "Lan mooy ndimbal bu nëbbu bi?",
            "answer": f"LIM-{index:06d}",
            "needle_word_position": position,
        })
    return result


def run(args: argparse.Namespace) -> int:
    packs = pack_documents(read_jsonl(args.input), args.target_words)
    if args.needles:
        packs = add_needles(packs, args.seed)
    write_jsonl(args.output, packs)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic long-context/needle inputs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-words", type=int, choices=(4000, 8000, 16000, 32000), default=4000)
    parser.add_argument("--needles", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
