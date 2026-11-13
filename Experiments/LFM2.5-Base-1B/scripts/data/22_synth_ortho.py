#!/usr/bin/env python3
"""Build validated stripped/restored Wolof diacritic instruction pairs."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.synthetic import read_jsonl, write_jsonl


def strip_diacritics(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def build_pairs(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        restored = unicodedata.normalize("NFC", str(row.get("text", "")).strip())
        stripped = strip_diacritics(restored)
        if restored and stripped != restored:
            output.append({
                "prompt": f"Jubbantil mbind mi te delloo màndarga yi: {stripped}",
                "chosen": restored,
                "task": "orthography",
                "synthetic": True,
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Wolof diacritic restoration pairs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_jsonl(args.output, build_pairs(read_jsonl(args.input)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
