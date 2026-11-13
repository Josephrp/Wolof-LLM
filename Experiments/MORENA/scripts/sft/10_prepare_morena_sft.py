#!/usr/bin/env python3
"""Prepare MORENA-marker SFT JSONL from a simple prompt/response parquet or JSONL.

Output JSONL fields: prompt, response — consumed by 01_pack_morena_shards.py --sft-jsonl.
Does NOT use LFM / ChatML templates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def rows_from_path(path: Path, prompt_col: str, response_col: str):
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            yield str(row[prompt_col]), str(row[response_col])
        return
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            yield str(row.get(prompt_col) or row.get("prompt") or ""), str(
                row.get(response_col) or row.get("response") or row.get("completion") or ""
            )
        return
    raise ValueError(f"unsupported input: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--response-col", default="response")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for prompt, response in rows_from_path(args.input, args.prompt_col, args.response_col):
            prompt, response = prompt.strip(), response.strip()
            if not prompt or not response:
                continue
            handle.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n += 1
            if args.max_rows and n >= args.max_rows:
                break
    print(json.dumps({"out": str(args.out), "n": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
