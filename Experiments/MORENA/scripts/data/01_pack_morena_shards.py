#!/usr/bin/env python3
"""Pack Wolof text into MORENA memmap shards (index.json + *.bin [+ optional mask]).

Shard format matches vendor modeling_morena.Source:
  index.json: {dtype, eos_id, shards:[{file,n_tokens,mask?}]}
  tokens: contiguous int32/uint16 memmap
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def iter_texts(path: Path, text_column: str):
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        for value in frame[text_column].dropna().astype(str):
            text = value.strip()
            if text:
                yield text
        return
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get(text_column) or row.get("text") or "").strip()
            if text:
                yield text
        return
    # plain text: one document per blank-line block or per line
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if buf:
                yield "\n".join(buf).strip()
                buf = []
            continue
        buf.append(line.rstrip())
    if buf:
        yield "\n".join(buf).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack Wolof corpus for MORENA trainer")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-repo", default="vamboai/morena-1.5b-base")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--dtype", default="uint16", choices=("uint16", "int32"))
    parser.add_argument("--shard-tokens", type=int, default=50_000_000)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument(
        "--sft-jsonl",
        action="store_true",
        help="Input JSONL with prompt/response; write loss mask (1=response)",
    )
    args = parser.parse_args()

    tok_path = args.tokenizer
    if tok_path is None:
        tok_path = Path(hf_hub_download(args.tokenizer_repo, "tokenizer.json"))
    tok = Tokenizer.from_file(str(tok_path))
    eos_id = tok.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer missing <eos>")
    user_id = tok.token_to_id("<reserved_0>")
    asst_id = tok.token_to_id("<reserved_1>")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(args.dtype)
    shards = []
    shard_idx = 0
    buffer: list[int] = []
    mask_buffer: list[int] = []

    def flush():
        nonlocal shard_idx, buffer, mask_buffer
        if not buffer:
            return
        name = f"shard_{shard_idx:05d}.bin"
        arr = np.asarray(buffer, dtype=dtype)
        arr.tofile(out / name)
        entry = {"file": name, "n_tokens": int(arr.size)}
        if args.sft_jsonl:
            mname = f"shard_{shard_idx:05d}.mask"
            np.asarray(mask_buffer, dtype=np.uint8).tofile(out / mname)
            entry["mask"] = mname
        shards.append(entry)
        shard_idx += 1
        buffer = []
        mask_buffer = []

    n_docs = 0
    if args.sft_jsonl:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or row.get("user") or "").strip()
            response = str(row.get("response") or row.get("assistant") or "").strip()
            if not prompt or not response:
                continue
            # MORENA chat format
            parts = []
            if user_id is not None and asst_id is not None:
                text = f"<reserved_0>\n{prompt}\n<reserved_1>\n{response}"
            else:
                text = f"{prompt}\n{response}"
            ids = [eos_id] + tok.encode(text, add_special_tokens=False).ids
            # Mask: train on assistant span only (approx: tokens after first <reserved_1>)
            mask = [0] * len(ids)
            if asst_id is not None and asst_id in ids:
                start = ids.index(asst_id) + 1
                for i in range(start, len(ids)):
                    mask[i] = 1
            else:
                # fallback: last half
                start = len(ids) // 2
                for i in range(start, len(ids)):
                    mask[i] = 1
            buffer.extend(ids)
            mask_buffer.extend(mask)
            n_docs += 1
            if args.max_docs and n_docs >= args.max_docs:
                break
            if len(buffer) >= args.shard_tokens:
                flush()
    else:
        for text in iter_texts(args.input, args.text_column):
            ids = [eos_id] + tok.encode(text, add_special_tokens=False).ids
            buffer.extend(ids)
            n_docs += 1
            if args.max_docs and n_docs >= args.max_docs:
                break
            if len(buffer) >= args.shard_tokens:
                flush()
    flush()

    index = {
        "dtype": str(np.dtype(dtype)),
        "eos_id": int(eos_id),
        "tokenizer_repo": args.tokenizer_repo,
        "n_docs": n_docs,
        "shards": shards,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "n_docs": n_docs, "n_shards": len(shards), "eos_id": eos_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
