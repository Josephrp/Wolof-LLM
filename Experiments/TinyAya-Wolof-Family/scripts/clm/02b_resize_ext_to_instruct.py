#!/usr/bin/env python3
"""Append Wolof ext chunks onto Tiny Aya Instruct (earth/global) tokenizers.

Base vs Instruct Cohere chat specials use different ids (e.g. START_RESPONSE),
so we cannot reuse the base-extended tokenizer.json as-is. Instead:

1. Load stock Instruct tokenizer (keeps Instruct chat specials / eos=END_OF_TURN).
2. Append the same pretok chunks + ``<|lang_wo|>`` / ``<|ortho_ns|>`` / ``<|ortho_std|>``
   listed in the base ``extension_meta.json``.
3. Resize Instruct weights; mean-pool new rows (or copy by token string from base
   ext-init when available).

Bos/pad stay 2/0; Instruct eos (typically 6) is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.anchoring import ANCHOR_SPECIALS
from common.pipeline import atomic_json_write, utc_now

EXPECTED_BOS = 2
EXPECTED_PAD = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append Wolof extension onto Tiny Aya Instruct tokenizer+weights"
    )
    parser.add_argument("--instruct-id", required=True)
    parser.add_argument(
        "--tokenizer",
        default="tokenizers/tinyaya-wolof-ext-anchors",
        help="Base extended tokenizer dir (provides extension_meta.json)",
    )
    parser.add_argument(
        "--source-model",
        default=None,
        help="Optional base ext-init; copy new rows by matching token string",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-bos", type=int, default=EXPECTED_BOS)
    parser.add_argument("--expected-pad", type=int, default=EXPECTED_PAD)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def append_chunks_to_instruct_tokenizer(
    stock_tok,
    chunks: list[str],
    anchors: tuple[str, ...] | list[str],
):
    """Append Wolof chunks + anchors via HF APIs so Instruct special ids stay put.

    Raw tokenizer.json surgery remaps Cohere chat specials (START_RESPONSE=261000
    …) to the end of the vocab after reload — breaking chat embeddings. ``add_tokens``
    / ``add_special_tokens`` keep existing added-token ids and only grow the tail.
    """
    from transformers import AutoTokenizer

    # Preserve stock special id map before mutation (post-condition check).
    stock_special_ids = {
        tok: stock_tok.convert_tokens_to_ids(tok)
        for tok in (
            "<|START_RESPONSE|>",
            "<|END_RESPONSE|>",
            "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|>",
            "<|SYSTEM_TOKEN|>",
            "<|USER_TOKEN|>",
            "<|CHATBOT_TOKEN|>",
        )
        if stock_tok.convert_tokens_to_ids(tok) != stock_tok.unk_token_id
    }
    v_start = len(stock_tok)

    existing = set(stock_tok.get_vocab())
    # Never re-add Instruct chat/control specials that live only in added_tokens.
    existing.update(stock_special_ids)
    nouveaux = [chunk for chunk in chunks if chunk and chunk not in existing]
    if nouveaux:
        stock_tok.add_tokens(nouveaux)

    existing = set(stock_tok.get_vocab())
    anchors_added = [special for special in anchors if special not in existing]
    if anchors_added:
        stock_tok.add_special_tokens({"additional_special_tokens": list(anchors_added)})

    # Fertility: prefer whole pretok matches (same as base extension).
    # Only flip ignore_merges; never rewrite vocab/added_tokens (that remaps specials).
    pre_fertility = stock_tok
    try:
        backend = json.loads(stock_tok.backend_tokenizer.to_str())
        if backend.get("model", {}).get("type") == "BPE":
            backend["model"]["ignore_merges"] = True
            tmp = tempfile.mkdtemp(prefix="tinyaya_instruct_tok_")
            stock_tok.save_pretrained(tmp)
            with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as handle:
                json.dump(backend, handle, ensure_ascii=False)
            for stale in ("vocab.json", "merges.txt"):
                path = os.path.join(tmp, stale)
                if os.path.exists(path):
                    os.remove(path)
            candidate = AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
            drifted = any(
                candidate.convert_tokens_to_ids(tok) != expected
                for tok, expected in stock_special_ids.items()
            )
            if drifted:
                print(
                    "  warn: ignore_merges reload remapped Instruct specials; "
                    "keeping add_tokens tokenizer without ignore_merges",
                    flush=True,
                )
                stock_tok = pre_fertility
            else:
                stock_tok = candidate
    except Exception as exc:  # noqa: BLE001 — best-effort fertility flag
        print(f"  warn: ignore_merges patch skipped: {exc}", flush=True)
        stock_tok = pre_fertility

    # Guard: chat specials must keep stock ids after append + optional reload.
    for tok, expected in stock_special_ids.items():
        got = stock_tok.convert_tokens_to_ids(tok)
        if got != expected:
            raise SystemExit(
                f"Instruct special id drift after append: {tok} "
                f"expected={expected} got={got}. Refusing to save broken tokenizer."
            )

    return stock_tok, {
        "V": v_start,
        "n_words": len(nouveaux),
        "nouveaux": nouveaux,
        "anchors_added": anchors_added,
        "preserved_special_ids": stock_special_ids,
    }


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    tok_path = Path(args.tokenizer)
    if not tok_path.is_absolute():
        tok_path = ROOT / tok_path
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out

    meta_path = tok_path / "extension_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"missing extension_meta.json under {tok_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = list(meta.get("sample_new_chunks") or [])
    # Prefer full nouveaux list when present in report.
    full_chunks = meta.get("nouveaux") or meta.get("new_chunks")
    if isinstance(full_chunks, list) and full_chunks:
        chunks = list(full_chunks)
    if not chunks and meta.get("sample_new_chunks"):
        # Fall back: re-read from tokenizer vocab beyond V_append_start if needed.
        chunks = list(meta.get("sample_new_chunks") or [])
    if len(chunks) < 100:
        # Recover full nouveaux from base extended tokenizer vocab.
        base_ext = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        v0 = int(meta.get("V_append_start", meta.get("V", 0)))
        recovered = [
            tok
            for tok, tid in sorted(base_ext.get_vocab().items(), key=lambda kv: kv[1])
            if tid >= v0 and not (tok.startswith("<|") and tok.endswith("|>"))
        ]
        if len(recovered) > len(chunks):
            chunks = recovered
            print(f"  recovered {len(chunks)} nouveaux from base ext vocab", flush=True)

    stock_tok = AutoTokenizer.from_pretrained(
        args.instruct_id, token=token, trust_remote_code=True
    )
    if (
        stock_tok.bos_token_id != args.expected_bos
        or stock_tok.pad_token_id != args.expected_pad
    ):
        raise SystemExit(
            f"stock Instruct bos/pad unexpected: bos={stock_tok.bos_token_id} "
            f"eos={stock_tok.eos_token_id} pad={stock_tok.pad_token_id}"
        )
    print(
        f"  Instruct specials bos={stock_tok.bos_token_id} "
        f"eos={stock_tok.eos_token_id} pad={stock_tok.pad_token_id} "
        f"len={len(stock_tok)}",
        flush=True,
    )

    ext_tok, append_meta = append_chunks_to_instruct_tokenizer(
        stock_tok, chunks, ANCHOR_SPECIALS
    )
    v_start = int(append_meta["V"])
    print(
        f"  appended n_words={append_meta['n_words']} "
        f"anchors={append_meta['anchors_added']} -> len={len(ext_tok)}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.instruct_id,
        token=token,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    emb_old = model.get_input_embeddings().weight.data.clone()
    old_rows = emb_old.shape[0]
    target = max(old_rows, len(ext_tok))
    model.resize_token_embeddings(target)
    emb = model.get_input_embeddings().weight.data

    # Map token string -> source embedding row from base ext-init when possible.
    src_by_token: dict[str, torch.Tensor] = {}
    if args.source_model:
        src = Path(args.source_model)
        if not src.is_absolute():
            src = ROOT / src
        if src.is_dir():
            src_tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
            src_model = AutoModelForCausalLM.from_pretrained(
                str(src), dtype=torch.float32, device_map="cpu", trust_remote_code=True
            )
            src_emb = src_model.get_input_embeddings().weight.data
            for tok_str, tid in src_tok.get_vocab().items():
                if tid < src_emb.shape[0]:
                    src_by_token[tok_str] = src_emb[tid]

    sans = 0
    copied = 0
    with torch.no_grad():
        for tok_str, nid in ext_tok.get_vocab().items():
            if nid < v_start:
                continue
            if tok_str in src_by_token and src_by_token[tok_str].shape[0] == emb.shape[1]:
                emb[nid] = src_by_token[tok_str]
                copied += 1
                continue
            texte = ext_tok.convert_tokens_to_string([tok_str])
            sub = [
                sid
                for sid in stock_tok.encode(texte, add_special_tokens=False)
                if sid < old_rows
            ]
            if not sub:
                emb[nid] = emb_old.mean(dim=0)
                sans += 1
                continue
            emb[nid] = emb_old[sub].mean(dim=0)

    # Preserve Instruct eos (END_OF_TURN); do not force base eos=3.
    model.config.bos_token_id = stock_tok.bos_token_id
    model.config.eos_token_id = stock_tok.eos_token_id
    model.config.pad_token_id = stock_tok.pad_token_id
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = True

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    ext_tok.save_pretrained(out)
    report = {
        "created_at": utc_now(),
        "instruct_id": args.instruct_id,
        "base_tokenizer_meta": str(tok_path),
        "source_model": args.source_model,
        "output": str(out),
        "old_rows": old_rows,
        "new_rows": target,
        "V": v_start,
        "n_words": append_meta["n_words"],
        "anchors_added": append_meta["anchors_added"],
        "copied_from_source_by_token": copied,
        "sans_subtokens": sans,
        "bos_token_id": ext_tok.bos_token_id,
        "eos_token_id": ext_tok.eos_token_id,
        "pad_token_id": ext_tok.pad_token_id,
        "ignore_merges": True,
    }
    atomic_json_write(out / "resize_meta.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("RESIZE_INSTRUCT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
