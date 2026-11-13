#!/usr/bin/env python3
"""Resize an LFM2.5 base to an existing extended tokenizer.

65k: keeps original special IDs and appends new rows.
128k: applies insert-before-specials (+N special shift) with embedding
row permutation, then mean-pools the inserted rows. Embedding norms are
never recalibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.model import ecrire_tie_flag, lire_tie_flag

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EOS_65K = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resize a 230M/350M LFM2.5 base to a 65k extended tokenizer"
    )
    parser.add_argument(
        "--base-id",
        required=True,
        help="Target 65k base, e.g. LiquidAI/LFM2.5-350M-Base",
    )
    parser.add_argument(
        "--source-tokenizer",
        required=True,
        help="Existing extended tokenizer directory or Hub id",
    )
    parser.add_argument(
        "--source-model",
        default=None,
        help="Optional extended init model; copy new rows when dimensions match",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-eos", type=int, default=EXPECTED_EOS_65K)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="CPU loading dtype; output is always saved as bfloat16",
    )
    return parser


def _special_ids(tokenizer) -> dict[str, int | None]:
    return {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
    }


def _verify_compatible_base(
    base_tok, ext_tok, expected_eos: int | None
) -> tuple[int, int]:
    """Return (extension_start, special_shift).

    65k: specials unchanged, extension_start=len(base_tok), shift=0.
    128k insert-before-specials: specials uniformly shifted by N, extension_start
    is the first special id on the *base* tokenizer (insert point).
    """
    if len(ext_tok) <= len(base_tok):
        raise ValueError(
            f"source tokenizer is not extended: {len(ext_tok)} <= {len(base_tok)}"
        )

    base_sp = _special_ids(base_tok)
    ext_sp = _special_ids(ext_tok)
    if base_sp["eos"] is None or ext_sp["eos"] is None:
        raise ValueError("eos_token_id missing on base or extended tokenizer")
    shift = int(ext_sp["eos"]) - int(base_sp["eos"])
    if shift < 0:
        raise ValueError(f"extended eos moved backwards: {base_sp} -> {ext_sp}")

    base_vocab = base_tok.get_vocab()
    ext_vocab = ext_tok.get_vocab()

    if shift == 0:
        mismatches = [
            token
            for token, token_id in base_vocab.items()
            if ext_vocab.get(token) != token_id
        ]
        if mismatches:
            example = ", ".join(repr(token) for token in mismatches[:5])
            raise ValueError(
                f"extended tokenizer is not append-only compatible: "
                f"{len(mismatches)} base IDs differ ({example})"
            )
        if base_sp != ext_sp:
            raise ValueError(
                f"special IDs changed: base={base_sp}, extended={ext_sp}"
            )
        if expected_eos is not None and base_tok.eos_token_id != expected_eos:
            raise ValueError(
                f"target eos_token_id={base_tok.eos_token_id}, expected {expected_eos}"
            )
        return len(base_tok), 0

    # 128k: require uniform special shift; BPE ids below pad stay fixed.
    for key in ("pad", "bos", "eos"):
        if base_sp[key] is None or ext_sp[key] is None:
            raise ValueError(f"missing special {key}: base={base_sp} ext={ext_sp}")
        if int(ext_sp[key]) != int(base_sp[key]) + shift:
            raise ValueError(
                f"non-uniform special shift on {key}: base={base_sp} ext={ext_sp}"
            )
    insert_at = int(base_sp["pad"])
    mismatches = [
        token
        for token, token_id in base_vocab.items()
        if token_id < insert_at and ext_vocab.get(token) != token_id
    ]
    if mismatches:
        example = ", ".join(repr(token) for token in mismatches[:5])
        raise ValueError(
            f"128k extended tokenizer changed base BPE ids: "
            f"{len(mismatches)} differ ({example})"
        )
    if len(ext_tok) != len(base_tok) + shift:
        raise ValueError(
            f"128k length mismatch: len(ext)={len(ext_tok)} "
            f"len(base)+shift={len(base_tok) + shift}"
        )
    if expected_eos is not None and ext_sp["eos"] != expected_eos:
        raise ValueError(
            f"extended eos_token_id={ext_sp['eos']}, expected {expected_eos}"
        )
    return insert_at, shift


def _mean_pool_new_rows(
    model, base_tok, ext_tok, extension_start: int, extension_end: int | None = None
) -> int:
    embeddings = model.get_input_embeddings().weight.data
    old_embeddings = embeddings[:extension_start].clone()
    end = len(ext_tok) if extension_end is None else extension_end
    missing = 0
    with torch.no_grad():
        for token, token_id in ext_tok.get_vocab().items():
            if token_id < extension_start or token_id >= end:
                continue
            text = ext_tok.convert_tokens_to_string([token])
            subtokens = [
                token_id
                for token_id in base_tok.encode(text, add_special_tokens=False)
                if token_id < extension_start
            ]
            if not subtokens:
                missing += 1
                continue
            embeddings[token_id] = old_embeddings[subtokens].mean(dim=0)
    return missing


def _copy_source_rows_if_compatible(
    model, source_model_id: str | None, extension_start: int, target_size: int, dtype
) -> bool:
    if not source_model_id:
        return False

    source = AutoModelForCausalLM.from_pretrained(
        source_model_id,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    source_weight = source.get_input_embeddings().weight.data
    target_weight = model.get_input_embeddings().weight.data
    compatible = (
        source_weight.shape[0] >= target_size
        and source_weight.shape[1] == target_weight.shape[1]
    )
    if compatible:
        with torch.no_grad():
            target_weight[extension_start:target_size].copy_(
                source_weight[extension_start:target_size]
            )
    del source
    return compatible


def embedding_rows_after_extension(old_rows: int, tokenizer_rows: int) -> int:
    """Preserve padded embedding capacity while covering every tokenizer ID."""
    if old_rows <= 0 or tokenizer_rows <= 0:
        raise ValueError("embedding and tokenizer row counts must be positive")
    return max(old_rows, tokenizer_rows)


def semantic_prefix_preserved(
    old_weight: torch.Tensor,
    new_weight: torch.Tensor,
    semantic_rows: int,
) -> bool:
    """Ignore overwritten padding rows when checking append-only preservation."""
    if semantic_rows < 0 or semantic_rows > old_weight.shape[0]:
        raise ValueError("semantic row count is outside the original embedding")
    return torch.equal(
        new_weight[:semantic_rows],
        old_weight[:semantic_rows],
    )


def run(args: argparse.Namespace) -> int:
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    base_tok = AutoTokenizer.from_pretrained(args.base_id)
    ext_tok = AutoTokenizer.from_pretrained(args.source_tokenizer)
    extension_start, special_shift = _verify_compatible_base(
        base_tok, ext_tok, args.expected_eos
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    tied = lire_tie_flag(model.config)
    old_weight = model.get_input_embeddings().weight.data.clone()
    old_rows = old_weight.shape[0]
    target_rows = embedding_rows_after_extension(old_rows, len(ext_tok))
    if special_shift:
        target_rows = max(target_rows, old_rows + special_shift)
    if target_rows != old_rows:
        model.resize_token_embeddings(target_rows)

    new_weight = model.get_input_embeddings().weight.data
    missing = 0
    if special_shift:
        initialization = "128k_insert_before_specials_permute_mean_pool"
        V, N = extension_start, special_shift
        with torch.no_grad():
            move = min(old_rows, target_rows - N) - V
            if move > 0:
                new_weight[V + N : V + N + move] = old_weight[V : V + move]
            if N > 0:
                new_weight[V:V + N] = 0
        missing = _mean_pool_new_rows(model, base_tok, ext_tok, extension_start, extension_start + special_shift)
        if not semantic_prefix_preserved(old_weight, new_weight, extension_start):
            raise RuntimeError("resize changed original target embedding rows")
        eos = model.config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        if eos is not None:
            model.config.eos_token_id = int(eos) + special_shift
        if model.config.bos_token_id is not None:
            model.config.bos_token_id = int(model.config.bos_token_id) + special_shift
        if model.config.pad_token_id is not None:
            model.config.pad_token_id = int(model.config.pad_token_id) + special_shift
    else:
        copied = _copy_source_rows_if_compatible(
            model,
            args.source_model,
            extension_start,
            len(ext_tok),
            dtype,
        )
        initialization = "copied_source_extension_rows"
        if not copied:
            initialization = "mean_pool_target_base_subtokens"
            missing = _mean_pool_new_rows(model, base_tok, ext_tok, extension_start)
        if not semantic_prefix_preserved(old_weight, new_weight, extension_start):
            raise RuntimeError("resize changed original target embedding rows")

    ratio = (
        new_weight[extension_start:len(ext_tok)].norm(dim=1).mean().item()
        / old_weight[:extension_start].norm(dim=1).mean().item()
    )

    model.config.vocab_size = target_rows
    ecrire_tie_flag(model.config, tied)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16)
    model.save_pretrained(output_dir, safe_serialization=True)
    ext_tok.save_pretrained(output_dir)

    metadata = {
        "target_base_id": args.base_id,
        "source_tokenizer": args.source_tokenizer,
        "source_model": args.source_model,
        "vocab_before": extension_start,
        "vocab_after": len(ext_tok),
        "embedding_rows_before": old_weight.shape[0],
        "embedding_rows_after": target_rows,
        "n_added": special_shift if special_shift else (len(ext_tok) - extension_start),
        "special_shift": special_shift,
        "special_ids": _special_ids(ext_tok),
        "tie_embeddings": tied,
        "initialization": initialization,
        "tokens_without_subtokens": missing,
        "new_to_old_mean_norm_ratio": ratio,
        "norm_recalibration": False,
    }
    (output_dir / "resize_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Wrote resized init model and tokenizer to {output_dir}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
