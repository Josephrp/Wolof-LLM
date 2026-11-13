#!/usr/bin/env python3
"""Port the shared Wolof-extended tokenizer onto another Qwen3.5 Base size.

Applies the same insert-before-specials embedding shift + mean-pool as
``02_build_tokenizer.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resize Qwen Base to extended Wolof tokenizer")
    parser.add_argument("--base-id", required=True)
    parser.add_argument(
        "--tokenizer",
        default="tokenizers/qwen35-wolof-ext-anchors",
        help="Extended tokenizer directory from 02_build_tokenizer.py",
    )
    parser.add_argument("--output", required=True, help="Output init model directory")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


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

    base_tok = AutoTokenizer.from_pretrained(args.base_id, token=token, trust_remote_code=True)
    ext_tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    meta_path = tok_path / "extension_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    v_start = int(meta.get("V_append_start", meta.get("V", 248044)))
    n_words = int(meta.get("n_words", meta.get("N_new", len(ext_tok) - len(base_tok))))
    special_shift = int(meta.get("init", {}).get("special_shift") or meta.get("special_shift") or n_words)
    n_anchors = int(meta.get("n_anchors") or len(meta.get("anchors_added") or []))
    expected_eos = int(meta.get("eos_token_id") or ext_tok.eos_token_id)

    if ext_tok.eos_token_id != expected_eos:
        raise SystemExit(
            f"tokenizer eos {ext_tok.eos_token_id} != meta eos {expected_eos}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        token=token,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    emb_old = model.get_input_embeddings().weight.data.clone()
    old_rows = emb_old.shape[0]
    target = max(old_rows + n_words + n_anchors, len(ext_tok))
    model.resize_token_embeddings(target)
    emb = model.get_input_embeddings().weight.data
    sans = 0
    with torch.no_grad():
        if special_shift:
            move = min(old_rows, target - n_words) - v_start
            if move > 0:
                emb[v_start + n_words : v_start + n_words + move] = emb_old[
                    v_start : v_start + move
                ]
            if n_words > 0:
                emb[v_start : v_start + n_words] = 0
        for tok_str, nid in ext_tok.get_vocab().items():
            if not (v_start <= nid < v_start + n_words):
                continue
            texte = ext_tok.convert_tokens_to_string([tok_str])
            sub = [
                sid
                for sid in base_tok.encode(texte, add_special_tokens=False)
                if sid < min(v_start, old_rows)
            ]
            if not sub:
                sans += 1
                continue
            emb[nid] = emb_old[sub].mean(dim=0)
        for special in meta.get("anchors_added") or []:
            nid = ext_tok.convert_tokens_to_ids(special)
            if isinstance(nid, int) and 0 <= nid < emb.shape[0]:
                emb[nid] = emb_old[: min(v_start, old_rows)].mean(dim=0)

    def _shift_id(value):
        if value is None:
            return None
        if isinstance(value, list):
            return [int(v) + special_shift for v in value]
        return int(value) + special_shift

    if special_shift:
        model.config.eos_token_id = _shift_id(model.config.eos_token_id)
        if getattr(model.config, "pad_token_id", None) is not None:
            model.config.pad_token_id = _shift_id(model.config.pad_token_id)
        if getattr(model.config, "bos_token_id", None) is not None:
            model.config.bos_token_id = _shift_id(model.config.bos_token_id)

    eos = model.config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    if eos != expected_eos:
        raise SystemExit(f"model eos {eos} != expected {expected_eos}")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    ext_tok.save_pretrained(out)
    report = {
        "created_at": utc_now(),
        "base_id": args.base_id,
        "tokenizer": str(tok_path),
        "output": str(out),
        "old_rows": old_rows,
        "new_rows": target,
        "V": v_start,
        "N": n_words,
        "special_shift": special_shift,
        "sans_subtokens": sans,
        "eos_token_id": ext_tok.eos_token_id,
    }
    atomic_json_write(out / "resize_meta.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
