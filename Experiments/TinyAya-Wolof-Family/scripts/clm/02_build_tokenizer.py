#!/usr/bin/env python3
"""Append-only Wolof vocab extension for Cohere Tiny Aya (LFM 65k-style).

Critical vs Qwen path
---------------------
Tiny Aya keeps bos=2 / eos=3 / pad=0 at low ids (END_OF_TURN may be id 6).
We APPEND after max(existing id)+1 — never reindex stock specials.

Recipe
------
1. Mine pretok chunks on ortho ns+mixed Wolof CPT; rank by freq*(n_tok-1).
2. Mutate tokenizer.json model.vocab: append new rows after max id.
3. Register <|lang_wo|> / <|ortho_ns|> as added_tokens after max id.
4. Resize CausalLM (tied embeddings); mean-pool new rows; no norm recalib.
5. Assert bos/eos/pad unchanged; fertility must not worsen.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.anchoring import ANCHOR_SPECIALS
from common.pipeline import atomic_json_write, utc_now

MOTS_TEST = [
    "jefandikoo",
    "wooral",
    "cakuteef",
    "jewrinu",
    "reew",
    "waaw",
    "ndaxte",
    "buur",
    "xamxam",
    "jàmm",
    "jamm",
    "mbokk",
    "teranga",
    "ndank",
]

EXPECTED_BOS = 2
EXPECTED_EOS = 3
EXPECTED_PAD = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiny Aya append-only Wolof tokenizer extension + mean-pool init"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--expected-bos", type=int, default=EXPECTED_BOS)
    parser.add_argument("--expected-eos", type=int, default=EXPECTED_EOS)
    parser.add_argument("--expected-pad", type=int, default=EXPECTED_PAD)
    return parser


def _inspect_backend(base_tok) -> dict:
    payload = json.loads(base_tok.backend_tokenizer.to_str())
    model = payload.get("model") or {}
    info = {
        "model_type": model.get("type", "unknown"),
        "ignore_merges": model.get("ignore_merges"),
    }
    print(f"  tokenizer backend model.type = {info['model_type']!r}")
    print(f"  tokenizer backend ignore_merges = {info['ignore_merges']!r}")
    if info["ignore_merges"] is False:
        print(
            "  WARNING: ignore_merges is false — fertility gate required.",
            file=sys.stderr,
        )
    return info


def assert_specials(tokenizer, *, bos: int, eos: int, pad: int, label: str) -> None:
    got = {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
    }
    expected = {"bos": bos, "eos": eos, "pad": pad}
    if got != expected:
        raise SystemExit(f"{label} specials mismatch: got {got}, expected {expected}")


def selectionner_candidats(base_tok, textes: list[str], n_echantillon: int, seed: int):
    random.seed(seed)
    echantillon = random.sample(textes, min(n_echantillon, len(textes)))
    pre = base_tok.backend_tokenizer.pre_tokenizer
    cnt: Counter[str] = Counter()
    for text in echantillon:
        for chunk, _span in pre.pre_tokenize_str(str(text)):
            cnt[chunk] += 1
    print(f"  segments distincts : {len(cnt):,}")

    base_vocab = set(base_tok.get_vocab())
    decoder = base_tok.backend_tokenizer.decoder
    candidats = []
    for chunk, freq in cnt.most_common():
        if chunk in base_vocab:
            continue
        texte = decoder.decode([chunk])
        n_tok = len(base_tok.encode(texte, add_special_tokens=False))
        if n_tok < 2:
            continue
        brut = texte.strip()
        if len(brut) < 3 or not any(ch.isalpha() for ch in brut):
            continue
        if chunk.startswith("<|") and chunk.endswith("|>"):
            continue
        candidats.append((chunk, freq, n_tok, freq * (n_tok - 1)))
    candidats.sort(key=lambda row: -row[3])
    print(f"  candidats : {len(candidats):,}")
    return candidats


def seed_probe_candidates(base_tok, candidats: list) -> list:
    """Prefer pretok chunks from MOTS_TEST so fertility probes can improve."""
    pre = base_tok.backend_tokenizer.pre_tokenizer
    decoder = base_tok.backend_tokenizer.decoder
    base_vocab = set(base_tok.get_vocab())
    seen = {row[0] for row in candidats}
    seeded: list = []
    for word in MOTS_TEST:
        for chunk, _span in pre.pre_tokenize_str(word):
            if chunk in seen or chunk in base_vocab:
                continue
            texte = decoder.decode([chunk])
            n_tok = len(base_tok.encode(texte, add_special_tokens=False))
            if n_tok < 2:
                continue
            row = (chunk, 10_000, n_tok, 10_000 * (n_tok - 1))
            seeded.append(row)
            seen.add(chunk)
    if seeded:
        print(f"  seeded probe pretok chunks: {len(seeded)}")
    return seeded + candidats


def construire_tokenizer_append(base_id: str, candidats, n_add: int, token: str | None):
    """True append-only: leave stock specials at original ids; append after max id."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id, token=token, trust_remote_code=True)
    base_bos, base_eos, base_pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    payload = json.loads(tok.backend_tokenizer.to_str())
    vocab = dict(payload["model"]["vocab"])
    specials = list(payload.get("added_tokens") or [])

    max_id = max(vocab.values()) if vocab else -1
    for entry in specials:
        if isinstance(entry, dict) and "id" in entry:
            max_id = max(max_id, int(entry["id"]))
    max_id = max(max_id, len(tok) - 1)
    next_id = max_id + 1
    v_start = next_id

    nouveaux: list[str] = []
    for chunk, _freq, _ntok, _gain in candidats:
        if len(nouveaux) >= n_add:
            break
        if chunk in vocab:
            continue
        vocab[chunk] = next_id + len(nouveaux)
        nouveaux.append(chunk)
    n_words = len(nouveaux)
    if n_words <= 0:
        raise RuntimeError("extension selected zero pretok chunks")

    existing_special_contents = {
        entry.get("content") for entry in specials if isinstance(entry, dict)
    }
    anchors_added: list[str] = []
    anchor_next = v_start + n_words
    for special in ANCHOR_SPECIALS:
        if special in existing_special_contents or special in vocab:
            continue
        specials.append(
            {
                "id": anchor_next,
                "content": special,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        anchors_added.append(special)
        anchor_next += 1

    # Cohere BPE ships with ignore_merges=false, so appended whole-chunk vocab
    # rows are never selected. Flip to true on the *extended* tokenizer only so
    # pretok chunks present in vocab encode as single tokens (LFM-style gate).
    payload["model"]["vocab"] = vocab
    payload["model"]["ignore_merges"] = True
    payload["added_tokens"] = specials
    print("  extended tokenizer: set ignore_merges=True for append-only fertility")

    tmp = tempfile.mkdtemp(prefix="tinyaya_tok_ext_")
    tok.save_pretrained(tmp)
    with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    for stale in ("vocab.json", "merges.txt"):
        path = os.path.join(tmp, stale)
        if os.path.exists(path):
            os.remove(path)

    ext_tok = AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
    if (
        ext_tok.bos_token_id != base_bos
        or ext_tok.eos_token_id != base_eos
        or ext_tok.pad_token_id != base_pad
    ):
        raise RuntimeError(
            "append-only specials drifted: "
            f"bos {base_bos}->{ext_tok.bos_token_id} "
            f"eos {base_eos}->{ext_tok.eos_token_id} "
            f"pad {base_pad}->{ext_tok.pad_token_id}"
        )
    return ext_tok, {
        "V": v_start,
        "N": n_words,
        "n_words": n_words,
        "n_anchors": len(anchors_added),
        "anchors_added": anchors_added,
        "special_shift": 0,
        "family": "tinyaya-append-only",
        "base_bos": base_bos,
        "base_eos": base_eos,
        "base_pad": base_pad,
        "nouveaux": nouveaux,
    }


def fertility(tokenizer, words: list[str]) -> dict:
    rows = []
    total = 0
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        total += len(ids)
        rows.append(
            {
                "word": word,
                "n_tok": len(ids),
                "pieces": tokenizer.convert_ids_to_tokens(ids),
            }
        )
    return {"mean_tokens": total / max(len(words), 1), "words": rows}


def initialiser_modele(base_id, base_tok, ext_tok, meta, sortie, token: str | None):
    """Resize tied embeddings; mean-pool NEW rows only; never shift specials."""
    import torch
    from transformers import AutoModelForCausalLM

    v_start = int(meta["V"])
    n_words = int(meta["N"])
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        token=token,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    emb_old = model.get_input_embeddings().weight.data.clone()
    old_rows = emb_old.shape[0]
    target_rows = max(old_rows, len(ext_tok))
    if target_rows != old_rows:
        model.resize_token_embeddings(target_rows)
    emb = model.get_input_embeddings().weight.data
    if emb.shape[0] != target_rows:
        raise RuntimeError(f"embedding rows {emb.shape[0]} != {target_rows}")

    sans = 0
    with torch.no_grad():
        for tok_str, nid in ext_tok.get_vocab().items():
            if nid < v_start:
                continue
            texte = ext_tok.convert_tokens_to_string([tok_str])
            sub = [
                sid
                for sid in base_tok.encode(texte, add_special_tokens=False)
                if sid < old_rows
            ]
            if not sub:
                # Anchor / OOV without subpieces: global mean of old rows.
                emb[nid] = emb_old.mean(dim=0)
                sans += 1
                continue
            emb[nid] = emb_old[sub].mean(dim=0)

    if n_words > 0:
        ratio = (
            emb[v_start : v_start + n_words].norm(dim=1).mean().item()
            / emb_old.norm(dim=1).mean().item()
        )
    else:
        ratio = float("nan")
    print(f"  tokens sans sous-tokens exploitables : {sans}")
    print(f"  ratio de norme : {ratio:.3f} (expect ~0.5; do NOT recalibrate)")

    # Preserve stock specials on config.
    model.config.bos_token_id = meta["base_bos"]
    model.config.eos_token_id = meta["base_eos"]
    model.config.pad_token_id = meta["base_pad"]
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = True

    sortie.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(sortie)
    return {
        "norm_ratio": ratio,
        "sans_subtokens": sans,
        "old_rows": old_rows,
        "new_rows": target_rows,
        "special_shift": 0,
    }


def load_wolof_texts(data_roots: list[Path], max_rows: int | None) -> list[str]:
    """Mine pretok candidates from one or more ortho CPT trains (ns + std)."""
    import pandas as pd

    frames = []
    for data_root in data_roots:
        train_path = data_root / "train.parquet"
        if not train_path.is_file():
            train_path = data_root / "wolof_bpe_train_v1.parquet"
        if not train_path.is_file():
            print(f"  WARNING: missing CPT train under {data_root}", file=sys.stderr)
            continue
        df = pd.read_parquet(train_path)
        if "lang" in df.columns:
            wo = df[df["lang"].astype(str) == "wo"]
            if len(wo) >= 50:
                df = wo
        if "ortho_label" in df.columns:
            keep = df[
                df["ortho_label"]
                .astype(str)
                .isin({"non_standard", "standard", "mixed"})
            ]
            if len(keep) >= 20:
                df = keep
        frames.append(df.assign(_src=str(data_root)))
        print(f"  mining corpus from {data_root}: {len(df):,} rows")
    if not frames:
        raise SystemExit("no CPT train parquet found for tokenizer mining")
    df = pd.concat(frames, ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    print(f"  combined mining rows: {len(df):,}")
    return df["text"].astype(str).tolist()


def main() -> int:
    args = build_parser().parse_args()
    from transformers import AutoTokenizer

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    # base_id = tokenizer source (earth shares 262k); model_id = CPT weight init (base).
    tokenizer_id = cfg.get("tokenizer_id") or cfg["base_id"]
    model_id = cfg.get("model_id") or cfg["base_id"]
    configured_roots = cfg.get("data_roots") or cfg.get("data_root")
    if configured_roots is None:
        configured_roots = [
            "data/ortho_ns_mixed_v1/cpt",
            "data/ortho_std_mixed_v1/cpt",
        ]
    if isinstance(configured_roots, str):
        configured_roots = [configured_roots]
    data_roots: list[Path] = []
    for item in configured_roots:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_dir():
            data_roots.append(path)
    if not data_roots:
        raise SystemExit("no tokenizer mining data_roots found on disk")

    tok_out = Path(cfg["tokenizer_path"])
    model_out = Path(cfg["model_path"])
    if not tok_out.is_absolute():
        tok_out = ROOT / tok_out
    if not model_out.is_absolute():
        model_out = ROOT / model_out

    n_tokens = int(cfg.get("n_tokens", 3000))
    n_sample = int(cfg.get("n_echantillon", 200_000))
    seed = int(cfg.get("seed", 42))
    expected_bos = int(cfg.get("expected_bos", args.expected_bos))
    expected_eos = int(cfg.get("expected_eos", args.expected_eos))
    expected_pad = int(cfg.get("expected_pad", args.expected_pad))

    print(f"tokenizer_id: {tokenizer_id}")
    print(f"model_id: {model_id}")
    base_tok = AutoTokenizer.from_pretrained(tokenizer_id, token=token, trust_remote_code=True)
    backend = _inspect_backend(base_tok)
    assert_specials(
        base_tok, bos=expected_bos, eos=expected_eos, pad=expected_pad, label="stock"
    )
    print(f"  eos={base_tok.eos_token_id} pad={base_tok.pad_token_id} len={len(base_tok)}")

    texts = load_wolof_texts(data_roots, args.max_rows)
    print(f"  corpus texts: {len(texts):,}")
    fert_before = fertility(base_tok, MOTS_TEST)
    print(f"  fertility before mean_tokens={fert_before['mean_tokens']:.3f}")

    candidats = selectionner_candidats(base_tok, texts, n_sample, seed)
    if not candidats:
        raise SystemExit("no Wolof extension candidates found — check CPT data")
    candidats = seed_probe_candidates(base_tok, candidats)

    print(f"\nappend-only up to {n_tokens} pretok chunks + anchors…")
    ext_tok, meta = construire_tokenizer_append(tokenizer_id, candidats, n_tokens, token)
    assert_specials(
        ext_tok, bos=expected_bos, eos=expected_eos, pad=expected_pad, label="extended"
    )

    fert_after = fertility(ext_tok, MOTS_TEST)
    print(f"  fertility after mean_tokens={fert_after['mean_tokens']:.3f}")
    for row in fert_after["words"]:
        before = next(r for r in fert_before["words"] if r["word"] == row["word"])
        print(f"    {row['word']}: {before['n_tok']} -> {row['n_tok']}  {row['pieces']}")

    if fert_after["mean_tokens"] > fert_before["mean_tokens"] + 1e-6:
        raise SystemExit(
            "FERTILITY GATE FAIL: mean tokens increased after extension "
            f"({fert_before['mean_tokens']:.3f} -> {fert_after['mean_tokens']:.3f})"
        )
    improved = sum(
        1
        for row in fert_after["words"]
        if row["n_tok"]
        < next(r["n_tok"] for r in fert_before["words"] if r["word"] == row["word"])
    )
    if improved < 1 and meta["n_words"] > 0:
        raise SystemExit(
            "FERTILITY GATE FAIL: no probe word reduced fragmentation after extension"
        )

    tok_out.mkdir(parents=True, exist_ok=True)
    ext_tok.save_pretrained(tok_out)

    print("\ninitialising embeddings (per-token mean-pool, no recalib)")
    init_meta = initialiser_modele(model_id, base_tok, ext_tok, meta, model_out, token)
    ext_tok.save_pretrained(model_out)

    report = {
        "created_at": utc_now(),
        "tokenizer_id": tokenizer_id,
        "model_id": model_id,
        "base_id": tokenizer_id,
        "backend": backend,
        "old_vocab": len(base_tok),
        "new_vocab": len(ext_tok),
        "V_append_start": meta["V"],
        "N_new": meta["N"],
        "n_words": meta["n_words"],
        "n_anchors": meta.get("n_anchors", 0),
        "anchors_added": meta["anchors_added"],
        "special_shift": 0,
        "bos_token_id": ext_tok.bos_token_id,
        "eos_token_id": ext_tok.eos_token_id,
        "pad_token_id": ext_tok.pad_token_id,
        "fertility_before": fert_before,
        "fertility_after": fert_after,
        "fertility_words_improved": improved,
        "sample_new_chunks": meta["nouveaux"][:25],
        "nouveaux": meta["nouveaux"],
        "tokenizer_path": str(tok_out),
        "model_path": str(model_out),
        "init": init_meta,
        "method": "append-only pretok chunks + mean-pool; stock specials preserved",
        "tie_word_embeddings": True,
        "ignore_merges": True,
    }
    atomic_json_write(tok_out / "extension_meta.json", report)
    atomic_json_write(model_out / "extension_meta.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "old_vocab",
                    "new_vocab",
                    "n_words",
                    "anchors_added",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "fertility_words_improved",
                    "method",
                )
            },
            indent=2,
        )
    )
    print("TOKENIZER_EXT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
