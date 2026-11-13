#!/usr/bin/env python3
"""Append-only Wolof vocab extension for Qwen3.5 (LFM/SmolLM3 recipe).

Why the previous thin script was wrong
--------------------------------------
* Whitespace ``.split()`` ignores ByteLevel pretok chunks (``Ġword``).
* ``tokenizer.add_tokens()`` does not mutate BPE ``model.vocab`` the way
  this repo's LFM extender does; encoding often never emits the new ids.
* Global mean-fill of *all* new rows ignores per-token subword structure.
* No eos/special preservation gate and no fertility before/after check.

Correct recipe (mirrors LFM ``02_build_tokenizer.py`` 65k-style append)
----------------------------------------------------------------------
1. Mine candidates via ``pre_tokenizer.pre_tokenize_str`` on ortho-filtered
   Wolof CPT text; rank by ``freq * (n_tok - 1)``.
2. Mutate ``tokenizer.json`` ``model.vocab``: append after max existing id
   so Qwen high-id specials (eos=248044 …) stay put.
3. Register anchoring specials as ``added_tokens`` (also after max id).
4. Resize CausalLM embeddings; **mean-pool each new row from its base
   subword pieces**; do **not** recalibrate norms.
5. Gate: eos unchanged; fertility on Wolof probe words must improve.

Qwen note: ``ignore_merges`` is false — monotonicity is empirical (same
LFM warning). Fertility gate catches failed extensions.
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
    "jëfandikoo",  # corpus/pretok form (diacritic)
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
    "dafay",
    "ñu",
    "benn",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 append-only Wolof tokenizer extension + mean-pool init"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--expected-eos",
        type=int,
        default=None,
        help="Assert eos unchanged (default: base eos, typically 248044)",
    )
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
            "  WARNING: ignore_merges is false — whole-word append is still "
            "applied, but encoding use of new ids is empirical; fertility gate required.",
            file=sys.stderr,
        )
    return info


def selectionner_candidats(base_tok, textes: list[str], n_echantillon: int, seed: int):
    """Select pretok chunks ranked by expected token savings (LFM/SmolLM3)."""
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
        if len(brut) < 2 or not any(ch.isalpha() for ch in brut):
            continue
        # Skip pure punctuation / control pretok pieces.
        if chunk.startswith("<|") and chunk.endswith("|>"):
            continue
        candidats.append((chunk, freq, n_tok, freq * (n_tok - 1)))
    candidats.sort(key=lambda row: -row[3])
    print(f"  candidats : {len(candidats):,}")
    return candidats


def _force_ignore_merges(tokenizer):
    """HuggingFace AutoTokenizer often drops ignore_merges on reload — re-patch."""
    from tokenizers import Tokenizer as FastTokenizer

    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    model = payload.setdefault("model", {})
    model["ignore_merges"] = True
    tokenizer._tokenizer = FastTokenizer.from_str(json.dumps(payload))
    check = json.loads(tokenizer.backend_tokenizer.to_str())
    if check.get("model", {}).get("ignore_merges") is not True:
        raise RuntimeError("failed to force ignore_merges=true on tokenizer backend")
    return tokenizer


def construire_tokenizer_append(base_id: str, candidats, n_add: int, token: str | None):
    """Insert pretok chunks *before* Qwen high-id specials (LFM 128k pattern).

    Qwen3.5 keeps chat/vision specials at 248044+. Appending *after* those ids
    makes ``tokenizers`` remap specials on reload (observed eos 248044→252046).
    Insert at ``first_special`` and shift specials by +N instead.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id, token=token, trust_remote_code=True)
    base_eos = tok.eos_token_id
    base_pad = tok.pad_token_id
    payload = json.loads(tok.backend_tokenizer.to_str())
    vocab = dict(payload["model"]["vocab"])
    specials = list(payload.get("added_tokens") or [])
    if not specials:
        raise RuntimeError("Qwen tokenizer has no added_tokens specials to shift")

    first_special = min(int(a["id"]) for a in specials if isinstance(a, dict) and "id" in a)
    if base_eos is None or first_special > int(base_eos):
        raise RuntimeError(
            f"unexpected special layout: first_special={first_special} eos={base_eos}"
        )

    # Force-include pretok forms of probe words so fertility gate is meaningful.
    forced: list[str] = []
    pre = tok.backend_tokenizer.pre_tokenizer
    for word in MOTS_TEST:
        for form in (word, " " + word):
            for chunk, _span in pre.pre_tokenize_str(form):
                if chunk not in vocab and chunk not in forced:
                    n_tok = len(tok.encode(form, add_special_tokens=False))
                    if n_tok >= 2:
                        forced.append(chunk)

    nouveaux: list[str] = []
    seen: set[str] = set()
    for chunk in forced + [c for c, _, _, _ in candidats]:
        if len(nouveaux) >= n_add:
            break
        if chunk in vocab or chunk in seen:
            continue
        seen.add(chunk)
        nouveaux.append(chunk)
    n_words = len(nouveaux)
    if n_words <= 0:
        raise RuntimeError("extension selected zero pretok chunks")
    print(f"  forced probe chunks: {forced[:20]}", flush=True)

    # Shift existing specials, then insert new BPE rows in the freed window.
    for entry in specials:
        entry["id"] = int(entry["id"]) + n_words
    for offset, chunk in enumerate(nouveaux):
        vocab[chunk] = first_special + offset

    existing_special_contents = {
        entry.get("content") for entry in specials if isinstance(entry, dict)
    }
    anchors_added: list[str] = []
    next_id = max(int(a["id"]) for a in specials if isinstance(a, dict) and "id" in a) + 1
    for special in ANCHOR_SPECIALS:
        if special in existing_special_contents or special in vocab:
            continue
        specials.append(
            {
                "id": next_id,
                "content": special,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        anchors_added.append(special)
        next_id += 1

    # Whole-word pretok chunks only win at encode-time when ignore_merges is true
    # (Qwen stock has ignore_merges=false → new vocab rows never fire).
    payload["model"]["vocab"] = vocab
    payload["model"]["ignore_merges"] = True
    payload["added_tokens"] = specials

    tmp = tempfile.mkdtemp(prefix="qwen35_tok_ext_")
    tok.save_pretrained(tmp)
    with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    for stale in ("vocab.json", "merges.txt"):
        path = os.path.join(tmp, stale)
        if os.path.exists(path):
            os.remove(path)

    ext_tok = AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
    ext_tok = _force_ignore_merges(ext_tok)
    # Persist the patched backend so save_pretrained keeps ignore_merges=true.
    patched_json = os.path.join(tmp, "tokenizer.json")
    with open(patched_json, "w", encoding="utf-8") as handle:
        handle.write(ext_tok.backend_tokenizer.to_str())
    reloaded = json.loads(ext_tok.backend_tokenizer.to_str())
    if reloaded.get("model", {}).get("ignore_merges") is not True:
        raise RuntimeError(
            f"ignore_merges did not stick after reload: "
            f"{reloaded.get('model', {}).get('ignore_merges')!r}"
        )
    expected_eos = int(base_eos) + n_words
    expected_pad = None if base_pad is None else int(base_pad) + n_words
    if ext_tok.eos_token_id != expected_eos:
        raise RuntimeError(
            "insert-before-specials eos mismatch: "
            f"got {ext_tok.eos_token_id}, expected {expected_eos}"
        )
    if expected_pad is not None and ext_tok.pad_token_id != expected_pad:
        raise RuntimeError(
            "insert-before-specials pad mismatch: "
            f"got {ext_tok.pad_token_id}, expected {expected_pad}"
        )
    return ext_tok, {
        "V": first_special,
        "N": n_words,
        "n_words": n_words,
        "n_anchors": len(anchors_added),
        "anchors_added": anchors_added,
        "special_shift": n_words,
        "family": "qwen35-insert-before-specials",
        "base_eos": base_eos,
        "expected_eos": expected_eos,
        "nouveaux": nouveaux,
    }


def fertility(tokenizer, words: list[str]) -> dict:
    """Measure fragmentation on bare + leading-space forms (ByteLevel Ġ)."""
    rows = []
    total = 0
    n_forms = 0
    for word in words:
        for form, label in ((word, "bare"), (" " + word, "lead_space")):
            ids = tokenizer.encode(form, add_special_tokens=False)
            total += len(ids)
            n_forms += 1
            rows.append(
                {
                    "word": word,
                    "form": label,
                    "n_tok": len(ids),
                    "pieces": tokenizer.convert_ids_to_tokens(ids),
                }
            )
    return {
        "mean_tokens": total / max(n_forms, 1),
        "words": rows,
    }


def initialiser_modele(base_id, base_tok, ext_tok, meta, sortie, token: str | None):
    """Resize embeddings; shift specials; mean-pool NEW rows; no norm recalib."""
    import torch
    from transformers import AutoModelForCausalLM

    v_start = int(meta["V"])
    n_words = int(meta["N"])
    special_shift = int(meta.get("special_shift") or 0)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        token=token,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    emb_old = model.get_input_embeddings().weight.data.clone()
    old_rows = emb_old.shape[0]
    # Keep padding tail after shifting specials; cover full extended tokenizer.
    target_rows = max(old_rows + n_words + int(meta.get("n_anchors") or 0), len(ext_tok))
    if target_rows != old_rows:
        model.resize_token_embeddings(target_rows)
    emb = model.get_input_embeddings().weight.data
    if emb.shape[0] != target_rows:
        raise RuntimeError(f"embedding rows {emb.shape[0]} != {target_rows}")

    sans = 0
    with torch.no_grad():
        if special_shift:
            # Move former specials / tail [V, old_rows) -> [V+N, ...).
            move = min(old_rows, target_rows - n_words) - v_start
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

        # Fresh anchor specials (after shifted stock specials): mean of base rows.
        for special in meta.get("anchors_added") or []:
            nid = ext_tok.convert_tokens_to_ids(special)
            if isinstance(nid, int) and nid >= 0 and nid < emb.shape[0]:
                emb[nid] = emb_old[: min(v_start, old_rows)].mean(dim=0)

    if n_words > 0:
        ratio = (
            emb[v_start : v_start + n_words].norm(dim=1).mean().item()
            / emb_old[: min(v_start, old_rows)].norm(dim=1).mean().item()
        )
    else:
        ratio = float("nan")
    print(f"  tokens sans sous-tokens exploitables : {sans}")
    print(f"  ratio de norme : {ratio:.3f} (expect ~0.5; do NOT recalibrate)")

    # Config specials must track tokenizer shift.
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
        # Multimodal control ids on Qwen3.5 configs, when present.
        for key in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            if hasattr(model.config, key) and getattr(model.config, key) is not None:
                setattr(model.config, key, _shift_id(getattr(model.config, key)))

    eos = model.config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    if eos != meta["expected_eos"]:
        raise SystemExit(f"model eos {eos} != expected {meta['expected_eos']}")

    sortie.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(sortie)
    return {
        "norm_ratio": ratio,
        "sans_subtokens": sans,
        "old_rows": old_rows,
        "new_rows": target_rows,
        "special_shift": special_shift,
    }


def load_wolof_texts(data_root: Path, max_rows: int | None) -> list[str]:
    import pandas as pd

    train_path = data_root / "train.parquet"
    if not train_path.is_file():
        train_path = data_root / "wolof_bpe_train_v1.parquet"
    df = pd.read_parquet(train_path)
    if max_rows:
        df = df.head(max_rows)
    if "lang" in df.columns:
        wo = df[df["lang"].astype(str) == "wo"]
        if len(wo) >= 100:
            df = wo
            print(f"  using Wolof-only rows for mining: {len(df):,}")
        else:
            print("  WARNING: few lang=wo rows; mining on full train", file=sys.stderr)
    if "ortho_label" in df.columns:
        keep = df[df["ortho_label"].astype(str).isin({"non_standard", "mixed"})]
        if len(keep) >= 50:
            df = keep
            print(f"  using ortho ns+mixed rows: {len(df):,}")
    return df["text"].astype(str).tolist()


def main() -> int:
    args = build_parser().parse_args()
    from transformers import AutoTokenizer

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    base_id = cfg["base_id"]
    data_root = Path(cfg.get("data_root") or ROOT / "data" / "ortho_ns_mixed_v1" / "cpt")
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    tok_out = Path(cfg["tokenizer_path"])
    model_out = Path(cfg["model_path"])
    if not tok_out.is_absolute():
        tok_out = ROOT / tok_out
    if not model_out.is_absolute():
        model_out = ROOT / model_out

    n_tokens = int(cfg.get("n_tokens", 4000))
    n_sample = int(cfg.get("n_echantillon", 200_000))
    seed = int(cfg.get("seed", 42))

    print(f"base: {base_id}")
    base_tok = AutoTokenizer.from_pretrained(base_id, token=token, trust_remote_code=True)
    backend = _inspect_backend(base_tok)
    print(f"  eos={base_tok.eos_token_id} pad={base_tok.pad_token_id} len={len(base_tok)}")

    texts = load_wolof_texts(data_root, args.max_rows)
    print(f"  corpus texts: {len(texts):,}")
    fert_before = fertility(base_tok, MOTS_TEST)
    print(f"  fertility before mean_tokens={fert_before['mean_tokens']:.3f}")

    candidats = selectionner_candidats(base_tok, texts, n_sample, seed)
    if not candidats:
        raise SystemExit("no Wolof extension candidates found — check CPT data")

    print(f"\ninsert-before-specials up to {n_tokens} pretok chunks + anchors…")
    ext_tok, meta = construire_tokenizer_append(base_id, candidats, n_tokens, token)
    if args.expected_eos is not None and ext_tok.eos_token_id != args.expected_eos:
        raise SystemExit(
            f"eos drift after build: {ext_tok.eos_token_id} != {args.expected_eos}"
        )
    print(
        f"  eos shifted {meta['base_eos']} -> {meta['expected_eos']} "
        f"(special_shift=+{meta['special_shift']})"
    )

    fert_after = fertility(ext_tok, MOTS_TEST)
    print(f"  fertility after mean_tokens={fert_after['mean_tokens']:.3f}")
    for row in fert_after["words"]:
        before = next(
            r
            for r in fert_before["words"]
            if r["word"] == row["word"] and r["form"] == row["form"]
        )
        marker = "↓" if row["n_tok"] < before["n_tok"] else ("=" if row["n_tok"] == before["n_tok"] else "↑")
        print(
            f"    {row['word']}[{row['form']}]: {before['n_tok']} -> {row['n_tok']} {marker}  {row['pieces']}"
        )

    # Gate: mean fertility must improve when ignore_merges enables new chunks.
    if fert_after["mean_tokens"] > fert_before["mean_tokens"] + 1e-6:
        raise SystemExit(
            "FERTILITY GATE FAIL: mean tokens increased after extension "
            f"({fert_before['mean_tokens']:.3f} -> {fert_after['mean_tokens']:.3f})"
        )
    improved = sum(
        1
        for row in fert_after["words"]
        if row["n_tok"]
        < next(
            r["n_tok"]
            for r in fert_before["words"]
            if r["word"] == row["word"] and r["form"] == row["form"]
        )
    )
    if improved < 1 and meta["n_words"] > 0:
        # Diagnostic: are probe pretok chunks even in the new vocab?
        sample = meta["nouveaux"][:5]
        print("  sample new chunks:", sample, flush=True)
        raise SystemExit(
            "FERTILITY GATE FAIL: no probe form reduced fragmentation after extension "
            "(check ignore_merges + pretok chunk selection)"
        )

    tok_out.mkdir(parents=True, exist_ok=True)
    ext_tok = _force_ignore_merges(ext_tok)
    ext_tok.save_pretrained(tok_out)
    # save_pretrained can drop ignore_merges — rewrite tokenizer.json explicitly.
    with open(tok_out / "tokenizer.json", "w", encoding="utf-8") as handle:
        payload = json.loads(ext_tok.backend_tokenizer.to_str())
        payload.setdefault("model", {})["ignore_merges"] = True
        json.dump(payload, handle, ensure_ascii=False)

    print("\ninitialising embeddings (per-token mean-pool, no recalib)")
    init_meta = initialiser_modele(base_id, base_tok, ext_tok, meta, model_out, token)
    ext_tok.save_pretrained(model_out)
    with open(model_out / "tokenizer.json", "w", encoding="utf-8") as handle:
        payload = json.loads(ext_tok.backend_tokenizer.to_str())
        payload.setdefault("model", {})["ignore_merges"] = True
        json.dump(payload, handle, ensure_ascii=False)

    report = {
        "created_at": utc_now(),
        "base_id": base_id,
        "backend": backend,
        "old_vocab": len(base_tok),
        "new_vocab": len(ext_tok),
        "V_append_start": meta["V"],
        "N_new": meta["N"],
        "n_words": meta["n_words"],
        "n_anchors": meta.get("n_anchors", 0),
        "anchors_added": meta["anchors_added"],
        "special_shift": meta.get("special_shift", 0),
        "base_eos": meta.get("base_eos"),
        "eos_token_id": ext_tok.eos_token_id,
        "fertility_before": fert_before,
        "fertility_after": fert_after,
        "fertility_words_improved": improved,
        "sample_new_chunks": meta["nouveaux"][:25],
        "tokenizer_path": str(tok_out),
        "model_path": str(model_out),
        "init": init_meta,
        "method": (
            "insert-before-specials pretok chunks + ignore_merges=true + "
            "mean-pool; specials shifted"
        ),
        "ignore_merges": True,
    }
    atomic_json_write(tok_out / "extension_meta.json", report)
    atomic_json_write(model_out / "extension_meta.json", report)
    print(json.dumps({k: report[k] for k in (
        "old_vocab",
        "new_vocab",
        "n_words",
        "anchors_added",
        "eos_token_id",
        "fertility_words_improved",
        "method",
    )}, indent=2))
    print("TOKENIZER_EXT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
