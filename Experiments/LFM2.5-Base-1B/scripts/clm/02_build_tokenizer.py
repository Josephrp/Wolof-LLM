#!/usr/bin/env python3
"""Build LFM2.5 Wolof extended tokenizer + mean-pooled init model.

LFM deltas vs SmolLM3 (Experiments/SmolLM3-3B-Base/scripts/02_build_tokenizer.py):

  * APPEND-ONLY vocab extension — do NOT reindex specials.
  * Preserve eos/bos/pad (65k family: eos_token_id = 7, <|im_end|>).
  * Mean-pool new embedding rows; NO norm recalibration.
  * 128k family path is stubbed until specials layout is inspected.

Heavy deps (torch / transformers) are imported lazily so:

  uv run python scripts/clm/02_build_tokenizer.py --help

works without ``uv sync --group train``.

Examples::

  uv run python scripts/clm/02_build_tokenizer.py \\
      --config configs/tokenizer/tokenizer-ext4000-65k.yaml
  uv run python scripts/clm/02_build_tokenizer.py \\
      --config configs/tokenizer-ext4000-65k.yaml --max-rows 500
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

try:
    import torch
    from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AddedToken = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.tracking import TrackingRun

MOTS_TEST = ["jefandikoo", "new", "wooral", "cakuteef", "jewrinu", "reew", "waaw"]
EXPECTED_EOS_65K = 7
EXPECTED_SPECIALS_128K = {"pad": 124893, "bos": 124894, "eos": 124900}


# ---------------------------------------------------------------------------
# Argparse (no heavy imports)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LFM2.5 append-only Wolof tokenizer extension + mean-pool init "
            "(no special reindex)."
        )
    )
    p.add_argument(
        "--config",
        default=None,
        help=(
            "YAML config, e.g. configs/tokenizer/tokenizer-ext4000-65k.yaml "
            "or configs/tokenizer-ext4000-65k.yaml"
        ),
    )
    p.add_argument("--racine", default=".", help="Experiment root (default: .)")
    p.add_argument("--data-root", default=None,
                   help="Optional local parquet root (SmolLM3 names).")
    p.add_argument("--output-root", default=None,
                   help="Where to write tokenizers/ and models/ (default: racine).")
    p.add_argument("--base-id", default=None)
    p.add_argument("--n-tokens", type=int, default=None)
    p.add_argument("--n-echantillon", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--family",
        choices=("65k", "128k"),
        default=None,
        help="Tokenizer family; 128k requires the validated 2.6B/8B special layout.",
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        help="Hub CLM dataset (default: galsenai/cleaned_data_for_clm).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Dry-run: limit Hub/local rows for structure tests.",
    )
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument(
        "--expected-eos",
        type=int,
        default=None,
        help="Assert eos_token_id after build (default: 7 for 65k from config/CLI).",
    )
    return p


def sous_output_root(chemin, output_root: Path) -> Path:
    p = Path(chemin)
    return p if p.is_absolute() else output_root / p


def _resolve_config_path(config: str | None, racine: Path) -> Path | None:
    if not config:
        return None
    p = Path(config)
    if p.is_absolute():
        return p
    cand = racine / p
    if cand.exists():
        return cand
    # Also try relative to this experiment ROOT
    cand2 = ROOT / p
    if cand2.exists():
        return cand2
    return cand  # let caller fail on open


# ---------------------------------------------------------------------------
# Tokenizer helpers (called after lazy imports)
# ---------------------------------------------------------------------------

def _inspect_backend(base_tok) -> dict:
    """Print and return model type + ignore_merges from tokenizers backend."""
    j = json.loads(base_tok.backend_tokenizer.to_str())
    model = j.get("model") or {}
    model_type = model.get("type", "unknown")
    ignore_merges = model.get("ignore_merges")
    print(f"  tokenizer backend model.type = {model_type!r}")
    print(f"  tokenizer backend ignore_merges = {ignore_merges!r}")
    if ignore_merges is False:
        print(
            "  WARNING: ignore_merges is false — whole-word candidates are still "
            "selected, but monotonicity is NOT guaranteed by construction. "
            "Rely on the validation suite.",
            file=sys.stderr,
        )
    elif ignore_merges is None:
        print(
            "  NOTE: ignore_merges missing from backend JSON; "
            "treat monotonicity as empirical only.",
            file=sys.stderr,
        )
    return {
        "model_type": model_type,
        "ignore_merges": ignore_merges,
    }


def selectionner_candidats(base_tok, textes, n_echantillon, seed):
    """Select whole-word forms ranked by expected token savings.

    Same ranking as SmolLM3: gain = freq * (n_tok - 1). On LFM, if
    ignore_merges is false, this is still the selection heuristic but
    monotonicity must be validated empirically.
    """
    random.seed(seed)
    echantillon = random.sample(textes, min(n_echantillon, len(textes)))

    pre = base_tok.backend_tokenizer.pre_tokenizer
    cnt = Counter()
    for t in echantillon:
        for chunk, _ in pre.pre_tokenize_str(t):
            cnt[chunk] += 1
    print(f"  segments distincts : {len(cnt):,}")

    base_vocab = set(base_tok.get_vocab())
    dec = base_tok.backend_tokenizer.decoder
    candidats = []
    for chunk, freq in cnt.most_common():
        if chunk in base_vocab:
            continue
        texte = dec.decode([chunk])
        n_tok = len(base_tok.encode(texte, add_special_tokens=False))
        if n_tok < 2:
            continue
        brut = texte.strip()
        if len(brut) < 2 or not any(c.isalpha() for c in brut):
            continue
        candidats.append((chunk, freq, n_tok, freq * (n_tok - 1)))
    candidats.sort(key=lambda x: -x[3])
    print(f"  candidats : {len(candidats):,}")
    return candidats


def construire_tokenizer_append(base_id, candidats, n_add, family: str = "65k"):
    """Append new tokens via vocab mutation (Ġ chunks for ByteLevel BPE).

    SmolLM3 reindexed specials after BPE (id_map). LFM 65k leaves specials at
    their original low ids and appends after max(existing id)+1.

    The 128k family keeps pad/bos/eos as high-id added_tokens just after the
    BPE table. ``tokenizers`` remaps if new model.vocab rows are placed after
    those specials, so we insert the N new Ġ-chunks *before* specials (at
    first_special_id) and shift every added_token id by +N. Model init must
    permute embedding rows by the same offset.
    """
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required to build a tokenizer")
    tok = AutoTokenizer.from_pretrained(base_id)

    j = json.loads(tok.backend_tokenizer.to_str())
    v = dict(j["model"]["vocab"])
    specials = list(j.get("added_tokens") or [])

    if family == "128k":
        if not specials:
            raise RuntimeError("128k tokenizer has no added_tokens specials to shift")
        first_special = min(int(a["id"]) for a in specials if isinstance(a, dict))
        if first_special != EXPECTED_SPECIALS_128K["pad"]:
            raise RuntimeError(
                f"128k first special id {first_special} != "
                f"{EXPECTED_SPECIALS_128K['pad']}; refusing insert-before-specials"
            )
        nouveaux = []
        for chunk, _, _, _ in candidats:
            if len(nouveaux) >= n_add:
                break
            if chunk in v:
                continue
            nouveaux.append(chunk)
        N = len(nouveaux)
        if N <= 0:
            raise RuntimeError("128k extension selected zero vocab chunks")
        for a in specials:
            a["id"] = int(a["id"]) + N
        for i, chunk in enumerate(nouveaux):
            v[chunk] = first_special + i
        j["model"]["vocab"] = v
        j["added_tokens"] = specials

        d = tempfile.mkdtemp()
        tok.save_pretrained(d)
        with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False)
        for fn in ("vocab.json", "merges.txt"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                os.remove(p)

        ext_tok = AutoTokenizer.from_pretrained(d)
        if ext_tok.eos_token_id != EXPECTED_SPECIALS_128K["eos"] + N:
            raise RuntimeError(
                "128k insert-before-specials eos mismatch: "
                f"got {ext_tok.eos_token_id}, expected "
                f"{EXPECTED_SPECIALS_128K['eos'] + N}"
            )
        if ext_tok.pad_token_id != first_special + N:
            raise RuntimeError(
                "128k insert-before-specials pad mismatch: "
                f"got {ext_tok.pad_token_id}, expected {first_special + N}"
            )
        id_map = {old: old + N for old in range(first_special, len(tok))}
        return ext_tok, {
            "V": first_special,
            "N": N,
            "id_map": id_map,
            "next_id": first_special,
            "special_shift": N,
            "family": "128k",
        }

    max_id = max(v.values()) if v else -1
    for a in specials:
        if isinstance(a, dict) and "id" in a:
            max_id = max(max_id, int(a["id"]))
    max_id = max(max_id, len(tok) - 1)
    next_id = max_id + 1
    V_old = next_id

    nouveaux = [c for c, _, _, _ in candidats[:n_add] if c not in v]
    for i, s in enumerate(nouveaux):
        v[s] = next_id + i
    N = len(nouveaux)

    j["model"]["vocab"] = v
    j["added_tokens"] = specials

    d = tempfile.mkdtemp()
    tok.save_pretrained(d)
    with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)
    for fn in ("vocab.json", "merges.txt"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)

    ext_tok = AutoTokenizer.from_pretrained(d)
    return ext_tok, {
        "V": V_old,
        "N": N,
        "id_map": {},
        "next_id": next_id,
        "family": "65k",
    }


def validate_family_layout(tokenizer, family: str) -> dict:
    """Validate known special IDs and contiguous append boundary."""
    specials = {
        "pad": tokenizer.pad_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
    }
    expected = (
        {"pad": 0, "bos": 1, "eos": 7}
        if family == "65k"
        else EXPECTED_SPECIALS_128K
    )
    if specials != expected:
        raise ValueError(
            f"{family} tokenizer special layout mismatch: got {specials}, "
            f"expected {expected}; refusing to append or reindex"
        )
    vocab_ids = set(tokenizer.get_vocab().values())
    if not vocab_ids or min(vocab_ids) != 0 or max(vocab_ids) != len(tokenizer) - 1:
        raise ValueError(
            "tokenizer IDs are not contiguous through len(tokenizer)-1; "
            "append boundary is unsafe"
        )
    return {
        "family": family,
        "special_ids": specials,
        "vocab_size": len(tokenizer),
        "append_start": len(tokenizer),
        "layout_validated": True,
    }


def valider(base_tok, ext_tok, textes_eval) -> bool:
    """Blocking validations: monotonicity, round-trip, FR/EN/code."""
    ok = True

    mots = []
    for t in textes_eval:
        for w in str(t).split():
            c = w.strip(".,;:!?\"'()[]{}<>-0123456789")
            if c and c.isalpha():
                mots.append(c)
    pire = mieux = 0
    for c in mots[:20000]:
        d = len(base_tok.encode(" " + c, add_special_tokens=False))
        e = len(ext_tok.encode(" " + c, add_special_tokens=False))
        if e > d:
            pire += 1
        elif e < d:
            mieux += 1
    print(f"  monotonicite : pire={pire} mieux={mieux}", "OK" if pire == 0 else "ECHEC")
    ok &= pire == 0

    def rt(tk):
        return sum(
            tk.decode(
                tk.encode(t, add_special_tokens=False),
                clean_up_tokenization_spaces=False,
            )
            == t
            for t in textes_eval[:500]
        )

    rb, re_ = rt(base_tok), rt(ext_tok)
    print(
        f"  round-trip   : base {rb}/500 | etendu {re_}/500",
        "OK" if re_ >= rb else "ECHEC",
    )
    ok &= re_ >= rb

    for phrase in [
        "Le chat dort sur le canape.",
        "The quick brown fox jumps over the lazy dog.",
        "def add(a, b): return a + b",
    ]:
        a = base_tok.encode(phrase, add_special_tokens=False)
        b = ext_tok.encode(phrase, add_special_tokens=False)
        if a != b:
            print(f"  REGRESSION sur : {phrase}")
            ok = False
    print("  non-regression FR/EN/code : OK" if ok else "  non-regression : ECHEC")

    for w in MOTS_TEST:
        nb = len(base_tok.encode(" " + w, add_special_tokens=False))
        ne = len(ext_tok.encode(" " + w, add_special_tokens=False))
        print(f"  probe {w!r}: {nb} -> {ne}")
    return ok


def _compute_fertility(tokenizer, textes):
    """Local fertility (avoid importing common.metrics which pulls torch)."""
    nm = nt = nc = 0
    for t in textes:
        t = "" if t is None else str(t)
        m = len(t.split())
        if m == 0:
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)
        nm += m
        nt += len(ids)
        nc += len(t)
    return {
        "tokens_par_mot": nt / max(nm, 1),
        "tokens_par_caractere": nt / max(nc, 1),
        "n_mots": nm,
        "n_tokens": nt,
    }


def embedding_rows_after_extension(old_rows: int, tokenizer_rows: int) -> int:
    """Keep a model's padded vocabulary capacity when it covers new token IDs."""
    if old_rows <= 0 or tokenizer_rows <= 0:
        raise ValueError("embedding and tokenizer row counts must be positive")
    return max(old_rows, tokenizer_rows)


def initialiser_modele(base_id, base_tok, ext_tok, meta, sortie, expected_eos):
    """Resize embeddings; mean-pool NEW rows; optional 128k special shift; no recalib.

    ATTENTION — do NOT recalibrate the norm of new embeddings.

    Mean-pooling yields a reduced norm (ratio ~0.5). That is CORRECT for tied
    embeddings (logit = h . e). Re-scaling to unit norm doubles logits of new
    tokens and collapses predictions onto them (SmolLM3 measured BPB 5.07 vs
    2.16). Keep the reduced norm as implicit regularization.

    For 128k insert-before-specials, rows [V, L) move to [V+N, L+N) and config
    bos/eos/pad become base_id + N.
    """
    from common.model import ecrire_tie_flag, lire_tie_flag

    if torch is None or AutoModelForCausalLM is None:
        raise RuntimeError("torch and transformers are required for model initialization")

    V, N = meta["V"], meta["N"]
    special_shift = int(meta.get("special_shift") or 0)
    id_map = meta.get("id_map") or {}
    if special_shift and not id_map:
        raise RuntimeError("128k special_shift requires id_map")
    if id_map and not special_shift:
        raise RuntimeError("LFM 65k append-only must not remap specials")

    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    emb_old = model.get_input_embeddings().weight.data.clone()
    old_rows = emb_old.shape[0]
    tied = lire_tie_flag(model.config)
    print(f"  tie_embedding/tie_word_embeddings = {tied}")
    print(f"  embedding rows before resize: {old_rows}")
    if special_shift:
        print(f"  128k special_shift=+{special_shift} insert_at={V}")

    target_rows = embedding_rows_after_extension(old_rows, len(ext_tok))
    if special_shift:
        # Keep the same padding tail length after shifting specials.
        target_rows = max(target_rows, old_rows + N)
    if target_rows != old_rows:
        model.resize_token_embeddings(target_rows)
    emb = model.get_input_embeddings().weight.data
    assert emb.shape[0] == target_rows, (
        f"embedding rows {emb.shape[0]} != target rows {target_rows}"
    )
    assert max(ext_tok.get_vocab().values()) < target_rows, (
        f"tokenizer max id exceeds embedding rows {target_rows}"
    )

    with torch.no_grad():
        if special_shift:
            L = len(base_tok)
            # [0, V) unchanged (already present after resize copy).
            # Move former specials / tail [V, old_rows) -> [V+N, ...).
            move = min(old_rows, target_rows - N) - V
            if move > 0:
                emb[V + N : V + N + move] = emb_old[V : V + move]
            # Clear the new-token window before mean-pool (avoid stale special rows).
            if N > 0:
                emb[V:V + N] = 0
        sans = 0
        for tok_str, nid in ext_tok.get_vocab().items():
            if not (V <= nid < V + N):
                continue
            texte = ext_tok.convert_tokens_to_string([tok_str])
            sub = [
                s for s in base_tok.encode(texte, add_special_tokens=False)
                if s < min(V, old_rows)
            ]
            if not sub:
                sans += 1
                continue
            emb[nid] = emb_old[sub].mean(dim=0)  # mean-pooling
    print(f"  tokens sans sous-tokens exploitables : {sans}")

    if N > 0:
        ratio = (
            emb[V:V + N].norm(dim=1).mean().item()
            / emb_old[: min(V, old_rows)].norm(dim=1).mean().item()
        )
    else:
        ratio = float("nan")
    print(
        f"  ratio de norme : {ratio:.3f}",
        "OK (~0.5 attendu, NE PAS recalibrer)" if ratio < 0.7 else "ANOMALIE",
    )

    eos = model.config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    bos = model.config.bos_token_id
    pad = model.config.pad_token_id
    if special_shift:
        if eos is not None:
            model.config.eos_token_id = int(eos) + special_shift
            eos = model.config.eos_token_id
        if bos is not None:
            model.config.bos_token_id = int(bos) + special_shift
            bos = model.config.bos_token_id
        if pad is not None:
            model.config.pad_token_id = int(pad) + special_shift
            pad = model.config.pad_token_id
        print(f"  specials shifted: bos={bos} eos={eos} pad={pad}")
    else:
        print(f"  specials preserved: bos={bos} eos={eos} pad={pad}")

    if expected_eos is not None and eos != expected_eos:
        raise SystemExit(
            f"ECHEC: eos_token_id={eos} != expected {expected_eos}"
        )
    if ext_tok.eos_token_id != eos:
        raise SystemExit(
            f"ECHEC: tokenizer.eos_token_id={ext_tok.eos_token_id} "
            f"vs config.eos_token_id={eos}"
        )

    model.config.vocab_size = target_rows
    ecrire_tie_flag(model.config, tied)

    Path(sortie).mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16)
    model.save_pretrained(sortie, safe_serialization=True)
    ext_tok.save_pretrained(sortie)
    del model, emb_old
    return (
        ratio,
        {"bos": bos, "eos": eos, "pad": pad},
        {
            "rows_before": old_rows,
            "rows_after": target_rows,
            "tokenizer_size": len(ext_tok),
            "preserved_padded_capacity": old_rows > len(base_tok),
            "special_shift": special_shift,
        },
    )


def _load_texts(args, cfg, racine: Path):
    """Prefer Hub via charger_splits_hub; optional local parquet fallback."""
    sys.path.insert(0, str(ROOT))
    from common.data import charger_splits_hub, charger_splits_local

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    dataset_id = (
        args.dataset_id
        or cfg.get("dataset_id")
        or "galsenai/cleaned_data_for_clm"
    )
    data_root_cfg = args.data_root or cfg.get("data_root")
    data_root = Path(data_root_cfg) if data_root_cfg else None
    if data_root is not None and not data_root.is_absolute():
        data_root = racine / data_root

    # Prefer local parquet when data_root is provided (filtered corpora, offline).
    if data_root is not None:
        try:
            print(f"Loading local parquet from {data_root} …")
            train, ev = charger_splits_local(racine, data_root=data_root)
            if args.max_rows is not None:
                train = train[: args.max_rows]
                ev = ev[: args.max_rows]
            return train, ev
        except FileNotFoundError as e:
            print(f"  local parquet missing ({e}) — falling back to Hub")

    if token:
        print(f"Loading Hub dataset {dataset_id} (max_rows={args.max_rows}) …")
        return charger_splits_hub(
            dataset_id=dataset_id, max_rows=args.max_rows, token=token
        )

    print(
        "ERROR: HF_TOKEN is not set and local parquet splits were not found.\n"
        "  Set HF_TOKEN to load the Hub CLM dataset, or prepare local parquet "
        "via data_root / scripts/data.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def run(args: argparse.Namespace) -> int:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required; install the train dependency group")

    racine = Path(args.racine).resolve()
    if not (racine / "pyproject.toml").exists() and (ROOT / "pyproject.toml").exists():
        racine = ROOT

    cfg_path = _resolve_config_path(args.config, racine)
    cfg = {}
    if cfg_path is not None:
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            return 1
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    family = args.family or cfg.get("family") or "65k"
    if family not in {"65k", "128k"}:
        raise ValueError(f"unsupported tokenizer family: {family}")

    base_id = args.base_id or cfg.get("base_id") or "LiquidAI/LFM2.5-1.2B-Base"
    output_root_cfg = args.output_root or cfg.get("output_root")
    output_root = Path(output_root_cfg) if output_root_cfg else racine
    if not output_root.is_absolute():
        output_root = racine / output_root

    n_tokens = args.n_tokens if args.n_tokens is not None else cfg.get("n_tokens", 4000)
    n_echantillon = (
        args.n_echantillon
        if args.n_echantillon is not None
        else cfg.get("n_echantillon", 300_000)
    )
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    tracking = TrackingRun(
        size=Path(base_id).name,
        stage="tokenizer-build",
        tokenizer_family=family,
        method=f"append-{n_tokens}",
        seed=int(seed),
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", f"{family}-{n_tokens}"),
        config=cfg,
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output_root / "results" / "full_pipeline" / "tracking"
        / f"tokenizer-{family}-{n_tokens}.json",
    ).start()
    atexit.register(tracking.finish, success=False)
    expected_eos = (
        args.expected_eos
        if args.expected_eos is not None
        else cfg.get(
            "eos_token_id",
            EXPECTED_EOS_65K if family == "65k" else EXPECTED_SPECIALS_128K["eos"],
        )
    )

    textes_train, textes_eval = _load_texts(args, cfg, racine)

    base_tok = AutoTokenizer.from_pretrained(base_id)
    layout_meta = validate_family_layout(base_tok, family)
    print(
        f"tokenizer de base : vocab_size={base_tok.vocab_size} "
        f"len={len(base_tok)} (utiliser len)"
    )
    print(
        f"  specials: bos={base_tok.bos_token_id} "
        f"eos={base_tok.eos_token_id} pad={base_tok.pad_token_id}"
    )
    if expected_eos is not None and base_tok.eos_token_id != expected_eos:
        print(
            f"  WARNING: base eos_token_id={base_tok.eos_token_id} "
            f"!= expected {expected_eos}",
            file=sys.stderr,
        )

    backend_meta = _inspect_backend(base_tok)

    fert_base = _compute_fertility(base_tok, textes_eval)["tokens_par_mot"]
    print(f"fertilite de base : {fert_base:.3f} tokens/mot\n")

    print("selection des candidats")
    candidats = selectionner_candidats(base_tok, textes_train, n_echantillon, seed)

    print(f"\nconstruction APPEND-ONLY du tokenizer etendu ({n_tokens} tokens)")
    ext_tok, meta = construire_tokenizer_append(
        base_id, candidats, n_tokens, family=family
    )
    fert_ext = _compute_fertility(ext_tok, textes_eval)["tokens_par_mot"]
    print(f"  vocabulaire : {len(base_tok):,} -> {len(ext_tok):,}")
    print(
        f"  fertilite   : {fert_base:.3f} -> {fert_ext:.3f} "
        f"({100 * (1 - fert_ext / fert_base):.1f} % de gain)"
    )
    special_shift = int(meta.get("special_shift") or 0)
    if family == "128k":
        if special_shift <= 0:
            raise SystemExit("ECHEC: 128k extension missing special_shift metadata")
        if ext_tok.eos_token_id != base_tok.eos_token_id + special_shift:
            raise SystemExit(
                f"ECHEC: 128k eos shift mismatch "
                f"{base_tok.eos_token_id}+{special_shift} -> {ext_tok.eos_token_id}"
            )
        expected_eos = base_tok.eos_token_id + special_shift
        print(
            f"  128k specials shifted by +{special_shift}: "
            f"eos {base_tok.eos_token_id} -> {ext_tok.eos_token_id}"
        )
    elif ext_tok.eos_token_id != base_tok.eos_token_id:
        raise SystemExit(
            f"ECHEC: eos shifted {base_tok.eos_token_id} -> {ext_tok.eos_token_id}"
        )

    print("\nvalidations")
    if not valider(base_tok, ext_tok, textes_eval):
        raise SystemExit("ECHEC des validations — tokenizer non sauvegarde")

    tok_dir = sous_output_root(
        cfg.get("tokenizer_path", f"tokenizers/lfm25_wolof_ext_{meta['N']}"),
        output_root,
    )
    tok_dir.mkdir(parents=True, exist_ok=True)
    ext_tok.save_pretrained(tok_dir)

    print("\ninitialisation des embeddings (mean-pool, no recalib)")
    if torch is None:
        print(
            f"torch missing — tokenizer saved to {tok_dir}, "
            "but model init skipped. Install train group and re-run.",
            file=sys.stderr,
        )
        with open(tok_dir / "extension_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_id": base_id,
                    "family": family,
                    "methode": "append-only whole-word; model init pending (no torch)",
                    "n_tokens": meta["N"],
                    "n_added": meta["N"],
                    "vocab_old": len(base_tok),
                    "vocab_total_after": len(ext_tok),
                    "eos_token_id": ext_tok.eos_token_id,
                    "fertility_before": fert_base,
                    "fertility_after": fert_ext,
                    "seed": seed,
                    "backend": backend_meta,
                    "layout": layout_meta,
                    "model_init": False,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return 2

    model_dir = sous_output_root(
        cfg.get("model_path", f"models/lfm25-wolof-ext{meta['N']}-init"),
        output_root,
    )
    ratio, specials, embedding_meta = initialiser_modele(
        base_id, base_tok, ext_tok, meta, model_dir, expected_eos
    )

    with open(tok_dir / "extension_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_id": base_id,
                "family": family,
                "methode": (
                    "mots entiers par frequence, APPEND-ONLY "
                    "(no special reindex); mean-pool; no norm recalib"
                ),
                "n_tokens": meta["N"],
                "n_added": meta["N"],
                "vocab_append_start": meta["V"],
                "vocab_total_before": len(base_tok),
                "vocab_total_after": len(ext_tok),
                "eos_token_id": specials["eos"],
                "bos_token_id": specials["bos"],
                "pad_token_id": specials["pad"],
                "fertility_before": fert_base,
                "fertility_after": fert_ext,
                "fertility_default": fert_base,
                "fertility_extended": fert_ext,
                "ratio_norme": ratio,
                "seed": seed,
                "backend": backend_meta,
                "layout": layout_meta,
                "embeddings": embedding_meta,
                "model_init": True,
                "model_path": str(model_dir),
                "tokenizer_path": str(tok_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\ntokenizer : {tok_dir}")
    print(f"modele    : {model_dir}")
    tracking.log(
        {
            "fertility_before": fert_base,
            "fertility_after": fert_ext,
            "fertility_gain_pct": 100 * (1 - fert_ext / fert_base),
            "token_count_added": meta["N"],
            "embedding_norm_ratio": ratio,
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except NotImplementedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
