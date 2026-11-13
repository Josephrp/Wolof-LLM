#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model (bf16), fix tie flags, BPB gate.

Port of SmolLM3 scripts/05_merge_adapter.py for LFM2.5:
  * lire_tie_flag / ecrire_tie_flag (tie_embedding OR tie_word_embeddings)
  * Hub eval texts when no local parquet
  * BPB delta > tolerance -> non-zero exit (default 0.002)

  uv run python scripts/clm/05_merge_adapter.py --help
  uv run python scripts/clm/05_merge_adapter.py --run smoke_350m_ext_lora-smoke
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def merge_within_tolerance(before: float, after: float, tolerance: float) -> bool:
    """Pure helper used by the blocking merge gate and offline tests."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return abs(after - before) <= tolerance


def artifact_reference(value: str | Path, project_root: Path) -> str:
    """Return a portable, non-private reference for persisted metadata."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge LoRA adapter (bf16) + tie-flag fix + BPB gate"
    )
    p.add_argument("--run", required=True, help="Run directory name under runs/ or results/smoke/")
    p.add_argument("--racine", default=".")
    p.add_argument("--output-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset-id", default="galsenai/cleaned_data_for_clm")
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument(
        "--n-textes",
        type=int,
        default=300,
        help="Eval texts for non-regression BPB check",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.002,
        help="Max |BPB_after - BPB_before|; exceed -> exit 1",
    )
    return p


def etat_tying(model):
    """Return (same_data_ptr, values_equal)."""
    import torch

    entree = model.get_input_embeddings().weight
    sortie_mod = model.get_output_embeddings()
    if sortie_mod is None:
        return True, True
    sortie = sortie_mod.weight
    memes_donnees = entree.data_ptr() == sortie.data_ptr()
    if memes_donnees:
        return True, True
    identiques = bool(torch.equal(entree.detach().cpu(), sortie.detach().cpu()))
    return False, identiques


def _find_run_dir(racine: Path, output_root: Path, run: str) -> Path:
    candidates = [
        output_root / "runs" / run,
        output_root / "results" / "smoke" / run,
        racine / "runs" / run,
        racine / "results" / "smoke" / run,
        output_root / run,
    ]
    # Also allow run path relative to output_root when config used nested output_root
    for c in candidates:
        if (c / "adapter").is_dir():
            return c
    # rglob fallback
    for root in (output_root, racine):
        hits = list(root.rglob(run))
        for h in hits:
            if h.is_dir() and (h / "adapter").is_dir():
                return h
    # Config run IDs can include the model size even when the output directory
    # does not (for example, v2_350m_ext4000_lora vs v2_ext4000_lora).
    seen: set[Path] = set()
    for root in (output_root / "runs", racine / "runs"):
        if not root.is_dir() or root in seen:
            continue
        seen.add(root)
        for journal_path in root.rglob("journal.json"):
            candidate = journal_path.parent
            if not (candidate / "adapter").is_dir():
                continue
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if journal.get("run") == run:
                return candidate
    raise SystemExit(
        f"adaptateur introuvable for run={run!r}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _load_eval_texts(racine, data_root, dataset_id, token, n):
    from common.data import charger_splits_hub, charger_splits_local

    if data_root is not None:
        try:
            _, ev = charger_splits_local(racine, data_root=Path(data_root))
            return ev[:n]
        except FileNotFoundError:
            pass
    try:
        _, ev = charger_splits_local(racine, data_root=racine / "data")
        return ev[:n]
    except FileNotFoundError:
        pass
    if not token:
        print(
            "ERROR: no local parquet and HF_TOKEN unset for merge BPB check.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    _, ev = charger_splits_hub(dataset_id=dataset_id, token=token)
    return ev[:n]


def run(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from common.metrics import compute_bpb
    from common.model import ecrire_tie_flag, lire_tie_flag
    from common.tracking import TrackingRun

    racine = Path(args.racine).resolve()
    if not (racine / "pyproject.toml").exists() and (ROOT / "pyproject.toml").exists():
        racine = ROOT
    output_root = Path(args.output_root) if args.output_root else racine
    if not output_root.is_absolute():
        output_root = racine / output_root

    dossier_run = _find_run_dir(racine, output_root, args.run)
    chemin_adapter = dossier_run / "adapter"
    tracking = TrackingRun(
        size=args.run,
        stage="cpt-merge",
        tokenizer_family="run-artifact",
        method="merge",
        seed=42,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", args.run),
        config=vars(args),
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=dossier_run / "trackio_merge.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    with open(dossier_run / "journal.json", encoding="utf-8") as journal_file:
        journal = json.load(journal_file)
    if journal.get("methode") != "lora":
        raise SystemExit(f"{args.run} n'est pas un run LoRA — rien a fusionner")
    chemin_base = journal["modele"]

    print(f"{'=' * 70}\n  fusion : {args.run}\n{'=' * 70}")
    print(f"  modele de base : {chemin_base}")
    print(f"  adaptateur     : {chemin_adapter}")

    tok = AutoTokenizer.from_pretrained(str(chemin_adapter))
    dm = {"": 0} if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        chemin_base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=dm,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    tied_config = lire_tie_flag(base.config)
    print(f"  tie flag (config base) : {tied_config}")

    model = PeftModel.from_pretrained(base, str(chemin_adapter))
    model.eval()

    tied_avant, identiques_avant = etat_tying(model)
    print(
        f"  avant fusion — memes donnees : {tied_avant} | "
        f"valeurs identiques : {identiques_avant}"
    )

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    textes = _load_eval_texts(
        racine, args.data_root, args.dataset_id, token, args.n_textes
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bpb_avant = compute_bpb(model, tok, textes, device=device)
    print(f"  BPB avant fusion : {bpb_avant['bpb']:.4f}")

    fusionne = model.merge_and_unload()
    tied_apres, identiques_apres = etat_tying(fusionne)
    print(
        f"  apres fusion  — memes donnees : {tied_apres} | "
        f"valeurs identiques : {identiques_apres}"
    )

    doit_etre_tied = tied_apres or identiques_apres
    actuel = lire_tie_flag(fusionne.config)
    if actuel != doit_etre_tied:
        print(f"  correction tie flag : {actuel} -> {doit_etre_tied}")
        ecrire_tie_flag(fusionne.config, doit_etre_tied)
    else:
        print(f"  tie flag deja correct ({doit_etre_tied})")

    bpb_apres = compute_bpb(fusionne, tok, textes, device=device)
    ecart = abs(bpb_apres["bpb"] - bpb_avant["bpb"])
    print(f"  BPB apres fusion : {bpb_apres['bpb']:.4f}  (ecart {ecart:+.5f})")

    if not merge_within_tolerance(bpb_avant["bpb"], bpb_apres["bpb"], args.tolerance):
        print(
            f"\n  ECHEC : fusion a change le BPB de {ecart:.5f} "
            f"(tolerance {args.tolerance}).\n"
            f"  Ne pas utiliser ce merge pour le SFT.",
            file=sys.stderr,
        )
        return 1

    sortie = output_root / "models" / f"{args.run}-merged"
    sortie.mkdir(parents=True, exist_ok=True)
    fusionne.save_pretrained(str(sortie), safe_serialization=True)
    tok.save_pretrained(str(sortie))

    with open(sortie / "fusion_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_source": args.run,
                "modele_base": artifact_reference(chemin_base, racine),
                "adaptateur": artifact_reference(chemin_adapter, racine),
                "tie_flag_config_base": tied_config,
                "tied_avant_fusion": tied_avant,
                "tied_apres_fusion": doit_etre_tied,
                "bpb_avant_fusion": bpb_avant["bpb"],
                "bpb_apres_fusion": bpb_apres["bpb"],
                "ecart_bpb": ecart,
                "tolerance": args.tolerance,
                "n_textes_verification": args.n_textes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n  fusion validee — modele complet : {sortie}")
    tracking.log(
        {
            "bpb_before_merge": bpb_avant["bpb"],
            "bpb_after_merge": bpb_apres["bpb"],
            "merge_delta": ecart,
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
