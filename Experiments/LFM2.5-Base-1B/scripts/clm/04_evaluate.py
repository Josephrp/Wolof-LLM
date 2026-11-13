#!/usr/bin/env python3
"""Recalculate BPB for all CLM runs on the SAME split; write CSV + plot.

Port of SmolLM3 scripts/04_evaluate.py — no hardcoded machine paths.
Prefer Hub cleaned_data_for_clm when local parquet is absent.

  uv run python scripts/clm/04_evaluate.py --help
  uv run python scripts/clm/04_evaluate.py --output-root .
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recalc BPB across LFM2.5 Wolof CLM runs -> results/"
    )
    p.add_argument("--racine", default=".")
    p.add_argument(
        "--data-root",
        default=None,
        help="Local parquet dir (SmolLM3 names). Default: try <racine>/data then Hub.",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Root containing runs/ (and where results/ is written). Default: <racine>.",
    )
    p.add_argument(
        "--runs-glob",
        default=None,
        help="Optional subdirectory under output-root to scan (default: runs/).",
    )
    p.add_argument("--base-id", default="LiquidAI/LFM2.5-1.2B-Base")
    p.add_argument("--n-textes", type=int, default=500)
    p.add_argument("--dataset-id", default="galsenai/cleaned_data_for_clm")
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include runs with 'smoke' in the directory name",
    )
    return p


def _load_eval_texts(racine, data_root, dataset_id, token, n_textes):
    from common.data import charger_splits_hub, charger_splits_local

    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root))
    candidates.extend([racine / "data", ROOT / "data"])

    for d in candidates:
        try:
            _, textes_eval = charger_splits_local(racine, data_root=d)
            print(f"data_root (local) = {d}")
            return textes_eval[:n_textes], textes_eval
        except FileNotFoundError:
            continue

    if not token:
        print(
            "ERROR: no local parquet and HF_TOKEN unset.\n"
            "  Set HF_TOKEN or pass --data-root.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"data_root (Hub) = {dataset_id}")
    _, textes_eval = charger_splits_hub(dataset_id=dataset_id, token=token)
    return textes_eval[:n_textes], textes_eval


def _iter_run_dirs(output_root: Path, runs_glob: str | None, include_smoke: bool):
    roots = []
    if runs_glob:
        roots.append(output_root / runs_glob)
    else:
        roots.append(output_root / "runs")
        # Also scan results/smoke when present
        smoke = output_root / "results" / "smoke"
        if smoke.is_dir():
            roots.append(smoke)

    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.rglob("journal.json")):
            dossier = d.parent
            if dossier in seen:
                continue
            if (not include_smoke) and "smoke" in dossier.name.lower():
                continue
            seen.add(dossier)
            yield dossier


def run(args: argparse.Namespace) -> int:
    import json

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from common.metrics import compute_bpb, compute_fertility
    from common.tracking import TrackingRun

    racine = Path(args.racine).resolve()
    if not (racine / "pyproject.toml").exists() and (ROOT / "pyproject.toml").exists():
        racine = ROOT
    output_root = Path(args.output_root) if args.output_root else racine
    if not output_root.is_absolute():
        output_root = racine / output_root
    tracking = TrackingRun(
        size=Path(args.base_id).name,
        stage="clm-eval",
        tokenizer_family="mixed",
        method="evaluation",
        seed=42,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", Path(args.base_id).name),
        config=vars(args),
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output_root / "results" / "trackio_eval.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    textes, textes_eval_full = _load_eval_texts(
        racine,
        Path(args.data_root) if args.data_root else None,
        args.dataset_id,
        token,
        args.n_textes,
    )

    lignes = []

    print(f"Reference base: {args.base_id}")
    base_tok = AutoTokenizer.from_pretrained(args.base_id)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_id,
        dtype=model_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    r = compute_bpb(base, base_tok, textes, device=device)
    f = compute_fertility(base_tok, textes_eval_full)
    lignes.append(
        {
            "run": "base (non entraine)",
            "methode": "-",
            "tokenizer": "defaut",
            "vocab": len(base_tok),
            "lr": None,
            "fertilite": round(f["tokens_par_mot"], 3),
            "BPB": round(r["bpb"], 4),
            "PPL": round(r["ppl"], 2),
            "duree_h": None,
        }
    )
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def charger_run(dossier):
        with open(dossier / "journal.json", encoding="utf-8") as journal_file:
            j = json.load(journal_file)
        dm = {"": 0} if torch.cuda.is_available() else "cpu"
        if j["methode"] == "lora":
            from peft import PeftModel

            socle = AutoModelForCausalLM.from_pretrained(
                j["modele"],
                dtype=model_dtype,
                device_map=dm,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(socle, str(dossier / "adapter")).eval()
            tok = AutoTokenizer.from_pretrained(str(dossier / "adapter"))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(dossier / "model"),
                dtype=model_dtype,
                device_map=dm,
                trust_remote_code=True,
            ).eval()
            tok = AutoTokenizer.from_pretrained(str(dossier / "model"))
        return j, tok, model

    for d in _iter_run_dirs(output_root, args.runs_glob, args.include_smoke):
        try:
            j, tok, model = charger_run(d)
        except Exception as ex:  # noqa: BLE001 - skip corrupt/incomplete runs
            print(f"  skip {d}: {ex}")
            continue
        r = compute_bpb(model, tok, textes, device=device)
        f = compute_fertility(tok, textes_eval_full)
        lignes.append(
            {
                "run": j["run"],
                "methode": j["methode"],
                "tokenizer": "etendu" if j["vocab"] > len(base_tok) else "defaut",
                "vocab": j["vocab"],
                "lr": j.get("hyperparametres", {}).get("lr"),
                "fertilite": round(f["tokens_par_mot"], 3),
                "BPB": round(r["bpb"], 4),
                "PPL": round(r["ppl"], 2),
                "duree_h": j.get("budget", {}).get("duree_heures"),
            }
        )
        print(f"  {j['run']:40} BPB={r['bpb']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tableau = pd.DataFrame(lignes).sort_values("BPB")
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "comparaison.csv"
    tableau.to_csv(csv_path, index=False)

    print("\n" + tableau.to_string(index=False))
    print("\nRappels : BPB seul est comparable entre tokenizers differents.")

    t = tableau[tableau["run"] != "base (non entraine)"]
    if len(t):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(9, 4.5))
        couleurs = [
            "#2ca02c" if x == "etendu" else "#7f7f7f" for x in t["tokenizer"]
        ]
        ax.barh(t["run"], t["BPB"], color=couleurs)
        base_bpb = tableau[tableau["run"] == "base (non entraine)"]["BPB"].iloc[0]
        ax.axvline(base_bpb, ls="--", color="red", label=f"base ({base_bpb:.3f})")
        ax.set_xlabel("BPB (plus bas = meilleur)")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(results_dir / "comparaison.png", dpi=200, bbox_inches="tight")
        print(f"\nresultats : {results_dir}")
    else:
        print(f"\nCSV only (no non-base runs): {csv_path}")
    tracking.log(
        {
            "evaluated_runs": len(tableau),
            "best_bpb": float(tableau["BPB"].min()),
            "base_bpb": float(
                tableau[tableau["run"] == "base (non entraine)"]["BPB"].iloc[0]
            ),
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
