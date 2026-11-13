#!/usr/bin/env python3
"""Recalcule le BPB de tous les runs sur le MEME split, et produit la comparaison.

Recalculer ici plutot que de relire les journaux garantit que tous les chiffres
proviennent de la meme fonction, du meme split et du meme nombre de textes.
Deux runs ayant mesure leur BPB differemment ne seraient pas comparables sans
que rien ne le signale.

    python scripts/04_evaluate.py --output-root /mnt/volume_w2l8nkp/galsenllm/OutputOfSmolLM_Model
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import charger_splits
from common.metrics import compute_bpb, compute_fertility


def charger_run(dossier, base_id):
    """Charge un run : adaptateur LoRA sur son socle, ou modele complet."""
    j = json.load(open(dossier / "journal.json", encoding="utf-8"))
    if j["methode"] == "lora":
        from peft import PeftModel
        socle = AutoModelForCausalLM.from_pretrained(
            j["modele"], dtype=torch.bfloat16, device_map={"": 0})
        model = PeftModel.from_pretrained(socle, str(dossier / "adapter")).eval()
        tok = AutoTokenizer.from_pretrained(str(dossier / "adapter"))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(dossier / "model"), dtype=torch.bfloat16, device_map={"": 0}).eval()
        tok = AutoTokenizer.from_pretrained(str(dossier / "model"))
    return j, tok, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    p.add_argument("--data-root", default=None,
                   help="Dossier contenant les splits parquet (défaut: /mnt/volume_w2l8nkp/galsenllm/data).")
    p.add_argument("--output-root", default=None,
                   help="Dossier contenant runs/ et où écrire results/ (défaut: <racine>).")
    p.add_argument("--base-id", default="HuggingFaceTB/SmolLM3-3B-Base")
    p.add_argument("--n-textes", type=int, default=500)
    a = p.parse_args()

    racine = Path(a.racine)
    output_root = Path(a.output_root) if a.output_root else racine
    if a.data_root:
        data_root = Path(a.data_root)
    else:
        data_root = Path("/mnt/volume_w2l8nkp/galsenllm/data")
        if not data_root.exists():
            data_root = output_root / "data"
            if not data_root.exists():
                data_root = racine / "data"
    print(f"data_root = {data_root}")
    _, textes_eval = charger_splits(racine, data_root=data_root)
    textes = textes_eval[:a.n_textes]

    lignes = []

    # reference : modele de base, non entraine
    base_tok = AutoTokenizer.from_pretrained(a.base_id)
    base = AutoModelForCausalLM.from_pretrained(a.base_id, dtype=torch.bfloat16,
                                                device_map={"": 0})
    r = compute_bpb(base, base_tok, textes)
    f = compute_fertility(base_tok, textes_eval)
    lignes.append({"run": "base (non entraine)", "methode": "-", "tokenizer": "defaut",
                   "vocab": len(base_tok), "lr": None,
                   "fertilite": round(f["tokens_par_mot"], 3),
                   "BPB": round(r["bpb"], 4), "PPL": round(r["ppl"], 2),
                   "duree_h": None})
    del base
    torch.cuda.empty_cache()

    runs_dir = output_root / "runs"
    for d in sorted(runs_dir.iterdir()):
        if not (d / "journal.json").exists() or "smoke" in d.name:
            continue
        j, tok, model = charger_run(d, a.base_id)
        r = compute_bpb(model, tok, textes)
        f = compute_fertility(tok, textes_eval)
        lignes.append({
            "run": j["run"], "methode": j["methode"],
            "tokenizer": "etendu" if j["vocab"] > len(base_tok) else "defaut",
            "vocab": j["vocab"], "lr": j["hyperparametres"]["lr"],
            "fertilite": round(f["tokens_par_mot"], 3),
            "BPB": round(r["bpb"], 4), "PPL": round(r["ppl"], 2),
            "duree_h": j["budget"]["duree_heures"]})
        print(f"  {j['run']:32} BPB={r['bpb']:.4f}")
        del model
        torch.cuda.empty_cache()

    tableau = pd.DataFrame(lignes).sort_values("BPB")
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(results_dir / "comparaison.csv", index=False)

    print("\n" + tableau.to_string(index=False))
    print("\nRappels de lecture :")
    print("  - Le BPB est la SEULE metrique comparable entre tokenizers differents.")
    print("    La PPL et la loss se calculent par token : ne les comparer qu'a")
    print("    tokenizer identique.")
    print("  - Verifier que le learning rate est constant sur l'axe compare :")
    print("    deux runs LoRA a des LR differents ne sont pas comparables.")

    # graphique
    t = tableau[tableau["run"] != "base (non entraine)"]
    if len(t):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        couleurs = ["#2ca02c" if x == "etendu" else "#7f7f7f" for x in t["tokenizer"]]
        ax.barh(t["run"], t["BPB"], color=couleurs)
        base_bpb = tableau[tableau["run"] == "base (non entraine)"]["BPB"].iloc[0]
        ax.axvline(base_bpb, ls="--", color="red", label=f"base ({base_bpb:.3f})")
        ax.set_xlabel("BPB (plus bas = meilleur)")
        ax.legend()
        ax.grid(alpha=.3, axis="x")
        plt.tight_layout()
        plt.savefig(results_dir / "comparaison.png", dpi=200, bbox_inches="tight")
        print(f"\nresultats : {results_dir}")


if __name__ == "__main__":
    main()
