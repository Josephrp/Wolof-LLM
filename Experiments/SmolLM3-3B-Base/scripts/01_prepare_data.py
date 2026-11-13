#!/usr/bin/env python3
"""Prepare les splits train/eval a partir du corpus wolof nettoye.

Produit trois fichiers, figes une fois pour toutes :
  data/wolof_bpe_train_v1.parquet   entrainement
  data/wolof_eval_v1.parquet        evaluation brute
  data/wolof_eval_clean_v1.parquet  evaluation sans quasi-doublons  <- a utiliser

Le split d'evaluation doit etre IDENTIQUE pour tous les runs de la grille,
sinon les comparaisons ne valent rien. La graine le garantit.

    python scripts/01_prepare_data.py --corpus data/wolof_corpus_clean.parquet
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def normaliser(t):
    """Normalisation agressive pour detecter les quasi-doublons."""
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--racine", default=".")
    p.add_argument("--data-root", default=None,
                   help="Dossier où écrire les splits parquet (défaut: <racine>/data).")
    p.add_argument("--corpus", required=True, help="parquet du corpus nettoye")
    p.add_argument("--taille-eval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    sortie = Path(a.data_root) if a.data_root else Path(a.racine) / "data"
    sortie.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.corpus)
    assert "text" in df.columns, f"colonne 'text' absente : {df.columns.tolist()}"
    print(f"corpus : {len(df):,} lignes")

    melange = df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    eval_df = melange.iloc[:a.taille_eval].reset_index(drop=True)
    train_df = melange.iloc[a.taille_eval:].reset_index(drop=True)
    print(f"train {len(train_df):,} | eval {len(eval_df):,}")

    # --- controle de fuite -------------------------------------------------
    # Le corpus source n'a subi qu'une deduplication EXACTE : des variantes de
    # casse ou de ponctuation peuvent subsister. Comme le split vient du meme
    # corpus, une fuite gonflerait tous les resultats par memorisation.
    train_norm = set(train_df["text"].map(normaliser))
    fuite = eval_df["text"].map(normaliser).isin(train_norm)

    octets = eval_df["text"].astype(str).map(lambda t: len(t.encode("utf-8")))
    part_textes = 100 * fuite.mean()
    part_octets = 100 * octets[fuite].sum() / octets.sum()

    print(f"\nfuite en textes : {part_textes:.2f} %")
    print(f"fuite en octets : {part_octets:.2f} %   <- ce qui compte pour le BPB")
    print("  (le BPB pondere par les octets : compter les textes surestime "
          "l'impact, la fuite se concentrant sur les textes courts)")

    eval_clean = eval_df[~fuite].reset_index(drop=True)
    print(f"eval propre : {len(eval_clean):,} textes")

    train_df.to_parquet(sortie / "wolof_bpe_train_v1.parquet", index=False)
    eval_df.to_parquet(sortie / "wolof_eval_v1.parquet", index=False)
    eval_clean.to_parquet(sortie / "wolof_eval_clean_v1.parquet", index=False)

    import json
    with open(sortie / "splits_meta.json", "w", encoding="utf-8") as f:
        json.dump({"corpus": str(a.corpus), "seed": a.seed,
                   "n_train": len(train_df), "n_eval": len(eval_df),
                   "n_eval_clean": len(eval_clean),
                   "fuite_pct_textes": round(part_textes, 3),
                   "fuite_pct_octets": round(part_octets, 3)},
                  f, indent=2, ensure_ascii=False)
    print(f"\nsplits figes dans {sortie}")


if __name__ == "__main__":
    main()
