#!/usr/bin/env python3
"""Nettoyage reproductible du corpus brut avant génération des splits.

Le corpus source fourni au projet contient des textes wolof dans une colonne
`text`, avec parfois des blocs de code délimités par des triples accents graves
ou du code inline. Cette étape retire ces séquences de code, normalise les
retours à la ligne et élimine les textes devenus trop courts après nettoyage.

Usage typique :
    python scripts/00_clean_corpus.py --input data/data_v1.parquet --output data/wolof_corpus_clean.parquet
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def strip_code(t):
    """Retire les blocs de code et les fragments inline de texte wolof."""
    t = str(t).replace("\\n", "\n")
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True,
                   help="Parquet source brut contenant une colonne 'text'.")
    p.add_argument("--output", "-o", default="data/wolof_corpus_clean.parquet",
                   help="Parquet nettoyé à écrire.")
    p.add_argument("--min-len", type=int, default=5,
                   help="Longueur minimale du texte nettoyé à conserver.")
    a = p.parse_args()

    input_path = Path(a.input)
    output_path = Path(a.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    assert "text" in df.columns, f"colonne 'text' absente : {df.columns.tolist()}"

    avant = len(df)
    df["text"] = df["text"].astype(str).map(strip_code)
    df = df[df["text"].str.strip().str.len() >= a.min_len].reset_index(drop=True)

    df.to_parquet(output_path, index=False)

    print(f"corpus source : {avant:,} lignes")
    print(f"corpus nettoyé : {len(df):,} lignes")
    print(f"lignes perdues : {avant - len(df):,}")
    print(f"écrit dans {output_path}")


if __name__ == "__main__":
    main()
