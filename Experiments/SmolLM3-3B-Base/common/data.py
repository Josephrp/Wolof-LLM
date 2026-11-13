"""Chargement des splits et mise en blocs (packing) des textes."""

from pathlib import Path

import pandas as pd
from datasets import Dataset


def charger_splits(racine, split_eval="wolof_eval_clean_v1.parquet", data_root=None):
    """Retourne (textes_train, textes_eval) depuis les splits figés.

    Le split d'évaluation par défaut est la version *propre*, débarrassée des
    quasi-doublons présents dans le train. Utiliser le split brut fausserait
    les mesures par mémorisation.
    """
    d = Path(data_root) if data_root else Path(racine) / "data"
    train_path = d / "wolof_bpe_train_v1.parquet"
    eval_path = d / split_eval
    if not train_path.exists() or not eval_path.exists():
        manquants = [str(p) for p in (train_path, eval_path) if not p.exists()]
        raise FileNotFoundError(
            "Splits parquet introuvables : "
            + ", ".join(manquants)
            + "\nLancer d'abord scripts/01_prepare_data.py avec le même data_root."
        )
    train = pd.read_parquet(train_path)
    ev = pd.read_parquet(eval_path)
    return (train["text"].astype(str).tolist(),
            ev["text"].astype(str).tolist())


def paqueter(textes, tokenizer, longueur, limite=None):
    """Concatène les textes puis découpe en blocs de `longueur` tokens.

    Sans packing, la majorité du calcul partirait en remplissage : les textes
    du corpus ont une médiane d'environ 97 caracteres pour une fenetre de
    1 024 tokens.

    Le packing depend du tokenizer : il doit etre refait pour chaque
    configuration de vocabulaire.
    """
    if limite:
        textes = textes[:limite]
    ids = []
    eos = tokenizer.eos_token_id
    for t in textes:
        ids.extend(tokenizer.encode(t, add_special_tokens=False) + [eos])
    n = (len(ids) // longueur) * longueur
    blocs = [ids[i:i + longueur] for i in range(0, n, longueur)]
    return Dataset.from_dict({"input_ids": blocs,
                              "labels": [b[:] for b in blocs]})
