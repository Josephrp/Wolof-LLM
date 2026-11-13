"""
Script unique de nettoyage du corpus Wolof.
Télécharge depuis HuggingFace Hub, nettoie, sauvegarde les statistiques et les données localement.
Toutes les valeurs viennent de config.json — ne rien modifier ici pour changer une expérimentation.

Usage :
    python3 preprocessing.py --config config.json
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_token(env_var: str) -> str:
    token = os.environ.get(env_var)
    if not token:
        raise ValueError(
            f"Token HuggingFace introuvable dans la variable d'environnement '{env_var}'. "
            f"Sur Colab : os.environ['{env_var}'] = 'hf_xxx'"
        )
    return token


def download_raw_dataset(cfg: dict) -> pd.DataFrame:
    """Télécharge le dataset brut depuis HF Hub et retourne un DataFrame."""
    raw_cfg = cfg["raw_dataset"]
    token = get_token(raw_cfg["hf_token_env_var"])

    print(f"Téléchargement de {raw_cfg['hub_repo_id']} depuis HuggingFace Hub...")
    dataset = load_dataset(raw_cfg["hub_repo_id"], split=raw_cfg["split"], token=token)
    df = dataset.to_pandas()
    print(f"{len(df)} lignes téléchargées.")
    return df


def compute_stats(df: pd.DataFrame, text_col: str) -> dict:
    """Statistiques équivalentes à la section 'Visualisation des données' du notebook."""
    # Calcul de la longueur des textes (en caractères) et statistiques de distribution
    # (moyenne, écart-type, quartiles...)
    lengths = df[text_col].str.len()
    return {
        # Dimensions du dataset
        "num_rows": int(len(df)),
        # Vérification des valeurs manquantes (NaN)
        "num_nan": int(df[text_col].isna().sum()),
        # Chaînes vides ou composées uniquement d'espaces
        "num_empty_or_whitespace": int((df[text_col].str.strip() == "").sum()),
        # Comptage des doublons exacts (lignes de texte strictement identiques)
        "num_exact_duplicates": int(df[text_col].duplicated().sum()),
        "length_stats": {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
            "min": int(lengths.min()),
            "25%": float(lengths.quantile(0.25)),
            "50%": float(lengths.quantile(0.50)),
            "75%": float(lengths.quantile(0.75)),
            "max": int(lengths.max()),
        },
        # Identification des lignes composées d'un seul mot (peu informatives pour l'entraînement)
        "num_single_word_lines": int(
            (df[text_col].str.strip().str.split().str.len() == 1).sum()
        ),
        # Identification des lignes contenant des guillemets, à nettoyer avant l'entraînement
        "num_lines_with_quotes": int(df[text_col].str.contains('"', na=False).sum()),
        # Identification des lignes contenant des balises HTML, à filtrer
        "num_lines_with_html": int(
            df[text_col].str.contains(r"<[^>]+>", regex=True, na=False).sum()
        ),
        # Identification des lignes contenant des URLs, à filtrer
        "num_lines_with_urls": int(
            df[text_col].str.contains(r"http[s]?://\S+", regex=True, na=False).sum()
        ),
    }


def clean_dataframe(df: pd.DataFrame, text_col: str, cfg: dict) -> pd.DataFrame:
    """Reprend exactement les étapes de la section 'Cleaning' du notebook."""

    # Suppression des colonnes non utiles pour la suite ('sources', 'len' calculée précédemment)
    if cfg["drop_columns"]:
        existing = [c for c in cfg["drop_columns"] if c in df.columns]
        df = df.drop(columns=existing)

    # Supprimer les NaN
    if cfg["remove_nan"]:
        df = df.dropna(subset=[text_col])

    # Supprimer les chaines vides ou whitespace
    if cfg["remove_empty_or_whitespace"]:
        df = df[df[text_col].str.strip() != ""]

    # supprimer les nombres (lignes composées uniquement de chiffres)
    if cfg["remove_numeric_only_lines"]:
        df = df[~df[text_col].str.strip().str.match(r"^\d+$")]

    # supprimer les lignes avec un seul mot
    if cfg["remove_single_word_lines"]:
        df = df[df[text_col].str.strip().str.split().str.len() > 1]

    # on supprime les guillemets du texte
    if cfg["remove_quotes"]:
        df[text_col] = df[text_col].str.replace('"', "", regex=False)

    # suppression des balises HTML
    if cfg["remove_html_tags"]:
        df = df[~df[text_col].str.contains(r"<[^>]+>", regex=True, na=False)]

    # suppression des lignes contenant des URLs
    if cfg["remove_urls"]:
        df = df[~df[text_col].str.contains(r"http[s]?://\S+", regex=True, na=False)]

    # suppression des textes trop courts (longueur minimale de caractères)
    if cfg.get("min_char_length"):
        df = df[df[text_col].str.strip().str.len() >= cfg["min_char_length"]]

    # Mise en minuscule de tout le texte (normalisation avant tokenisation)
    if cfg["lowercase"]:
        df[text_col] = df[text_col].str.lower()

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers config.json")
    parser.add_argument("--skip-download", action="store_true", help="Utiliser un fichier local existant")
    parser.add_argument("--local-file", help="Fichier parquet local (si --skip-download)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["input"]["text_column"]
    version = cfg["output"]["version"]

    # 1. Chargement (depuis HF Hub ou local)
    if args.skip_download:
        print(f"Chargement local de {args.local_file}...")
        df = pd.read_parquet(args.local_file)
    else:
        df = download_raw_dataset(cfg)

    # 2. Stats avant nettoyage
    stats_before = compute_stats(df, text_col)

    # 3. Nettoyage
    df_clean = clean_dataframe(df, text_col, cfg["cleaning"])
    print(f"{len(df_clean)} lignes après nettoyage.")

    # 4. Stats après nettoyage
    stats_after = compute_stats(df_clean, text_col)

    # 5. Sauvegarde locale versionnée
    out_dir = Path(cfg["output"]["local_output_dir"]) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_{version}.parquet"
    df_clean.to_parquet(out_path)
    print(f"Données nettoyées sauvegardées : {out_path}")

    # 6. Sauvegarde des statistiques
    stats_dir = Path(cfg["output"]["stats_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"stats_{version}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"before": stats_before, "after": stats_after}, f, ensure_ascii=False, indent=2)
    print(f"Statistiques sauvegardées : {stats_path}")


if __name__ == "__main__":
    main()