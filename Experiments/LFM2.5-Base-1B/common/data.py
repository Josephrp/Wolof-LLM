"""Chargement des splits et mise en blocs (packing) des textes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset


def charger_splits_hub(
    dataset_id: str = "galsenai/cleaned_data_for_clm",
    max_rows: int | None = None,
    token: str | None = None,
):
    """Retourne (textes_train, textes_test) depuis le Hub.

    Split d'evaluation = `test` (clean). Passer `max_rows` pour un dry-run.
    """
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    ds = load_dataset(dataset_id, **kwargs)
    if "train" not in ds or "test" not in ds:
        raise KeyError(
            f"{dataset_id}: expected train/test splits, got {list(ds.keys())}"
        )
    train = ds["train"]
    test = ds["test"]
    if max_rows is not None:
        train = train.select(range(min(max_rows, len(train))))
        test = test.select(range(min(max_rows, len(test))))
    if "text" not in train.column_names:
        raise KeyError(f"{dataset_id}: missing 'text' column")
    return (
        [str(t) for t in train["text"]],
        [str(t) for t in test["text"]],
    )


def charger_splits_local(
    racine,
    split_eval="wolof_eval_clean_v1.parquet",
    data_root=None,
):
    """Retourne (textes_train, textes_eval) depuis les splits parquet figes.

    Noms de fichiers aligns sur SmolLM3. Le split d'evaluation par defaut est
    la version *propre*, debarrassee des quasi-doublons presents dans le train.
    """
    d = Path(data_root) if data_root else Path(racine) / "data"
    train_path = d / "wolof_bpe_train_v1.parquet"
    eval_path = d / split_eval
    if not train_path.exists() or not eval_path.exists():
        manquants = [str(p) for p in (train_path, eval_path) if not p.exists()]
        raise FileNotFoundError(
            "Splits parquet introuvables : "
            + ", ".join(manquants)
            + "\nLancer d'abord scripts/clm/01_prepare_data.py avec le meme data_root,"
            " ou utiliser charger_splits_hub()."
        )
    train = pd.read_parquet(train_path)
    ev = pd.read_parquet(eval_path)
    return (
        train["text"].astype(str).tolist(),
        ev["text"].astype(str).tolist(),
    )


# Alias SmolLM3
charger_splits = charger_splits_local


def paqueter(textes, tokenizer, longueur, limite=None, eos_token_id=None):
    """Concatene les textes puis decoupe en blocs de `longueur` tokens.

    Sans packing, la majorite du calcul partirait en remplissage : les textes
    du corpus ont une mediane d'environ 97 caracteres pour une fenetre de
    1 024 tokens.

    Le packing depend du tokenizer : il doit etre refait pour chaque
    configuration de vocabulaire.

    `eos_token_id` : override optionnel (defaut = tokenizer.eos_token_id).
    Sur LFM2.5 65k, EOS attendu = 7.
    """
    if limite:
        textes = textes[:limite]
    eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    assert eos is not None, "eos_token_id manquant (tokenizer ou override)"
    print(f"  paqueter: eos_token_id={eos} -> {tokenizer.decode([eos])!r}")
    ids = []
    for t in textes:
        ids.extend(tokenizer.encode(t, add_special_tokens=False) + [eos])
    n = (len(ids) // longueur) * longueur
    blocs = [ids[i:i + longueur] for i in range(0, n, longueur)]
    return Dataset.from_dict({
        "input_ids": blocs,
        "labels": [b[:] for b in blocs],
    })


def _safe_json_loads(value, default=None):
    """Parse JSON strings safely; leave already-parsed objects untouched."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return default
    return default


def _normalize_sft_row(row: dict) -> dict:
    out = dict(row)
    if "meta" in out:
        out["meta"] = _safe_json_loads(out["meta"], default={})
    messages = out.get("messages")
    if isinstance(messages, str):
        messages = _safe_json_loads(messages, default=[])
    if isinstance(messages, list):
        cleaned = []
        for msg in messages:
            if not isinstance(msg, dict):
                cleaned.append(msg)
                continue
            m = dict(msg)
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                fixed = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        fixed.append(tc)
                        continue
                    t = dict(tc)
                    fn = t.get("function")
                    if isinstance(fn, dict):
                        f = dict(fn)
                        if "arguments" in f:
                            f["arguments"] = _safe_json_loads(
                                f["arguments"], default=f["arguments"]
                            )
                        t["function"] = f
                    elif "arguments" in t:
                        t["arguments"] = _safe_json_loads(
                            t["arguments"], default=t["arguments"]
                        )
                    fixed.append(t)
                m["tool_calls"] = fixed
            cleaned.append(m)
        out["messages"] = cleaned
    return out


def charger_sft_messages(
    path_or_hub,
    split: str = "train",
    max_rows: int | None = None,
    token: str | None = None,
):
    """Charge des lignes ChatML-ish (messages/task/source/meta).

    - Hub id (ex. `galsenai/wolof_sft`) via `load_dataset`
    - Fichier local `.jsonl` / `.json` / `.parquet`
    - Dossier local contenant `train.jsonl` / `test.jsonl`

    `meta` et `tool_calls[].arguments` (souvent des strings JSON) sont parses
    via json.loads de facon tolerant.
    """
    p = Path(path_or_hub)
    rows: list[dict] = []

    if p.exists():
        if p.is_dir():
            candidates = [
                p / f"{split}.parquet",
                p / f"{split}.jsonl",
                p / f"{split}.json",
                p / "data" / f"{split}.parquet",
                p / "data" / f"{split}.jsonl",
                p / "data" / f"{split}.json",
            ]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                raise FileNotFoundError(
                    f"Aucun {split}.parquet/.jsonl/.json sous {p}"
                )
            p = path
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            if max_rows is not None:
                df = df.head(max_rows)
            rows = df.to_dict(orient="records")
        elif p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_rows is not None and i >= max_rows:
                        break
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "messages" in data:
                rows = [data]
            else:
                raise ValueError(f"JSON SFT non reconnu: {p}")
            if max_rows is not None:
                rows = rows[:max_rows]
        else:
            raise ValueError(f"Format SFT non supporte: {p}")
    else:
        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        ds = load_dataset(path_or_hub, **kwargs)
        if split not in ds:
            # Some Hub layouts expose data files as config; try data/train.jsonl
            raise KeyError(
                f"{path_or_hub}: split '{split}' absent, got {list(ds.keys())}"
            )
        subset = ds[split]
        if max_rows is not None:
            subset = subset.select(range(min(max_rows, len(subset))))
        rows = [dict(r) for r in subset]

    return [_normalize_sft_row(r) for r in rows]
