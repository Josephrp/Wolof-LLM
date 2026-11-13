"""Métriques d'évaluation — définition UNIQUE, partagée par tous les runs.

Avoir une seule définition du BPB dans tout le projet est essentiel : deux
implémentations légèrement différentes rendraient les runs non comparables
sans que rien ne le signale.
"""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def compute_bpb(model, tokenizer, textes, max_len=1024, device="cuda:0"):
    """Bits-per-byte : entropie croisée normalisée par les octets UTF-8.

        BPB = somme(cross-entropy en nats) / (ln2 * nombre d'octets)

    Contrairement à la perplexité — calculée *par token* — le BPB ne dépend
    pas du découpage. C'est la SEULE métrique valide pour comparer deux
    modèles dont les tokenizers diffèrent : un tokenizer qui produit moins de
    tokens obtiendrait mécaniquement une perplexité différente sans être
    meilleur ni pire.

    La perplexité reste retournée, comparable uniquement à tokenizer identique.
    """
    etait_en_train = model.training
    model.eval()
    nats = octets = tokens = 0
    for t in textes:
        t = str(t)
        b = len(t.encode("utf-8"))
        if b == 0:
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)[:max_len]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=device)
        logits = model(x).logits[0, :-1]
        nats += F.cross_entropy(logits.float(), x[0, 1:], reduction="sum").item()
        octets += b
        tokens += len(ids) - 1
    if etait_en_train:
        model.train()
    return {"bpb": nats / (math.log(2) * octets),
            "ppl": math.exp(nats / tokens),
            "n_octets": octets,
            "n_tokens": tokens}


def compute_fertility(tokenizer, textes):
    """Fertilité : nombre moyen de tokens par mot.

    Mesurée EN CONTEXTE (phrases entières, espaces compris). Mesurer sur des
    mots isolés donne des chiffres trompeurs, un BPE byte-level encodant
    différemment un mot précédé d'un espace.
    """
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
    return {"tokens_par_mot": nt / max(nm, 1),
            "tokens_par_caractere": nt / max(nc, 1),
            "n_mots": nm,
            "n_tokens": nt}
