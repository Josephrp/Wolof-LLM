"""Metriques d'evaluation — definition UNIQUE, partagee par tous les runs.

Avoir une seule definition du BPB dans tout le projet est essentiel : deux
implementations legerement differentes rendraient les runs non comparables
sans que rien ne le signale.
"""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def compute_bpb(model, tokenizer, textes, max_len=1024, device=None):
    """Bits-per-byte : entropie croisee normalisee par les octets UTF-8.

        BPB = somme(cross-entropy en nats) / (ln2 * nombre d'octets)

    Contrairement a la perplexite — calculee *par token* — le BPB ne depend
    pas du decoupage. C'est la SEULE metrique valide pour comparer deux
    modeles dont les tokenizers different : un tokenizer qui produit moins de
    tokens obtiendrait mecaniquement une perplexite differente sans etre
    meilleur ni pire.

    La perplexite reste retournee, comparable uniquement a tokenizer identique.
    """
    etait_en_train = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    limits = [max_len]
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and 1 < value < 10**6:
            limits.append(value)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 1 < tok_limit < 10**6:
        limits.append(tok_limit)
    window = max(2, min(limits))
    chunk_size = window - 1
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = getattr(tokenizer, "cls_token_id", None)

    nats = octets = tokens = n_textes = 0
    for t in textes:
        t = str(t)
        b = len(t.encode("utf-8"))
        if b == 0:
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)
        if not ids or (bos_id is None and len(ids) < 2):
            continue
        unscored_bytes = 0
        if bos_id is None:
            first = tokenizer.decode(ids[:1], skip_special_tokens=True)
            unscored_bytes = len(first.encode("utf-8"))
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            if start == 0:
                input_ids = [bos_id, *chunk] if bos_id is not None else chunk
            else:
                input_ids = [ids[start - 1], *chunk]
            if len(input_ids) < 2:
                continue
            x = torch.tensor([input_ids], device=device)
            logits = model(x).logits[0, :-1].float()
            targets = x[0, 1:]
            nats += F.cross_entropy(logits, targets, reduction="sum").item()
            tokens += targets.numel()
        octets += max(0, b - unscored_bytes)
        n_textes += 1
    if etait_en_train:
        model.train()
    if not octets or not tokens:
        return {
            "bpb": float("nan"),
            "ppl": float("nan"),
            "n_octets": octets,
            "n_bytes": octets,
            "n_tokens": tokens,
            "n_textes_scored": n_textes,
        }
    return {
        "bpb": nats / (math.log(2) * octets),
        "ppl": math.exp(nats / tokens),
        "n_octets": octets,
        "n_bytes": octets,
        "n_tokens": tokens,
        "n_textes_scored": n_textes,
    }


def compute_fertility(tokenizer, textes):
    """Fertilite : nombre moyen de tokens par mot.

    Mesuree EN CONTEXTE (phrases entieres, espaces compris). Mesurer sur des
    mots isoles donne des chiffres trompeurs, un BPE byte-level encodant
    differemment un mot precede d'un espace.
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
