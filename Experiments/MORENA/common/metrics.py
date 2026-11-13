"""BPB helpers for MORENA native forward (logits tensor, not .logits)."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def compute_bpb_morena(model, tokenizer, texts, *, max_len=1024, device=None, eos_id=None):
    """Bits-per-byte with MORENA Transformer forward(idx, pos, None, T, None).

    Tokenizer is HuggingFace `tokenizers.Tokenizer` (encode(...).ids).
    Documents conventionally start with <eos>; we do **not** prepend EOS for BPB
    scoring of raw corpus text (matches evaluate_clm first-token handling when
    no BOS).
    """
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    window = max(2, min(max_len, 4096))
    chunk_size = window - 1

    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    n_texts_scored = 0

    for text in texts:
        text = str(text)
        n_bytes = len(text.encode("utf-8"))
        if n_bytes == 0:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 2:
            continue
        first_piece = tokenizer.decode(ids[:1], skip_special_tokens=True)
        unscored_bytes = len(first_piece.encode("utf-8"))

        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            if start == 0:
                input_ids = chunk
            else:
                input_ids = [ids[start - 1]] + chunk
            if len(input_ids) < 2:
                continue
            x = torch.tensor([input_ids], device=device)
            pos = torch.arange(x.shape[1], device=device).unsqueeze(0)
            logits = model(x, pos, None, x.shape[1], None)[0, :-1].float()
            targets = x[0, 1:]
            total_nats += F.cross_entropy(logits, targets, reduction="sum").item()
            total_tokens += int(targets.numel())

        total_bytes += max(0, n_bytes - unscored_bytes)
        n_texts_scored += 1

    if was_training:
        model.train()
    if total_bytes == 0 or total_tokens == 0:
        return {
            "bpb": float("nan"),
            "ppl": float("nan"),
            "n_bytes": total_bytes,
            "n_tokens": total_tokens,
            "n_texts_scored": n_texts_scored,
        }
    return {
        "bpb": total_nats / (math.log(2) * total_bytes),
        "ppl": math.exp(total_nats / total_tokens),
        "n_bytes": total_bytes,
        "n_tokens": total_tokens,
        "n_texts_scored": n_texts_scored,
    }


def fertility(tokenizer, texts: list[str]) -> float:
    tokens = 0
    words = 0
    for text in texts:
        text = str(text).strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        tokens += len(ids)
        words += max(1, len(text.split()))
    return tokens / words if words else float("nan")
