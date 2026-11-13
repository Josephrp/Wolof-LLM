#!/usr/bin/env python3
"""Evaluate MORENA with native Transformer forward (protocol: morena-native).

BPB uses the same formula as evaluate_clm.py. Do not compare PPL across tokenizer
families; BPB is the cross-family gold metric.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from tqdm.auto import tqdm

EVAL_ROOT = Path(__file__).resolve().parents[1]
MORENA_ROOT = EVAL_ROOT.parent / "Experiments" / "MORENA"
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MORENA_ROOT))

from verbalizer import verbalize_text  # noqa: E402
from common.load import load_morena  # noqa: E402
from common.metrics import compute_bpb_morena, fertility  # noqa: E402


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_texts(cfg: dict, config_path: Path) -> list[str]:
    corpus = Path(cfg["corpus_csv"])
    if not corpus.is_absolute():
        candidates = [
            (config_path.parent / corpus).resolve(),
            (EVAL_ROOT / corpus).resolve(),
        ]
        stripped = corpus.as_posix()
        while stripped.startswith("../"):
            stripped = stripped[3:]
        candidates.append((EVAL_ROOT / stripped).resolve())
        corpus = next((c for c in candidates if c.is_file()), candidates[-1])
    frame = pd.read_csv(corpus)
    column = cfg.get("text_column", "text")
    texts = frame[column].dropna().astype(str).tolist()
    n_texts = cfg.get("n_texts")
    if n_texts is not None and str(n_texts).lower() != "all":
        texts = texts[: int(n_texts)]
    if cfg.get("apply_verbalization", True):
        texts = [verbalize_text(t) for t in texts]
    return texts


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt_ids, *, max_new, temperature=0.0, eos_id=None):
    device = next(model.parameters()).device
    x = torch.tensor([prompt_ids], device=device)
    out = []
    for _ in range(max_new):
        pos = torch.arange(x.shape[1], device=device).unsqueeze(0)
        logits = model(x, pos, None, x.shape[1], None)[:, -1, :].float()
        if temperature <= 0:
            nxt = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        if eos_id is not None and nxt == eos_id:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def compute_bleu_chrf(model, tokenizer, texts, *, n_samples=100, seed=0, prompt_ratio=0.5, length_ratio=1.3):
    try:
        from sacrebleu import corpus_bleu, corpus_chrf
    except ImportError as exc:
        raise RuntimeError("sacrebleu required for BLEU/chrF") from exc

    rng = random.Random(seed)
    candidates = [t for t in texts if len(str(t).split()) >= 8]
    if not candidates:
        return {"BLEU": float("nan"), "chrF": float("nan"), "n_bleu_pairs": 0}
    sample = candidates if n_samples is None or str(n_samples).lower() == "all" else (
        candidates if len(candidates) <= n_samples else rng.sample(candidates, n_samples)
    )
    eos_id = tokenizer.token_to_id("<eos>")
    hyps, refs = [], []
    for text in tqdm(sample, desc="BLEU", unit="ex"):
        words = str(text).split()
        cut = max(1, int(len(words) * prompt_ratio))
        prompt = " ".join(words[:cut])
        reference = " ".join(words[cut:])
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if not ids:
            continue
        # Docs start with EOS for generation (matches load_example).
        prompt_ids = [eos_id] + ids if eos_id is not None else ids
        max_new = max(16, int(len(tokenizer.encode(reference, add_special_tokens=False).ids) * length_ratio))
        gen_ids = generate_continuation(model, tokenizer, prompt_ids, max_new=max_new, eos_id=eos_id)
        hyp = tokenizer.decode(gen_ids, skip_special_tokens=True)
        hyps.append(hyp)
        refs.append(reference)
    if not hyps:
        return {"BLEU": float("nan"), "chrF": float("nan"), "n_bleu_pairs": 0}
    bleu = corpus_bleu(hyps, [refs]).score
    chrf = corpus_chrf(hyps, [refs]).score
    return {
        "BLEU": round(float(bleu), 2),
        "chrF": round(float(chrf), 2),
        "n_bleu_pairs": len(hyps),
        "bleu_seed": seed,
        "bleu_prompt_ratio": prompt_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MORENA (native)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    texts = load_texts(cfg, config_path)

    allow_cpu = bool(cfg.get("allow_cpu", False))
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA required (set allow_cpu: true to override)")

    model, tokenizer, meta, raw_cfg = load_morena(
        cfg["base_model_id"],
        attn_mode=cfg.get("attn_mode", "sdpa"),
        cache_dir=Path(cfg["cache_dir"]) if cfg.get("cache_dir") else None,
    )
    max_len = int(cfg.get("bpb_max_len", 1024))
    bpb = compute_bpb_morena(model, tokenizer, texts, max_len=max_len)
    gen = {}
    if cfg.get("compute_bleu", True):
        gen = compute_bleu_chrf(
            model,
            tokenizer,
            texts,
            n_samples=cfg.get("bleu_n_texts", 100),
            seed=int(cfg.get("bleu_seed", 0)),
            prompt_ratio=float(cfg.get("bleu_prompt_ratio", 0.5)),
            length_ratio=float(cfg.get("bleu_length_ratio", 1.3)),
        )

    fert = fertility(tokenizer, texts[: min(500, len(texts))])
    result = {
        "protocol": "morena-native",
        "run": meta["run"],
        "base_model_id": cfg["base_model_id"],
        "BPB": round(float(bpb["bpb"]), 4) if bpb["bpb"] == bpb["bpb"] else None,
        "perplexity": round(float(bpb["ppl"]), 2) if bpb["ppl"] == bpb["ppl"] else None,
        "n_bytes": bpb["n_bytes"],
        "n_tokens": bpb["n_tokens"],
        "n_texts_scored": bpb["n_texts_scored"],
        "n_texts_evaluated": len(texts),
        "fertility_tokens_per_word": round(float(fert), 4) if fert == fert else None,
        "vocab_size": meta["vocab_size"],
        "device": meta["device"],
        "bpb_max_len": max_len,
        "model": raw_cfg.get("model"),
        **gen,
        "note": "BPB is cross-tokenizer gold; PPL only within MORENA 65k tokenizer.",
    }

    out = Path(cfg["output_json"])
    if not out.is_absolute():
        out = (EVAL_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
