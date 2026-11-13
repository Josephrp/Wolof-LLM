#!/usr/bin/env python3
"""
evaluate_clm.py
---------------
Evaluate a single Hugging Face model on a shared text corpus and write the
results (BPB, perplexity, BLEU, chrF) to a JSON file.

Text is normalized through the project's Wolof verbalizer (imported from
verbalizer.py) before any metric is computed.

All parameters are loaded from a YAML config file instead of CLI flags.

BPB / perplexity are computed in this file (see compute_bpb below) so that a
single, self-contained definition is used for every run.

Requirements
------------
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers pyyaml sacrebleu tqdm accelerate

Example Usage
-------------
python evaluate_clm.py --config ../configs/clm/qwen.yaml
"""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verbalizer import verbalize_text


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(config_path):
    """Load all run parameters from a YAML config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

def verbalize_corpus(texts):
    """Apply verbalize_text() (from verbalizer.py) to every text in the corpus."""
    return [verbalize_text(t) for t in texts]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_corpus(corpus_csv, text_column, n_texts=None, apply_verbalization=True):
    """
    Load evaluation texts from a CSV corpus (as produced by the HF download
    script). Set n_texts to None or "all" to use the whole corpus.
    """
    df = pd.read_csv(corpus_csv)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {corpus_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].dropna().astype(str).tolist()

    if n_texts is not None and str(n_texts).lower() != "all":
        texts = texts[: int(n_texts)]

    if apply_verbalization:
        texts = verbalize_corpus(texts)

    return texts


def load_model(base_model_id, allow_cpu=False, subfolder=None, tokenizer_subfolder=None):
    """
    Load a model and tokenizer directly from the Hugging Face Hub.

    subfolder selects a directory inside the repo (e.g. "v2/ext4000-lora") when
    several checkpoints live in the same repository. Leave it unset to load from
    the repo root.

    tokenizer_subfolder defaults to subfolder, but can be set separately: a
    checkpoint directory often ships only model weights, with the tokenizer
    files kept at the repo root. Set it to "" to force the root explicitly.
    """
    if tokenizer_subfolder is None:
        tokenizer_subfolder = subfolder

    # from_pretrained expects "" (not None) to mean "repo root".
    subfolder = subfolder or ""
    tokenizer_subfolder = tokenizer_subfolder or ""

    # clean_up_tokenization_spaces is destructive for BPE (it strips spaces
    # before punctuation), which would corrupt both the BLEU hypotheses and
    # the first-token byte accounting in compute_bpb.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        subfolder=tokenizer_subfolder,
        clean_up_tokenization_spaces=False,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        device_map = {"": 0}
        device_name = torch.cuda.get_device_name(0)
    elif allow_cpu:
        device_map = "cpu"
        device_name = "cpu"
    else:
        raise RuntimeError(
            "CUDA is not available. A multi-billion-parameter model in bfloat16 "
            "on CPU is slow enough to look like a hang. Fix the driver/torch "
            "mismatch, or set allow_cpu: true in the config to run anyway."
        )

    print(f"Device: {device_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        subfolder=subfolder,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()

    # "run" is the identifier that ends up in the results JSON, so it has to
    # distinguish two checkpoints coming from the same repo.
    run_name = f"{base_model_id}/{subfolder}" if subfolder else base_model_id

    metadata = {
        "run": run_name,
        "base_model_id": base_model_id,
        "subfolder": subfolder or None,
        "vocab_size": len(tokenizer),
        "device": device_name,
    }

    return model, tokenizer, metadata


# --------------------------------------------------------------------------- #
# Metrics: bits-per-byte and perplexity
# --------------------------------------------------------------------------- #

def _resolve_context_length(model, tokenizer, requested):
    """Clamp the requested window to what the model/tokenizer actually supports."""
    limits = []
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and 0 < value < 10 ** 6:
            limits.append(value)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 10 ** 6:
        limits.append(tok_max)

    window = min([requested] + limits) if limits else requested
    return max(2, int(window))


@torch.no_grad()
def compute_bpb(model, tokenizer, texts, max_len=1024):
    """
    Bits-per-byte: cross-entropy normalized by the number of UTF-8 bytes.

        BPB = sum(cross-entropy in nats) / (ln2 * n_bytes)

    Unlike perplexity — computed *per token* — BPB does not depend on the
    tokenization, so it is the only metric that is valid for comparing models
    whose tokenizers differ. Perplexity is still returned, but it is only
    comparable across runs sharing an identical tokenizer.

    Implementation notes (these matter for the numbers to be correct):

    * Texts longer than the context window are split into consecutive
      non-overlapping windows instead of being truncated. Truncating tokens
      while counting the bytes of the *whole* text silently deflates BPB.
    * Every token is scored exactly once. A causal LM cannot score its first
      token, so the first token of each window is supplied as context: either
      a BOS token, or (when the tokenizer has none) the last token of the
      previous window. For the very first window without BOS, the first token
      is unscored and its bytes are excluded from the denominator, keeping
      numerator and denominator over the same span of text.
    * Logits are upcast to float32 before the cross-entropy so that a
      bf16-loaded model does not lose precision in the sum.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    window = _resolve_context_length(model, tokenizer, max_len)
    chunk_size = window - 1  # one slot is reserved for the context token

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = getattr(tokenizer, "cls_token_id", None)

    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    n_texts_scored = 0

    progress = tqdm(texts, desc="BPB", unit="text", dynamic_ncols=True)
    for text in progress:
        text = str(text)
        n_bytes = len(text.encode("utf-8"))
        if n_bytes == 0:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        # Bytes covered by tokens we will actually score.
        unscored_bytes = 0
        if bos_id is None:
            if len(ids) < 2:
                continue
            first_piece = tokenizer.decode(ids[:1], skip_special_tokens=True)
            unscored_bytes = len(first_piece.encode("utf-8"))

        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            if start == 0:
                # First window: BOS as context, or the first token itself.
                input_ids = ([bos_id] + chunk) if bos_id is not None else chunk
            else:
                # Carry one token of context over from the previous window.
                input_ids = [ids[start - 1]] + chunk

            if len(input_ids) < 2:
                continue

            x = torch.tensor([input_ids], device=device)
            logits = model(x).logits[0, :-1].float()
            targets = x[0, 1:]
            total_nats += F.cross_entropy(logits, targets, reduction="sum").item()
            total_tokens += targets.numel()

        total_bytes += max(0, n_bytes - unscored_bytes)
        n_texts_scored += 1

        # Running values: the metric is cumulative, so these converge to the
        # final number and make a bad run obvious well before it finishes.
        if total_bytes > 0 and total_tokens > 0:
            progress.set_postfix(
                bpb=f"{total_nats / (math.log(2) * total_bytes):.4f}",
                ppl=f"{math.exp(total_nats / total_tokens):.1f}",
                kB=f"{total_bytes / 1000:.0f}",
                refresh=False,
            )

    progress.close()
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


# --------------------------------------------------------------------------- #
# Metrics: BLEU
# --------------------------------------------------------------------------- #

def compute_bleu(
    model,
    tokenizer,
    texts,
    prompt_ratio=0.5,
    max_new_tokens=256,
    length_ratio=1.3,
    min_new_tokens=16,
    n_samples=100,
    seed=0,
):
    """
    Compute corpus-level BLEU and chrF by prompting the model with the first
    part of each text and comparing its generated continuation against the
    real continuation (the rest of the text) as reference.

    Because each example costs one autoregressive generate() call, these are
    computed on a random subset of n_samples texts (set n_samples to None or
    "all" for the whole corpus). The subset is drawn with a fixed seed from
    the texts that are long enough to split, so every run scores the SAME
    examples and the numbers stay comparable across models. BPB is still
    computed on the full corpus; it is the metric carrying the argument.

    The generation budget is derived per example from the length of the
    reference *in this model's own tokens* (length_ratio x reference length,
    floored at min_new_tokens, capped at max_new_tokens). A fixed budget would
    not be tokenizer-neutral: a model whose tokenizer fragments Wolof heavily
    would be cut off mid-continuation and penalized for its fertility rather
    than its modelling quality. max_new_tokens is therefore a safety ceiling,
    not the operative constraint.

    chrF is reported alongside BLEU because it is character-based, and so far
    less sensitive to tokenization and to the morphological variation of
    agglutinative and code-switched text.
    """
    import sacrebleu

    # Keep only texts that can actually be split into prompt + reference, then
    # sample from those, so n_samples is the number of scored pairs rather
    # than a number that silently shrinks.
    eligible = [t for t in texts if len(str(t).split()) >= 4]
    if n_samples is not None and str(n_samples).lower() != "all":
        n_samples = int(n_samples)
        if 0 < n_samples < len(eligible):
            eligible = random.Random(seed).sample(eligible, n_samples)

    hypotheses = []
    references = []
    running_chrf = 0.0

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    window = _resolve_context_length(model, tokenizer, 1024)
    prompt_max_len = max(1, window - max_new_tokens)

    progress = tqdm(eligible, desc="BLEU/chrF", unit="text", dynamic_ncols=True)
    for text in progress:
        words = text.split()
        if len(words) < 4:
            continue

        split_idx = max(1, int(len(words) * prompt_ratio))
        prompt = " ".join(words[:split_idx])
        reference = " ".join(words[split_idx:])
        if not reference:
            continue

        n_ref_tokens = len(tokenizer.encode(reference, add_special_tokens=False))
        budget = min(max_new_tokens, max(min_new_tokens, int(n_ref_tokens * length_ratio)))

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=prompt_max_len
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        hypotheses.append(generated_text)
        references.append(reference)

        # Sentence-level chrF averaged so far: only an indication of progress.
        # The corpus-level scores returned below are the reported figures.
        running_chrf += sacrebleu.sentence_chrf(generated_text, [reference]).score
        progress.set_postfix(
            pairs=len(hypotheses),
            chrf_approx=f"{running_chrf / len(hypotheses):.1f}",
            refresh=False,
        )

    progress.close()
    if not hypotheses:
        return {"bleu": 0.0, "chrf": 0.0, "n_pairs": 0}

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": bleu.score, "chrf": chrf.score, "n_pairs": len(hypotheses)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face model (BPB, perplexity, BLEU) and write results to JSON."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML config file (default: %(default)s)"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    # configs live under Evaluation/configs/{clm,gguf}/...
    eval_root = config_path.parents[2] if config_path.parent.name in {"clm", "gguf"} else Path.cwd()

    def resolve_path(value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [
            (Path.cwd() / path).resolve(),
            (config_path.parent / path).resolve(),
            (eval_root / path).resolve(),
        ]
        # Common legacy form: ../data/foo from configs/clm → Evaluation/data/foo
        stripped = path.as_posix()
        while stripped.startswith("../"):
            stripped = stripped[3:]
        candidates.append((eval_root / stripped).resolve())
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Prefer eval_root stripped path for outputs that do not exist yet
        return (eval_root / stripped).resolve()

    corpus_csv = resolve_path(config["corpus_csv"])
    text_column = config.get("text_column", "text")
    n_texts = config.get("n_texts")  # None or "all" -> use the whole corpus
    apply_verbalization = config.get("apply_verbalization", True)
    bpb_max_len = int(config.get("bpb_max_len", 1024))
    bleu_prompt_ratio = config.get("bleu_prompt_ratio", 0.5)
    bleu_max_new_tokens = int(config.get("bleu_max_new_tokens", 256))
    bleu_length_ratio = float(config.get("bleu_length_ratio", 1.3))
    bleu_min_new_tokens = int(config.get("bleu_min_new_tokens", 16))
    bleu_n_texts = config.get("bleu_n_texts", 100)
    bleu_seed = int(config.get("bleu_seed", 0))
    base_model_id = config["base_model_id"]
    subfolder = config.get("subfolder")  # None/empty -> load from the repo root
    tokenizer_subfolder = config.get("tokenizer_subfolder")  # defaults to subfolder
    output_json = resolve_path(config["output_json"])

    print(f"corpus_csv = {corpus_csv}")
    texts = load_corpus(corpus_csv, text_column, n_texts, apply_verbalization)
    print(f"Evaluating on {len(texts)} texts (verbalization: {apply_verbalization})")

    model, tokenizer, metadata = load_model(
        base_model_id,
        allow_cpu=config.get("allow_cpu", False),
        subfolder=subfolder,
        tokenizer_subfolder=tokenizer_subfolder,
    )
    print(f"Loaded model: {metadata['run']}")

    bpb_result = compute_bpb(model, tokenizer, texts, max_len=bpb_max_len)
    bleu_result = compute_bleu(
        model,
        tokenizer,
        texts,
        prompt_ratio=bleu_prompt_ratio,
        max_new_tokens=bleu_max_new_tokens,
        length_ratio=bleu_length_ratio,
        min_new_tokens=bleu_min_new_tokens,
        n_samples=bleu_n_texts,
        seed=bleu_seed,
    )

    results = {
        **metadata,
        "BPB": round(bpb_result["bpb"], 4),
        "perplexity": round(bpb_result["ppl"], 2),
        "BLEU": round(bleu_result["bleu"], 2),
        "chrF": round(bleu_result["chrf"], 2),
        "n_texts_evaluated": len(texts),
        "n_texts_scored": bpb_result["n_texts_scored"],
        "n_bleu_pairs": bleu_result["n_pairs"],
        "n_bytes": bpb_result["n_bytes"],
        "n_tokens": bpb_result["n_tokens"],
        "bpb_max_len": bpb_max_len,
        "bleu_prompt_ratio": bleu_prompt_ratio,
        "bleu_length_ratio": bleu_length_ratio,
        "bleu_n_texts": bleu_n_texts,
        "bleu_seed": bleu_seed,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[ok] results written to: {output_json}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()