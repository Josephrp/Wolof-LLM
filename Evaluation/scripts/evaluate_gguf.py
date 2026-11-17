#!/usr/bin/env python3
"""Evaluate a GGUF checkpoint with llama.cpp (PPL + BLEU/chrF).

Protocol tag: gguf-llama-cpp. Results are NOT HF-BPB-equivalent; use HF CLM
JSON as the cross-tokenizer gold metric and these numbers for quant tracking.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verbalizer import verbalize_text  # noqa: E402


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_texts(cfg: dict, config_path: Path) -> list[str]:
    corpus = Path(cfg["corpus_csv"])
    eval_root = config_path.parents[2] if config_path.parent.name in {"clm", "gguf"} else EVAL_ROOT
    if not corpus.is_absolute():
        candidates = [
            (config_path.parent / corpus).resolve(),
            (eval_root / corpus).resolve(),
        ]
        stripped = corpus.as_posix()
        while stripped.startswith("../"):
            stripped = stripped[3:]
        candidates.append((eval_root / stripped).resolve())
        corpus = next((c for c in candidates if c.is_file()), candidates[-1])
    column = cfg.get("text_column", "text")
    frame = pd.read_csv(corpus)
    texts = frame[column].dropna().astype(str).tolist()
    n_texts = cfg.get("n_texts")
    if n_texts is not None and str(n_texts).lower() != "all":
        texts = texts[: int(n_texts)]
    if cfg.get("apply_verbalization", True):
        texts = [verbalize_text(t) for t in texts]
    return texts


def resolve_bin(cfg: dict, name: str) -> Path:
    root = Path(cfg.get("llamacpp_root", "/mnt/volume_w2l8nkp/tools/llama.cpp"))
    candidates = [
        root / "build" / "bin" / name,
        root / "build" / name,
        Path(cfg[name]) if cfg.get(name) else None,
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    raise FileNotFoundError(f"{name} not found under {root}")


def parse_ppl(text: str) -> float | None:
    # Typical: Final estimate: PPL = 12.3456 ± ...
    patterns = [
        r"Final estimate:\s*PPL\s*=\s*([0-9.]+)",
        r"perplexity:\s*([0-9.]+)",
        r"PPL\s*=\s*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def run_perplexity(cfg: dict, texts: list[str]) -> dict:
    llama_ppl = resolve_bin(cfg, "llama-perplexity")
    gguf = Path(cfg["gguf_path"])
    if not gguf.is_file():
        raise FileNotFoundError(f"missing gguf: {gguf}")
    ctx = int(cfg.get("ctx_size", 1024))
    chunks = cfg.get("ppl_chunks")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    ) as handle:
        handle.write("\n\n".join(texts))
        tmp = Path(handle.name)
    cmd = [
        str(llama_ppl),
        "-m",
        str(gguf),
        "-f",
        str(tmp),
        "-c",
        str(ctx),
        "-ngl",
        str(int(cfg.get("n_gpu_layers", 99))),
    ]
    if chunks is not None and str(chunks).lower() != "all":
        cmd.extend(["--chunks", str(int(chunks))])
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False, capture_output=True)
    combined = (proc.stdout or b"").decode("utf-8", errors="replace") + "\n" + (
        proc.stderr or b""
    ).decode("utf-8", errors="replace")
    tmp.unlink(missing_ok=True)
    ppl = parse_ppl(combined)
    if proc.returncode != 0 or ppl is None:
        raise RuntimeError(
            f"llama-perplexity failed (code={proc.returncode}): {combined[-2000:]}"
        )
    return {
        "perplexity": ppl,
        "ppl_ctx_size": ctx,
        "ppl_chunks": chunks,
        "ppl_stderr_tail": combined[-1500:],
    }


def run_bleu_chrf(cfg: dict, texts: list[str]) -> dict:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for GGUF BLEU/chrF") from exc

    llama_cli = resolve_bin(cfg, "llama-completion")
    gguf = Path(cfg["gguf_path"])
    prompt_ratio = float(cfg.get("bleu_prompt_ratio", 0.5))
    max_new = int(cfg.get("bleu_max_new_tokens", 256))
    min_new = int(cfg.get("bleu_min_new_tokens", 16))
    length_ratio = float(cfg.get("bleu_length_ratio", 1.3))
    n_samples = cfg.get("bleu_n_texts", 100)
    seed = int(cfg.get("bleu_seed", 0))

    # Optional cap (bleu_max_text_words): without it, a single very long text
    # in the sample yields a multi-thousand-word prompt/reference that no
    # bounded generation can match, collapsing corpus chrF for every quant.
    max_words = cfg.get("bleu_max_text_words")
    eligible = [t for t in texts if len(str(t).split()) >= 4
                and (not max_words or len(str(t).split()) <= int(max_words))]
    if n_samples is not None and str(n_samples).lower() != "all":
        n = int(n_samples)
        if 0 < n < len(eligible):
            eligible = random.Random(seed).sample(eligible, n)

    hypotheses: list[str] = []
    references: list[str] = []
    for text in eligible:
        words = text.split()
        split_idx = max(1, int(len(words) * prompt_ratio))
        prompt = " ".join(words[:split_idx])
        reference = " ".join(words[split_idx:])
        if not reference:
            continue
        # Approximate budget from whitespace tokens (GGUF has no HF tokenizer here).
        budget = min(max_new, max(min_new, int(len(reference.split()) * length_ratio)))
        cmd = [
            str(llama_cli),
            "-m",
            str(gguf),
            "-p",
            prompt,
            "-n",
            str(budget),
            "--temp",
            "0",
            "-no-cnv",
            "--single-turn",
            "-ngl",
            str(int(cfg.get("n_gpu_layers", 99))),
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            continue
        generated = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        # Strip echoed prompt / chat labels when present.
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].strip()
        for marker in ("assistant\n", "assistant:", "Assistant:"):
            if marker in generated:
                generated = generated.split(marker, 1)[-1].strip()
        hypotheses.append(generated)
        references.append(reference)

    if not hypotheses:
        return {"bleu": 0.0, "chrf": 0.0, "n_pairs": 0}
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": bleu.score, "chrf": chrf.score, "n_pairs": len(hypotheses)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one GGUF with llama.cpp")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    cfg = load_config(cfg_path)
    texts = load_texts(cfg, cfg_path)
    ppl = run_perplexity(cfg, texts)
    bleu = run_bleu_chrf(cfg, texts)
    results = {
        "protocol": "gguf-llama-cpp",
        "run": cfg.get("run") or Path(cfg["gguf_path"]).name,
        "gguf_path": Path(cfg["gguf_path"]).name,
        "quant": cfg.get("quant", "Q4_K_M"),
        "hf_repo": cfg.get("hf_repo"),
        "hf_clm_ref": cfg.get("hf_clm_ref"),
        "perplexity": round(float(ppl["perplexity"]), 4),
        "BLEU": round(float(bleu["bleu"]), 2),
        "chrF": round(float(bleu["chrf"]), 2),
        "n_texts_evaluated": len(texts),
        "n_bleu_pairs": bleu["n_pairs"],
        "ppl_ctx_size": ppl["ppl_ctx_size"],
        "ppl_chunks": ppl["ppl_chunks"],
        "bleu_seed": int(cfg.get("bleu_seed", 0)),
        "bleu_max_text_words": cfg.get("bleu_max_text_words"),
        "note": "Not HF-BPB-equivalent; compare PPL only within same GGUF/tokenizer family.",
    }
    out = Path(cfg["output_json"])
    if not out.is_absolute():
        eval_root = cfg_path.parents[2] if cfg_path.parent.name in {"clm", "gguf"} else EVAL_ROOT
        stripped = out.as_posix()
        while stripped.startswith("../"):
            stripped = stripped[3:]
        out = (eval_root / stripped).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
