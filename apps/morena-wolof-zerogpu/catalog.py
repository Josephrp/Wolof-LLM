"""MORENA Wolof family catalog: Hub repos, eval metrics, and honest training implications."""

from __future__ import annotations

from typing import Any

# Gold protocol: morena-native BPB (lower is better). Stock baselines for context.
STOCK_BASELINES: dict[str, dict[str, float]] = {
    "1.5b": {"BPB": 2.8985, "BLEU": 0.89, "chrF": 11.36, "label": "vamboai/morena-1.5b-base"},
    "0.5b": {"BPB": 3.0270, "BLEU": 0.96, "chrF": 11.74, "label": "vamboai/morena-0.5b-mini"},
    "0.2b": {"BPB": 3.1097, "BLEU": 0.64, "chrF": 11.41, "label": "vamboai/morena-0.2b-nano"},
}

# Dropdown label -> catalog entry
MODELS: dict[str, dict[str, Any]] = {
    "0.2B Nano · Distill (recommended)": {
        "repo_id": "Tonic/MORENA-0.2B-Nano-Wolof-Distill",
        "size_key": "0.2b",
        "stage": "Distill",
        "params": "~0.2B",
        "prompt_mode": "completion",
        "chat_template": "none",
        "metrics": {"BPB": 3.0845, "BLEU": 0.03, "chrF": 7.0, "PPL": 280.99},
        "summary": (
            "Student distilled from the 1.5B Wolof teacher into the nano shape. "
            "Keeps stock-level BPB while remaining small enough for snappy ZeroGPU demos."
        ),
        "implication": (
            "**Training outcome: relative success on the continuum gate.** "
            "BPB ≈ 3.08 vs stock nano 3.11 — distillation preserved language-model quality "
            "without the collapse seen in the tiny-SFT Instruct runs. BLEU/chrF are weak "
            "because this stage was optimized for next-token CE+KL on packed CLM, not chat "
            "continuation. Prefer this (or 0.5B Distill) for demos of fluent Wolof CLM."
        ),
    },
    "0.5B Mini · Distill (recommended)": {
        "repo_id": "Tonic/MORENA-0.5B-Mini-Wolof-Distill",
        "size_key": "0.5b",
        "stage": "Distill",
        "params": "~0.5B",
        "prompt_mode": "completion",
        "chat_template": "none",
        "metrics": {"BPB": 3.0955, "BLEU": 0.17, "chrF": 5.14, "PPL": 286.69},
        "summary": (
            "Mini student distilled from 1.5B Wolof CPT teacher. Best balance of size and "
            "stock-competitive BPB in the published Wolof continuum."
        ),
        "implication": (
            "**Training outcome: successful transfer.** "
            "BPB 3.10 is within ~2% of stock mini (3.03). Distillation worked as intended: "
            "shape-matched student + KL/CE on Wolof shards retained causal-LM competence. "
            "Not an instruction model — prime with plain Wolof document prefixes only."
        ),
    },
    "0.5B Mini · Wolof CPT": {
        "repo_id": "Tonic/MORENA-0.5B-Mini-Wolof-CPT",
        "size_key": "0.5b",
        "stage": "CPT",
        "params": "~0.5B",
        "prompt_mode": "completion",
        "chat_template": "none",
        "metrics": {"BPB": 3.9783, "BLEU": 2.67, "chrF": 14.53, "PPL": 1439.82},
        "summary": (
            "Continued pretraining on packed Wolof CLM with Muon/FSDP2 native trainer "
            "(full budget after smoke). Stronger BLEU/chrF than distill, weaker BPB than stock."
        ),
        "implication": (
            "**Training outcome: mixed / data-limited CPT.** "
            "BPB rose from stock 3.03 → 3.98 (worse LM compression) while BLEU/chrF improved. "
            "Typical when CPT overfits a small Wolof pack: surface form / continuation metrics "
            "can look better even as bits-per-byte degrades. Useful to inspect domain style; "
            "not a clean win on the gold BPB protocol."
        ),
    },
    "1.5B · Wolof CPT": {
        "repo_id": "Tonic/MORENA-1.5B-Wolof-CPT",
        "size_key": "1.5b",
        "stage": "CPT",
        "params": "~1.5B",
        "prompt_mode": "completion",
        "chat_template": "none",
        "metrics": {"BPB": 4.3261, "BLEU": 3.14, "chrF": 16.74, "PPL": 2718.95},
        "summary": (
            "Largest CPT run in the family (2000 steps). Same native Muon trainer and frozen 65k "
            "tokenizer as upstream MORENA."
        ),
        "implication": (
            "**Training outcome: same pattern as mini CPT — not a BPB win.** "
            "Stock base BPB 2.90 → Wolof CPT 4.33. BLEU/chrF are the highest among CPT runs, "
            "suggesting the model adapted to the small Wolof corpus surface, but gold BPB "
            "says general compression quality slipped. Scale alone did not fix data scarcity."
        ),
    },
    "0.2B Nano · Wolof CPT": {
        "repo_id": "Tonic/MORENA-0.2B-Nano-Wolof-CPT",
        "size_key": "0.2b",
        "stage": "CPT",
        "params": "~0.2B",
        "prompt_mode": "completion",
        "chat_template": "none",
        "metrics": {"BPB": 4.5616, "BLEU": 3.26, "chrF": 17.97, "PPL": 4181.91},
        "summary": "Nano CPT on Wolof CLM (1000 steps). Fastest CPT checkpoint in the family.",
        "implication": (
            "**Training outcome: overfit CPT on tiny capacity.** "
            "Worst BPB among CPT sizes (4.56 vs stock 3.11) with inflated BLEU/chrF — classic "
            "small-model / small-data memorization of the pack. Prefer Distill nano for demos."
        ),
    },
    "1.5B · Wolof Instruct (SFT)": {
        "repo_id": "Tonic/MORENA-1.5B-Wolof-Instruct",
        "size_key": "1.5b",
        "stage": "Instruct",
        "params": "~1.5B",
        "prompt_mode": "chat",
        "chat_template": "reserved",
        "metrics": {"BPB": 11.4207, "BLEU": 0.03, "chrF": 9.65, "PPL": 1.16586550211e9},
        "summary": (
            "Loss-masked SFT with MORENA chat markers `<reserved_0>` / `<reserved_1>` on a "
            "repeated toy Wolof SFT mixture (smoke-scale data)."
        ),
        "implication": (
            "**Training outcome: failure / collapse — included for transparency.** "
            "Train loss → 0 on a tiny repeated JSONL, then BPB exploded (11.4 vs stock ~2.9). "
            "This is catastrophic forgetting of CLM, not a usable instruct model. "
            "Demonstrates why SFT needs a real multi-domain Wolof instruction mix before "
            "publishing Instruct weights."
        ),
    },
    "0.5B Mini · Wolof Instruct (SFT)": {
        "repo_id": "Tonic/MORENA-0.5B-Mini-Wolof-Instruct",
        "size_key": "0.5b",
        "stage": "Instruct",
        "params": "~0.5B",
        "prompt_mode": "chat",
        "chat_template": "reserved",
        "metrics": {"BPB": 15.7872, "BLEU": 0.01, "chrF": 2.26, "PPL": 3.41281796117558e12},
        "summary": "Same fragile SFT recipe as 1.5B Instruct, applied after distill→SFT init.",
        "implication": (
            "**Training outcome: severe collapse.** "
            "BPB 15.8 and near-zero BLEU — the model memorized the toy SFT windows and lost "
            "open-domain Wolof CLM. Do not use for product demos; kept so the family eval "
            "story is complete and reproducible on Hub."
        ),
    },
    "Stock 1.5B Instruct (public baseline)": {
        "repo_id": "vamboai/morena-1.5b-instruct",
        "size_key": "1.5b",
        "stage": "Stock-Instruct",
        "params": "~1.5B",
        "prompt_mode": "chat",
        "chat_template": "reserved",
        "metrics": {"BPB": 2.945, "BLEU": 1.15, "chrF": 12.32, "PPL": 217.77},
        "summary": "Upstream Vambo MORENA 1.5B instruct — public baseline for chat markers.",
        "implication": (
            "**Baseline, not a Wolof continuum run.** "
            "Stock instruct BPB ~2.95 is the quality bar. Our Wolof SFT runs are far worse; "
            "Distill/CPT should be compared against stock base (~2.90) instead."
        ),
        "public": True,
    },
}

DEFAULT_MODEL = "0.5B Mini · Distill (recommended)"

EXPERIMENT_OVERVIEW = """
## GalsenAI · MORENA Wolof continuum

**MORENA** (Vambo) is a plain PyTorch Llama-style decoder — **not** a Transformers model.
This Space loads private `Tonic/MORENA-*-Wolof-*` checkpoints (plus an optional public stock
baseline) with the frozen **65k** tokenizer and chat markers `<reserved_0>` / `<reserved_1>`.

### Pipeline stages
1. **Stock eval** — BPB gold on verbalized Wolof CLM (`morena-native`).
2. **CPT** — Muon + AdamW, document-packed Wolof shards.
3. **Prune → Distill** — shape-matched students with CE + KL from the 1.5B teacher.
4. **SFT** — loss-masked instruct on MORENA markers (toy mix in this run → collapse).
5. **Hub publish** — `model.safetensors`, `tokenizer.json`, `metrics.json`, model-index.

### How to prompt
- **CPT / Distill** (*base*): plain Wolof document completion — no chat markers. Optional system is plain leading context.
- **Instruct / Stock-Instruct**: wrap with `<reserved_0>` (user) / `<reserved_1>` (assistant); system folds into the user turn.
- Sampler defaults: temperature **0.7**, top-p **0.9**, docs primed with `<eos>`. Demo also exposes top-k / mild repetition penalty (set top-k=0, rep=1.0 for the exact upstream loader).

### How to read scores
- **BPB (lower better)** is the gold metric (cross-tokenizer bits per byte).
- **BLEU / chrF** are secondary continuation probes — they can rise even when BPB worsens.
- **Success here** means Distill ≈ stock BPB; CPT is mixed; SFT Instruct is a documented failure
  until a real Wolof instruction corpus is used.

Set Space secret `HF_TOKEN` (read access) for private `Tonic/*` repos.
"""


def format_model_card(label: str) -> str:
    entry = MODELS[label]
    m = entry["metrics"]
    stock = STOCK_BASELINES.get(entry["size_key"], {})
    stock_bpb = stock.get("BPB")
    delta = None
    if stock_bpb is not None and m.get("BPB") is not None:
        delta = float(m["BPB"]) - float(stock_bpb)
        delta_s = f"{delta:+.3f} vs stock {stock_bpb:.3f}"
    else:
        delta_s = "n/a"

    return f"""
### {label}

| Field | Value |
|---|---|
| Hub | `{entry['repo_id']}` |
| Stage | **{entry['stage']}** · {entry['params']} · prompt=`{entry.get('prompt_mode', '?')}` |
| BPB (↓ better) | **{m.get('BPB')}** ({delta_s}) |
| BLEU | {m.get('BLEU')} |
| chrF | {m.get('chrF')} |
| PPL | {m.get('PPL')} |

{entry['summary']}

{entry['implication']}
"""


def model_choices() -> list[str]:
    return list(MODELS.keys())
