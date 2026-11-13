"""LFM2.5 Wolof Instruct-v3 family catalog (Hub repos + family-eval metrics)."""

from __future__ import annotations

from typing import Any

# Stock LiquidAI LFM2.5 Base BPB (same HF evaluate_clm protocol) for deltas.
STOCK_BASELINES: dict[str, dict[str, float]] = {
    "230m": {"BPB": 3.2886, "BLEU": 0.44, "chrF": 9.57, "label": "LiquidAI/LFM2.5-230M-Base"},
    "350m": {"BPB": 3.1890, "BLEU": 2.87, "chrF": 15.40, "label": "LiquidAI/LFM2.5-350M-Base"},
    "1.2b": {"BPB": 3.0714, "BLEU": 2.59, "chrF": 15.41, "label": "LiquidAI/LFM2.5-1.2B-Base"},
    "2.6b": {"BPB": 2.6340, "BLEU": 4.23, "chrF": 16.66, "label": "LiquidAI/LFM2.5-2.6B-Base"},
    "8b": {"BPB": 2.9328, "BLEU": 3.37, "chrF": 17.19, "label": "LiquidAI/LFM2.5-8B-A1B"},
}

# Dropdown label -> catalog entry. Metrics from FAMILY_EVAL.md clm-instruct-v3:*
MODELS: dict[str, dict[str, Any]] = {
    "1.2B · Instruct-v3 (recommended)": {
        "repo_id": "Tonic/LFM2.5-1.2B-Wolof-Instruct-v3",
        "size_key": "1.2b",
        "params": "~1.2B",
        "arch": "lfm2",
        "tokenizer_family": "65k-ext",
        "base_cpt": "Tonic/LFM2.5-1.2B-Wolof-CPT-v3",
        "zerogpu_size": "large",
        "metrics": {"BPB": 1.4263, "BLEU": 3.4, "chrF": 16.52, "PPL": 11.41},
        "summary": (
            "Hero size: CPT-v3 (ortho non-standard+mixed pack) → SFT Instruct-v3. "
            "Best balance of quality, latency, and ZeroGPU fit."
        ),
        "implication": (
            "**Strong Wolof instruct checkpoint.** BPB 1.43 vs stock base 3.07 "
            "(~−1.65). Prefer this for demos unless you need the 2.6B BPB win or "
            "tiny on-device sizes."
        ),
    },
    "2.6B · Instruct-v3 (best BPB)": {
        "repo_id": "Tonic/LFM2.5-2.6B-Wolof-Instruct-v3",
        "size_key": "2.6b",
        "params": "~2.6B",
        "arch": "lfm2",
        "tokenizer_family": "128k-ext",
        "base_cpt": "Tonic/LFM2.5-2.6B-Wolof-CPT-v3",
        "zerogpu_size": "large",
        "metrics": {"BPB": 1.1581, "BLEU": 3.44, "chrF": 19.69, "PPL": 29.14},
        "summary": (
            "Best family Instruct-v3 BPB on the ortho NS+mixed verbalized test. "
            "Uses the 128k-ext tokenizer family."
        ),
        "implication": (
            "**Quality leader on gold BPB.** 1.16 vs stock 2.63. Heavier than 1.2B; "
            "use when fluency matters more than turn latency. Do not compare PPL "
            "to 65k-ext models — BPB is the cross-tokenizer gold."
        ),
    },
    "350M · Instruct-v3": {
        "repo_id": "Tonic/LFM2.5-350M-Wolof-Instruct-v3",
        "size_key": "350m",
        "params": "~350M",
        "arch": "lfm2",
        "tokenizer_family": "65k-ext",
        "base_cpt": "Tonic/LFM2.5-350M-Wolof-CPT-v3",
        "zerogpu_size": "large",
        "metrics": {"BPB": 1.5566, "BLEU": 1.0, "chrF": 11.85, "PPL": 14.25},
        "summary": "Compact Instruct-v3 after CPT-v3; snappy ZeroGPU turns.",
        "implication": (
            "Solid small model: BPB 1.56 vs stock 3.19. Good for latency A/B "
            "against 1.2B; expect shorter, less reliable answers."
        ),
    },
    "230M · Instruct-v3": {
        "repo_id": "Tonic/LFM2.5-230M-Wolof-Instruct-v3",
        "size_key": "230m",
        "params": "~230M",
        "arch": "lfm2",
        "tokenizer_family": "65k-ext",
        "base_cpt": "Tonic/LFM2.5-230M-Wolof-CPT-v3",
        "zerogpu_size": "large",
        "metrics": {"BPB": 1.8738, "BLEU": 1.14, "chrF": 8.86, "PPL": 26.37},
        "summary": "Smallest Instruct-v3 in the family — edge / smoke demos.",
        "implication": (
            "Still far below stock base BPB (1.87 vs 3.29) but weakest instruct "
            "fluency in the v3 set. Prefer 350M+ for readable chat."
        ),
    },
    "8B-A1B · Instruct-v3 (MoE)": {
        "repo_id": "Tonic/LFM2.5-8B-A1B-Wolof-Instruct-v3",
        "size_key": "8b",
        "params": "~8B-A1B MoE",
        "arch": "lfm2_moe",
        "tokenizer_family": "128k-ext",
        "base_cpt": "Tonic/LFM2.5-8B-A1B-Wolof-CPT-v3",
        "zerogpu_size": "xlarge",
        "metrics": {"BPB": 1.4549, "BLEU": 2.92, "chrF": 16.25, "PPL": 69.16},
        "summary": (
            "MoE instruct (active ~1B). Needs ZeroGPU **xlarge**. Chat template "
            "includes richer tool-call formatting than dense LFM2 sizes."
        ),
        "implication": (
            "BPB 1.45 vs stock 2.93 — strong, but not ahead of dense 2.6B on this "
            "protocol. Use for MoE / long-context experiments; expect colder starts "
            "and higher quota cost."
        ),
    },
}

DEFAULT_MODEL = "1.2B · Instruct-v3 (recommended)"

EXPERIMENT_OVERVIEW = """
## GalsenAI · LFM2.5 Wolof Instruct-v3

Five **LiquidAI LFM2.5** checkpoints adapted to Wolof by GalsenAI and published
under `Tonic/LFM2.5-*-Wolof-Instruct-v3`:

1. **Stock Base** → tokenizer extension (65k-ext or 128k-ext)
2. **CPT-v3** — continual pretraining on the ortho **non-standard + mixed** Wolof pack
3. **Instruct-v3** — SFT on the reweighted Wolof instruction mixture
4. **Eval** — HF `evaluate_clm.py` **BPB** (bits/byte, lower better) is gold

### How to prompt
All five are **chat** models. This Space uses each repo’s `chat_template.jinja`
via `tokenizer.apply_chat_template` (`<|im_start|>` / `<|im_end|>`).

### How to read scores
- **BPB** is comparable across tokenizer families; **PPL is not**.
- Deltas vs stock Base are shown on each card (from `FAMILY_EVAL.md`).
- Native-speaker review still required before product use.

Set Space secret `HF_TOKEN` with **read** access to private `Tonic/LFM2.5-*-Wolof-Instruct-v3` repos.
"""


def format_model_card(label: str) -> str:
    entry = MODELS[label]
    m = entry["metrics"]
    stock = STOCK_BASELINES.get(entry["size_key"], {})
    stock_bpb = stock.get("BPB")
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
| Base CPT | `{entry['base_cpt']}` |
| Size | **{entry['params']}** · arch=`{entry['arch']}` · tok=`{entry['tokenizer_family']}` |
| ZeroGPU | `{entry['zerogpu_size']}` |
| BPB (↓ better) | **{m.get('BPB')}** ({delta_s}) |
| BLEU | {m.get('BLEU')} |
| chrF | {m.get('chrF')} |
| PPL | {m.get('PPL')} *(same-tokenizer only)* |

{entry['summary']}

{entry['implication']}
"""


def model_choices() -> list[str]:
    return list(MODELS.keys())
