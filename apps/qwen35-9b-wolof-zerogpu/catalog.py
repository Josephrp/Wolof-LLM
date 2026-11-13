"""Qwen3.5-9B Wolof family catalog for the ZeroGPU demo."""

from __future__ import annotations

from typing import Any

# Metrics from Experiments/Qwen3.5-Wolof-Family/docs/FAMILY_EVAL.md (full evaluate_clm).
MODELS: dict[str, dict[str, Any]] = {
    "9B · Instruct-v1 (recommended)": {
        "repo_id": "Tonic/Qwen3.5-9B-Wolof-Instruct-v1",
        "size_key": "9b",
        "params": "~9B",
        "mode": "chat",
        "zerogpu_size": "large",
        "base": "Tonic/Qwen3.5-9B-Wolof-CPT-v1",
        "metrics": {"BPB": 0.9981, "BLEU": 5.94, "chrF": 26.93, "PPL": 6.40},
        "sft_eval_loss": 0.7285,
        "summary": (
            "Largest GalsenAI Qwen3.5 Wolof chat checkpoint: vocab-extended CPT "
            "then SFT-from-CPT (anchored `<|lang_wo|>` `<|ortho_ns|>`)."
        ),
        "implication": (
            "**Best interactive demo.** Instruct BPB 1.00 / BLEU 5.94 / chrF 26.9 "
            "on the ortho NS+mixed CLM test; SFT eval_loss 0.729. Prefer this for "
            "Wolof chat; switch to CPT for raw continuation."
        ),
    },
    "9B · CPT-v1 (best BPB)": {
        "repo_id": "Tonic/Qwen3.5-9B-Wolof-CPT-v1",
        "size_key": "9b",
        "params": "~9B",
        "mode": "completion",
        "zerogpu_size": "large",
        "base": "Tonic/Qwen3.5-9B-Wolof-ext-init-v1",
        "metrics": {"BPB": 0.8478, "BLEU": 8.33, "chrF": 29.29, "PPL": 4.84},
        "cpt_eval_loss": 1.7502,
        "summary": (
            "Causal LM after 3-epoch NTP CPT on ortho-filtered Wolof "
            "(non_standard + mixed). Family gold BPB winner at 9B."
        ),
        "implication": (
            "**Best language-model scores.** BPB 0.85 vs stock ext-init 1.84 "
            "(~−1.0); BLEU 8.33 / chrF 29.3. Uses continuation prompting "
            "(no chat template)."
        ),
    },
}

DEFAULT_MODEL = "9B · Instruct-v1 (recommended)"

STOCK_BASELINE = {
    "label": "Tonic/Qwen3.5-9B-Wolof-ext-init-v1 (stock)",
    "BPB": 1.8371,
    "BLEU": 3.55,
    "chrF": 21.55,
    "PPL": 30.47,
}

EXPERIMENT_OVERVIEW = """
### Qwen3.5 Wolof family (GalsenAI)

Ladder **0.8B → 2B → 4B → 9B** with orthography filter (formal/`standard` dropped),
vocab extension + anchors, **CPT 3 epochs** then dual SFT (from-CPT / from-InstructStock)
and DPO. This Space ships the **9B** hero artifacts.

| Checkpoint | BPB ↓ | BLEU | chrF | Role |
|---|---:|---:|---:|---|
| Stock ext-init | 1.84 | 3.55 | 21.6 | Baseline |
| **CPT-v1** | **0.85** | **8.33** | **29.3** | Best LM |
| Instruct-v1 | 1.00 | 5.94 | 26.9 | Best chat |

Protocol: `evaluate_clm.py` on ortho-filtered Wolof CLM test. **BPB** is the
cross-tokenizer gold metric; PPL is same-tokenizer only.
"""


def model_choices() -> list[str]:
    return list(MODELS.keys())


def format_model_card(label: str) -> str:
    entry = MODELS[label]
    m = entry["metrics"]
    stock = STOCK_BASELINE
    lines = [
        f"### `{entry['repo_id']}`",
        "",
        entry["summary"],
        "",
        f"- **Params:** {entry['params']}",
        f"- **Mode:** `{entry['mode']}`",
        f"- **ZeroGPU size:** `{entry['zerogpu_size']}`",
        f"- **Parent:** `{entry['base']}`",
        "",
        "| Metric | This model | Stock ext-init |",
        "|---|---:|---:|",
        f"| BPB ↓ | **{m['BPB']:.4f}** | {stock['BPB']:.4f} |",
        f"| BLEU | **{m['BLEU']:.2f}** | {stock['BLEU']:.2f} |",
        f"| chrF | **{m['chrF']:.2f}** | {stock['chrF']:.2f} |",
        f"| PPL | {m['PPL']:.2f} | {stock['PPL']:.2f} |",
        "",
        entry["implication"],
    ]
    if "sft_eval_loss" in entry:
        lines.append(f"\nSFT-from-CPT best-pool `eval_loss`: **{entry['sft_eval_loss']:.4f}**.")
    if "cpt_eval_loss" in entry:
        lines.append(f"\nCPT best-pool `eval_loss`: **{entry['cpt_eval_loss']:.4f}**.")
    return "\n".join(lines)
