# Qwen3.5 Wolof Family

Distinct experiment tree for adapting [Qwen3.5 Base](https://huggingface.co/collections/Qwen/qwen35) (≤9B) to Wolof.

## Pipeline

1. Materialize NTP `70_20_10` + stratified SFT
2. Ortho-label; **drop formal/`standard`**; keep `non_standard`+`mixed` (FR/EN kept in NTP)
3. Inject Qwen chat-template anchors (`<|lang_wo|>`, `<|ortho_ns|>`, system turn); prepare prompts
4. Publish datasets to `Tonic/`
5. Vocab extension + CPT **3 epochs**
6. Dual SFT **3 epochs**: from CPT-base and from stock Instruct
7. Optional DPO

## Quick start (GPU volume)

```bash
cd /mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family
# Torch must be cu128 (host driver 12.8) — pinned in pyproject (not default cu130).
uv sync --group train --group data
# Kernels: FA3 via `kernels` Hub path, Liger qwen3_5 FLCE, FLA + causal-conv1d for GDN.
# Override: QWEN_ATTN_IMPL=flash_attention_3,qwen… QWEN_DISABLE_LIGER=1
uv run python scripts/publication/44_setup_trackio_space.py --restart
# Gradio dashboard: https://huggingface.co/spaces/Tonic/qwen35-wolof-trackio
uv run python scripts/publication/50_run_family.py --smoke --only-size 08b
```

Production (after smoke):

```bash
uv run python scripts/publication/50_run_family.py --resume --only-size 08b
```

## Non-goals

MoE (no size under 26B), vision SFT, editing LFM / Base-2B trees.
