# Orange Wolof-Qwen3 v2610 — Evaluation, Quantization & v2611 Improvement

Independent evaluation/quantization of Orange's Wolof-Qwen3 family
([arXiv:2610.20824](https://arxiv.org/abs/2610.20824)) and the plan to improve it
(**v2611**) with this repository's Wolof pipeline. Orange repos are read-only for
us; derived artifacts publish under `Tonic/` with the inherited
**CC-BY-NC-SA-4.0** license.

## 🔴 Gate 0 finding (2026-11-17, live Hub audit)

`Orange/Wolof-Qwen3-1.7B-v2610` contains **no weights** (11 MB: tokenizer/config
only) — it cannot be evaluated or quantized until Orange completes the upload.
All other sizes verified with SHAs pinned in `configs/family.yaml`.
8B/14B are FP16 (merged), 4B/0.6B BF16; all stock Qwen3 tokenizer (151,936 vocab),
config context 40,960 vs 2,048 training sequence length.

## Documents

- [`docs/EVALUATION_AND_QUANTIZATION_PLAN.md`](docs/EVALUATION_AND_QUANTIZATION_PLAN.md) —
  Tier-1 CLM (repo `evaluate_clm.py`, BPB gold) → Tier-2 public chat bench
  (translation / IF / function-calling false-positives / long-context / safety)
  → Tier-3 human arena; GGUF + ONNX quantization matrices with acceptance
  thresholds; HF Jobs flavors and ≈ $25–60 wave-1 budget.
- [`docs/TRAINING_IMPROVEMENT_PLAN.md`](docs/TRAINING_IMPROVEMENT_PLAN.md) —
  diagnosis D1–D11 (function-calling collapse, IF regression, 0%-Wolof DPO,
  no CPT, stock tokenizer, 7K conversations, fixed ⅓-merge, single seed) and the
  staged fix: fertility/tokenizer → CPT → rebalanced dual SFT → Wolof-mix DPO →
  merge sweep + golden-BPB select → long-context, per-size recipes.

## Layout (mirrors `Experiments/Qwen3.5-Wolof-Family/`)

```
configs/family.yaml                  # audited Hub SHAs, per-size recipe, quant matrix
configs/eval/clm/*.yaml              # 10 Tier-1 configs (5 Orange + 5 stock Qwen3)
configs/eval/gguf/*.yaml             # Q4_K_M GGUF eval configs (evaluate_gguf.py)
configs/publication/{gguf,onnx}_family.yaml
configs/training/v2611_recipes.yaml  # Stage A–F hyperparameters + gates
scripts/jobs/job_eval_clm_family.py  # HF Jobs GPU: hf jobs uv run --flavor h200 --secrets HF_TOKEN --env ONLY=14b …
scripts/jobs/job_gguf_family.py      # HF Jobs cpu-xl: GGUF -> Tonic/*-GGUF
scripts/jobs/job_onnx_family.py      # HF Jobs cpu-xl: ONNX/WebGPU (0.6B, 4B)
scripts/publication/run_eval_family.py   # local/H200 equivalent (--resume)
scripts/publication/aggregate_eval.py    # -> results/FAMILY_EVAL.md (+ --check-shas)
results/                             # FAMILY_EVAL.md, clm_eval_state.json (generated)
```

## Quick start

```bash
# Local smoke (GPU host): one size first
python Experiments/Orange-Wolof-Qwen3-v2610/scripts/publication/run_eval_family.py --only 06b
# Full Tier-1 matrix via HF Jobs (detach, then `hf jobs logs <id> --follow`)
hf jobs uv run --flavor h200 --timeout 4h --secrets HF_TOKEN --env ONLY=14b \
  Experiments/Orange-Wolof-Qwen3-v2610/scripts/jobs/job_eval_clm_family.py
# GGUF for one size
hf jobs uv run --flavor cpu-xl --timeout 6h --secrets HF_TOKEN --env ONLY=4b \
  Experiments/Orange-Wolof-Qwen3-v2610/scripts/jobs/job_gguf_family.py
# Aggregate whatever metrics exist into the family table
python Experiments/Orange-Wolof-Qwen3-v2610/scripts/publication/aggregate_eval.py --check-shas
```

Nothing here has been launched yet: Gate 1 (smoke) is the first paid step.
