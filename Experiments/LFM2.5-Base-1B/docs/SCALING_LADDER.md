# Scaling ladder — small tests → full training

Principle: **every increase in compute must be justified by an eval gate**. Fail → fix data/script/HP; do not “hope scale fixes it.”

Mirror SmolLM3 lessons: smoke (10 steps), vocab ablation (~250 steps), equal-data v1, early-stop v2.

---

## Stage overview

| Stage | Name | Compute | Models | Gate (must pass) | Next |
|---:|---|---|---|---|---|
| 0 | Env + load | minutes | 350M, 1.2B | `from_pretrained` + 1 forward | 1 |
| 1 | Data audit | minutes | — | Histograms written; CLM test = 4282 | 2 |
| 2 | Tokenizer | <1 h | 65k family | Fertility ↓; mono/roundtrip/non-regress OK | 3 |
| 3 | Smoke CPT | minutes–1 h | 350M then 1.2B | 10 steps no NaN; journal written; BPB finite | 4 |
| 4 | Mini CPT | few h | **350M** ext-LoRA | BPB ≪ base; loss trending down | 5 |
| 5 | Vocab ablation | few h | 350M | Pick N∈{2k,4k,8k} by post-mini BPB | 6 |
| 6 | v1 equal-data | ~1 day | 350M full grid | Ranking interpretable; ext-LoRA competitive | 7 |
| 7 | v1/v2 on **1.2B** | multi-day | 1.2B | Beat 350M BPB or match with better instruct later | 8 |
| 8 | Merge gate | hours | 1.2B | ΔBPB ≤ 0.002 | 9 |
| 9 | Smoke SFT | hours | 1.2B | Loss↓; stays in WO on 20 prompts | 10 |
| 10 | Mini SFT | <1 day | 1.2B 50k–100k mix | Val by-task improves vs CPT-only chat | 11 |
| 11 | Full SFT | multi-day | 1.2B | Publishable instruct + Evaluation JSON | 12 |
| 12 | Family roll | variable | 230M, 350M publish | Edge models pass smoke+mini+short v2 | 13 |
| 13 | 128k family | multi-day | 2.6B | Short CPT gate then longctx mini | 14 |
| 14 | MoE | multi-day | 8B-A1B | Stable train + tool eval | publish |
| S* | Synthetic scale | interleaved | — | G0–G4 in SYNTHETIC_DATA.md | merge into 10–14 |

---

## Stage details

### Stage 0 — Environment

```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('LiquidAI/LFM2.5-350M-Base', dtype='bfloat16')"
python scripts/clm/06_probe_modules.py --base-id LiquidAI/LFM2.5-1.2B-Base
```

Gate: no import errors; module list saved.

### Stage 1 — Data audit

```bash
python scripts/data/20_audit_datasets.py \
  --clm galsenai/cleaned_data_for_clm \
  --sft galsenai/wolof_sft \
  --raw galsenai/wolof_centalized_corpus \
  --out results/data_audit.json
```

Gate: confirms train/test sizes; flags translation %; length p50.

### Stage 2 — Tokenizer

Build on 1.2B (or 350M) Base; validate; measure fertility on CLM test.

Gate: fertility drop ≥20% target aspirational; **0** monotonicity regressions; FR/EN/code identical encodings.

### Stage 3 — Smoke CPT (`configs/scaling/smoke_*.yaml`)

- `max_steps: 10`, subset ~20k texts (SmolLM3 `--smoke` pattern).
- Runs: default-LoRA + ext-LoRA on 350M.

Gate: finishes; `journal.json`; BPB before/after finite; CUDA OOM fixed.

### Stage 4 — Mini CPT (~250–500 steps)

Like SmolLM3 vocab ablation budget.

Gate: ext-LoRA BPB improvement vs step-0; if default beats ext badly → check `modules_to_save` / init norms.

### Stage 5 — Vocab size pick

Mini runs N=2000/4000/8000 on 350M; choose N for family (expect ~4000).

### Stage 6–7 — Equal data → early stop

350M v1 grid first; copy best HP to 1.2B v1 then v2 (`patience`).

Gate: 1.2B BPB on clean test clearly below stock Base (order-of-magnitude improvement expected if SmolLM3-like).

### Stage 8 — Merge

`05_merge_adapter.py` BPB tolerance 0.002.

### Stage 9–11 — SFT ladder

1. Smoke 10–50 steps on 1k mixture rows.
2. Mini on 50k–100k stratified mixture.
3. Full mixture (± synth after S* gates).

Eval each time: val loss **by task**, 20 qualitative WO prompts, CLM BPB (must not explode).

### Stage 12 — Edge publish

230M/350M: short v2 + SFT using 1.2B hyperparameters scaled down.

### Stage 13 — 2.6B

1. New 128k tokenizer + smoke.
2. Short CPT gate.
3. Longctx mini (synth packs) → needle eval.
4. SFT.

### Stage 14 — 8B-A1B

MoE smoke on tiny batch → mini CPT → SFT tool-heavy.

---

## Eval instruments per stage

| Instrument | Stages |
|---|---|
| Fertility | 2, 5 |
| BPB (internal metrics) | 3–8, 11–14 |
| BPB+BLEU+chrF (`evaluate_clm.py`) | after merge & instruct publish |
| Task-wise SFT val | 9–11 |
| Qualitative WO panel (human) | 11, publish |
| Needle longctx | 13 |
| Tool schema validity | 11, 14 |
| Afri* SFT test | 11+ (with leakage caveat) |

---

## Promotion rules (summary)

```text
IF smoke fails → fix code, do not mini
IF mini BPB not << base → fix tok/LoRA/data, do not v1
IF v1 ranking nonsense → fix eval split / LR, do not v2
IF merge ΔBPB > tol → fix tying, do not SFT
IF SFT leaves WO → fix mixture/synth, do not publish
IF 2.6B short CPT fails → do not longctx
IF MoE unstable → reduce LR / freeze more experts
```

## Config stubs to create

- [ ] `configs/scaling/smoke_350m_default_lora.yaml`
- [ ] `configs/scaling/smoke_350m_ext_lora.yaml`
- [ ] `configs/scaling/mini_350m_ext_lora.yaml`
- [ ] `configs/scaling/smoke_1.2b_ext_lora.yaml`
- [ ] `configs/scaling/mini_1.2b_ext_lora.yaml`
- [ ] `configs/scaling/smoke_2.6b_ext_lora.yaml`
- [ ] `configs/scaling/smoke_8b_moe_lora.yaml`
