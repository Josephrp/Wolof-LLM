# Publication plan — Wolof LFM2.5

## Goals

1. Ship a strong **Wolof instruct** model on LFM2.5-1.2B (hero).
2. Publish **family analyses** 230M → 8B-A1B vs SmolLM3/Qwen/Aya on shared Evaluation.
3. Document dataset limits (translation skew, Afri* leakage) and synthetic fills honestly.

## Artifacts

| Artifact | Where |
|---|---|
| CPT + Instruct weights | `galsenai/LFM2.5-{230M,350M,1.2B,2.6B,8B-A1B}-*_wolof` |
| Repro configs/scripts | this experiment folder |
| CLM metrics JSON | `Evaluation/metrics/clm/lfm_*.json` |
| Data audit | `results/data_audit.json` |
| Collection | HF Collection “Wolof LFM2.5” |
| Narrative | Blog / HF article / short tech report |

## Model card outline

1. Model summary (Base → CPT → SFT → optional DPO).
2. Data: `cleaned_data_for_clm` + reweighted `wolof_sft` + synth packs used.
3. Languages & orthography notes.
4. Tokenizer extension (family 65k vs 128k).
5. Eval tables + leakage caveats.
6. Intended use / out-of-scope.
7. Citations: `arXiv:2511.23404` + GalsenAI.

## Story angles

1. Edge Wolof (230M/350M/1.2B) + Liquid on-device story.
2. Dataset engineering: why cleaned CLM ≠ raw centralized (code token mass).
3. SFT mixture science (translation cap) + synthetic reasoning/long-context.
4. Tokenizer surgery: 65k append vs 128k end-specials.
5. Size Pareto + MoE active-param efficiency (8B-A1B).
6. BPB vs PPL; CPT→SFT vs Instruct-only.

## Checklist

- [ ] Hero 1.2B gates green (SCALING_LADDER through stage 11)
- [ ] Family rows filled for sizes trained
- [ ] Evaluation JSONs committed
- [ ] Native-speaker spot-check
- [ ] License compatibility with Liquid LFM license reviewed
