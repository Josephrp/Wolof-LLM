# Tiny Aya Wolof Family — Implementation Plan (operational)

See also: `.cursor/plans/tiny_aya_wolof_family_98b6ec94.plan.md` (do not edit from this tree).

## Locked decisions

| Topic | Choice |
|-------|--------|
| Root | `Experiments/TinyAya-Wolof-Family/` (distinct) |
| Size | single `3b35` (3.35B) |
| Base / Instruct | `CohereLabs/tiny-aya-base` · earth (primary) · global |
| Ortho | KEEP `non_standard`+`mixed`; DROP `standard`+`not_wolof`; NTP FR/EN pass-through |
| Anchors | Cohere developer preamble + `<\|lang_wo\|>` `<\|ortho_ns\|>`; **no** `<think>` |
| SFT mask | `<\|START_RESPONSE\|>` … `<\|END_RESPONSE\|>` |
| Vocab | Append-only; assert `bos=2` `eos=3` `pad=0`; tied embeds; LoRA `modules_to_save: [embed_tokens]` |
| Epochs | CPT=3, SFT=3 |
| SFT starts | `merged_cpt` \| `stock_earth` \| `stock_global` |
| License | cc-by-nc-4.0 + Cohere Labs AUP (gated) |
| Remote | `/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/TinyAya-Wolof-Family` |

## Runner stages

`datasets → fertility → tokenizer → resize_instructs → cpt → merge_cpt → prepare → sft_earth → sft_global → sft_cpt → pick_best → dpo → eval → publish`

State: `results/full_pipeline/tinyaya_state.json` · OK: `TINYAYA_FAMILY_OK`

## Helper

`scripts/remote/ssh_exec.py`
