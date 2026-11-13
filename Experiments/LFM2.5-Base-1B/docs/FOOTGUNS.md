# Footguns — LFM2.5 Wolof

Operator checklist. Full context: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §4.

| If you see… | You likely forgot… |
|---|---|
| BPB ≫ base right after embedding init | Norm recalibration of mean-pooled rows — **never** rescale |
| New Wolof tokens never improve | `modules_to_save: [embed_tokens, lm_head]` on LoRA |
| Chat tokens decode as garbage after “tokenizer surgery” | Applied SmolLM3 special **reindex** on LFM — use **append-only** |
| Sequences glue wrongly in packing | Using EOS id 2 (`<|endoftext|>`) instead of **7** (`<|im_end|>`) |
| Merged model worse than adapter | `tie_embedding` / re-tie on load — run `05_merge` BPB gate |
| Ext tokenizer “loses” on PPL | Comparing PPL across tokenizers — use **BPB** |
| Amazing eval BPB, bad real use | Train/eval leakage — use `wolof_eval_clean_v1` |
| Model answers only in French | SFT mix / system prompt — up-sample Wolof-only turns |
| `Lfm2ForCausalLM` import error | `transformers` too old — need LFM2 support (≥4.57 / 5.x per card) |
| Cross-size table looks random | Mixing **65k** (≤1.2B) and **128k** (≥2.6B) tokenizer families |
| CPT on centralized corpus “as is” | ~70% tokens are code — use **`cleaned_data_for_clm`** |
| SFT without task weights | 57.9% translation → translator model |
| Trust AfriMMLU/MGSM SFT test blindly | May overlap pretrain text — disclose + add native WO test |
| Jump smoke → multi-day 2.6B | Violates SCALING_LADDER gates |
| Long-context claims on short CPT only | Need synth packs + needle eval |
| 8B MoE under-adapts | Missing expert LoRA targets |

Before every long run: `--smoke` (10 steps) on the exact config.  
Promote compute only when the previous stage gate is green (`docs/SCALING_LADDER.md`).
