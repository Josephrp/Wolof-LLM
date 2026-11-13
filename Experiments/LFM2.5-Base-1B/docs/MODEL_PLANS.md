# Per-model training plans (LFM2.5 text Bases)

All sizes: start from **Base**, CPT on `galsenai/cleaned_data_for_clm`, SFT from stratified `galsenai/wolof_sft` (+ synthetic packs as gated). Shared scripts; size-specific YAML under `configs/clm/<size>/` and `configs/sft/<size>/`.

Family reference (user research + Hub cards):

| Size | Base | Instruct-class Hub | Params | Vocab | Ctx | Pretrain tokens |
|---|---|---|---:|---:|---:|---|
| 230M | `LFM2.5-230M-Base` | `LFM2.5-230M` | 230M | 65 536 | 32k | 28T |
| 350M | `LFM2.5-350M-Base` | `LFM2.5-350M` | 354M | 65 536 | 32k | 28T |
| 1.2B | `LFM2.5-1.2B-Base` | `…-Instruct` / JP / Thinking | 1.17B | 65 536 | 32k | 28T |
| 2.6B | `LFM2.5-2.6B-Base` | `LFM2.5-2.6B` | 2.69B | **128 000** | **128k** | 34T |
| 8B-A1B | `LFM2.5-8B-A1B-Base` | `LFM2.5-8B-A1B` | 8.3B / **1.5B act.** | **128 000** | **128k** | 38T |

Tokenizer families: **65k** (≤1.2B) vs **128k** (≥2.6B). Build extension once per family.

---

## Shared recipe (all dense sizes)

1. Probe modules → LoRA targets (Unsloth tip: include MLP on small models).
2. Fertility baseline on CLM `test`.
3. Tokenizer extend (65k append-only; 128k **re-verify special layout** — pad/bos/eos near 124893 on 2.6B).
4. Scaling ladder: smoke → mini → v1/v2 (see SCALING_LADDER.md).
5. Merge + BPB gate.
6. SFT mixture → optional synth → optional DPO.
7. `Evaluation/configs/clm/lfm_<size>_*.yaml`.

Default CPT packing: `seq_len=2048` for ≤1.2B; consider 4096+ for 2.6B/8B once short CPT converges.

---

## A. LFM2.5-230M-Base — edge / smoke vehicle

**Role:** Cheapest full pipeline rehearsal; publishable edge Wolof demo (GGUF story).

| Item | Choice |
|---|---|
| Method | QLoRA CPT + QLoRA SFT only (skip full FT) |
| Tokenizer | Reuse **65k ext** from 1.2B/350M (resize emb) |
| CPT budget | Mini → short v2 early-stop (do not over-train; capacity limited) |
| SFT | Mixture capped; shorter seq (1024–2048); less tool_use |
| Synth | Ortho + tiny instruction only; **no** long-context |
| Eval | BPB/BLEU/chrF + qualitative; latency note optional |
| Hub | `galsenai/LFM2.5-230M-Base_wolof` |

**Risks:** Underfitting Wolof morphology; translation collapse — keep mixture aggressive toward `instruction`.

**Order:** Run smoke+mini **before** 1.2B long jobs to validate scripts.

---

## B. LFM2.5-350M-Base — ablation workhorse

**Role:** Fast scientific grid (tokenizer ±, LoRA vs full, mixture weights) at low cost.

| Item | Choice |
|---|---|
| Method | Full v1 4-run grid + v2 early-stop (mirrors SmolLM3 protocol) |
| Tokenizer | Shared 65k ext; fertility ablation 2k/4k/8k **here first** |
| CPT | Equal-data 1 epoch then best-of-each |
| SFT | Full mixture experiments (translation caps 20/35/50%) |
| Synth | Reasoning 5k pilot; measure AfriMGSM-style before scaling |
| Eval | Primary for hyperparameter papers |
| Hub | `galsenai/LFM2.5-350M-Base_wolof` |

**Why first scientifically:** Same tok family as 1.2B; results transfer; failures are cheap.

---

## C. LFM2.5-1.2B-Base — **hero**

**Role:** Flagship Wolof instruct; JP-style language specialist narrative.

| Item | Choice |
|---|---|
| Method | Best recipe from 350M; LoRA+ext expected champion; full FT optional |
| Tokenizer | Canonical 65k+4k init shipped with model |
| CPT | v2 until val floor (SmolLM3-like) |
| SFT | Balanced mixture + synth reasoning after gate; tool_use slice for demos |
| Ablations | CPT→SFT vs stock Instruct→SFT; ±ext tok |
| Ctx | 2048–4096 SFT; no heavy 32k mid-train required |
| Hub | `galsenai/LFM2.5-1.2B-Base_wolof` + `…_instruct` |
| Publish | Primary model card + GGUF |

**Success:** Beats stock Base BPB by large margin; instruct stays in Wolof on native prompts; Evaluation JSON vs SmolLM3.

---

## D. LFM2.5-2.6B-Base — quality + long-context

**Role:** Best dense Wolof LM; 128k context product claims.

| Item | Choice |
|---|---|
| Method | LoRA CPT (full FT only if multi-GPU); separate **128k** tokenizer extension |
| Tokenizer | Inspect specials at high ids before append; new `tokenizer-ext4000-128k` |
| CPT phase 1 | Short seq (2048) on cleaned CLM until BPB gate |
| CPT phase 2 | Mid-train on `synth_longctx` packs (4k→16k→32k curriculum) |
| SFT | Mixture + reasoning + longctx QA; tool_use for agentic 2.6B story |
| Eval | Standard CLM **plus** needle @ 8k/16k; task-wise SFT |
| Hub | `galsenai/LFM2.5-2.6B-Base_wolof` |
| Hardware | Expect multi-GPU / larger microbatch tricks |

**Footgun:** Do not reuse 65k extended tokenizer. Do not start long-context mid-train before short CPT improves BPB.

---

## E. LFM2.5-8B-A1B-Base — MoE agentic

**Role:** Throughput-efficient large Wolof assistant; tool calling.

| Item | Choice |
|---|---|
| Arch | `Lfm2MoeForCausalLM` — 32 experts, 4 active; PEFT must consider experts |
| Method | QLoRA/LoRA on attention + selected expert projections (probe first) |
| Tokenizer | Share **128k** extension with 2.6B if vocab identical |
| CPT | Shorter than dense 2.6B if unstable; monitor expert load balance if possible |
| SFT | Emphasize `tool_use` + Wolof wrappers + reasoning; Liquid positions this for agents |
| Longctx | Optional lighter mid-train than 2.6B |
| Eval | Tool schema validity + CLM BPB + instruct; note active-param efficiency in paper |
| Hub | `galsenai/LFM2.5-8B-A1B-Base_wolof` |
| Order | **Last** — after 2.6B recipe stable |

**Risks:** MoE fine-tune instability; merge/export for llama.cpp; higher VRAM.

---

## F. Cross-size execution order

```text
230M smoke ──► 350M mini+grid ──► 1.2B hero ──► 230M/350M publish
                                      │
                                      ▼
                               2.6B 128k tok + short CPT
                                      │
                                      ▼
                               longctx mid-train + SFT
                                      │
                                      ▼
                               8B-A1B MoE CPT/SFT
```

## G. Config naming convention

```text
configs/clm/350m/v1_ext4000_lora.yaml
configs/clm/1.2b/v2_ext4000_lora.yaml
configs/clm/2.6b/v1_ext4000_lora.yaml
configs/sft/1.2b/mixture_v1.yaml
configs/scaling/smoke_350m_lora.yaml
configs/scaling/mini_1.2b_lora.yaml
```

Each YAML must set `base_id`, `tokenizer_family: 65k|128k`, `size`, `stage: smoke|mini|v1|v2|sft`.
