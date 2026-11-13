# LFM2.5 Wolof — Implementation Plan

Complete plan to produce and evaluate Wolof **continual-pretrained + instruct** models on Liquid AI LFM2.5, mirroring the proven SmolLM3 pipeline and feeding the shared `Evaluation/` harness.

---

## 0. Verdict (read first)

| Decision | Choice | Why |
|---|---|---|
| **Primary base** | [`LiquidAI/LFM2.5-1.2B-Base`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) | Official CPT starting point (28T pretrain, 1.17B, vocab **65 536**). Liquid explicitly recommends Base for language-specific heavy fine-tuning (same pattern as their JP specialist). |
| **Folder name** | Keep `Experiments/LFM2.5-Base-1B/` | Historical; map internally to **1.2B**. Do not rename mid-project. |
| **Method template** | Port `Experiments/SmolLM3-3B-Base/` scripts → runnable LFM scripts | SmolLM3 is the only complete, footgun-aware pipeline (tokenizer surgery, BPB, LoRA+`modules_to_save`, embedding warmup, merge verification). |
| **Instruct path** | CPT (CLM) → merge → SFT on ChatML (`galsenai/wolof_sft`) → optional DPO | Matches Liquid post-training recipe (SFT → preference) and repo goal (instruct models). |
| **Primary metric** | **BPB** (bits-per-byte) on shared eval split | Only fair comparison across tokenizers; PPL is secondary and tokenizer-tied. |
| **Publish** | `galsenai/LFM2.5-*-Base_wolof` family (+ analyses) | Hero 1.2B first; then edge + 2.6B + MoE. |
| **CPT data** | `galsenai/cleaned_data_for_clm` | SmolLM3-frozen splits; not raw centralized. |
| **SFT data** | Reweighted `galsenai/wolof_sft` | Cap translation; boost instruction/qa; + gated synth. |
| **Scale rule** | `docs/SCALING_LADDER.md` | Smoke → mini (350M) → hero 1.2B → family. |

**Do not start from Instruct for CPT.** Instruct already spent alignment budget on EN-centric chat; language adaptation belongs on Base (Liquid + JP precedent).

---

## 1. Investigation summary

### 1.1 Existing codebase (what to reuse)

| Component | Location | Role for LFM |
|---|---|---|
| Corpus clean | `SmolLM3-3B-Base/scripts/00_clean_corpus.py` | Copy almost verbatim |
| Train/eval splits + leakage | `…/01_prepare_data.py` | Copy; **reuse same frozen splits** if possible so LFM ↔ SmolLM BPB is comparable |
| Tokenizer extension | `…/02_build_tokenizer.py` | **Port with LFM-specific surgery** (specials layout differs — §3) |
| CPT train | `…/03_train.py` + `common/{model,data,metrics,journal}.py` | Port; adapt LoRA targets + `tie_embedding` |
| Unified BPB compare | `…/04_evaluate.py` | Port |
| LoRA merge + tying check | `…/05_merge_adapter.py` | Port; handle `tie_embedding` **and** `tie_word_embeddings` |
| Hub CLM eval | `Evaluation/scripts/evaluate_clm.py` + verbalizer | Add LFM YAML configs; **do not fork BPB** |
| Exploratory notebooks | SmolLM3 `notebooks/`, Qwen SFT notebook | Reference only; ship **scripts** as source of truth |
| Qwen CLM script | `Qwen3.5-Base-2B/scripts/clm/train_clm.py` | Too thin (no packing/BPB/tokenizer); do **not** use as CPT template |
| SFT data | `galsenai/wolof_sft` (private, ~990k ChatML) | Primary instruct data; fallback public Alpaca-Wolof if needed |

SmolLM3 hard-won findings to preserve:

1. **BPB over PPL** for cross-tokenizer claims (`SmolLM3 README` + `common/metrics.py`).
2. **Do not recalibrate** mean-pooled embedding norms (`02_build_tokenizer.py` L172–186).
3. LoRA **must** use `modules_to_save: [embed_tokens, lm_head]` when vocab is extended (`common/model.py` L68–78).
4. Full FT + extended vocab needs **lr_embeddings** + optional embedding warmup (`03_train.py` L192–228, `model.py` L125–188).
5. Merge must verify BPB and fix tying flags (`05_merge_adapter.py` L10–33, L123–145).
6. Eval split must be leakage-cleaned (`01_prepare_data.py` L51–68).

### 1.2 LFM2.5 family (Hugging Face, investigated)

| Hub id | Params | Vocab | Role |
|---|---:|---:|---|
| **`LiquidAI/LFM2.5-1.2B-Base`** | 1.17B | **65 536** | **Primary CPT base** |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.17B | 65 536 | Reference chat / optional SFT-from-Instruct ablation |
| `LiquidAI/LFM2.5-1.2B-Thinking` | 1.17B | 65 536 | Out of scope for v1 Wolof |
| `LiquidAI/LFM2.5-1.2B-JP` | 1.17B | 65 536 | **Precedent**: language specialist from Base |
| `LiquidAI/LFM2.5-350M-Base` | 354M | **65 536** | Cheap ablation / family size curve (shared tokenizer family) |
| `LiquidAI/LFM2.5-230M-Base` | 230M | **65 536** | Optional edge size |
| `LiquidAI/LFM2.5-2.6B-Base` | 2.70B | **128 000** | Quality ceiling; **separate tokenizer extension** |
| `LiquidAI/LFM2.5-8B-A1B-Base` | 8.5B (MoE, ~1.5B active) | **128 000** | Optional later; `Lfm2MoeForCausalLM` |

Architecture (all dense): `Lfm2ForCausalLM`, hybrid **gated short conv + GQA**, `tie_embedding: true` (1.2B/350M) or `tie_word_embeddings: true` (2.6B). Context advertised 32k; config `max_position_embeddings` up to 128k. Paper: [arXiv:2511.23404](https://arxiv.org/abs/2511.23404).

Official fine-tuning (Liquid docs):

- CPT: Unsloth notebooks (completion / translation).
- SFT/DPO: TRL; recommended LoRA targets for chat: `q_proj,k_proj,v_proj,o_proj` ([docs](https://docs.liquid.ai/lfm/fine-tuning/trl)).
- For **Wolof CPT with vocab extension**, prefer SmolLM3-style attention+MLP targets + `modules_to_save` embeddings; treat Liquid’s attention-only list as the **SFT default**, not the CPT default.

### 1.3 Tokenizer facts that change the SmolLM3 recipe

LFM2.5-1.2B specials live at **low ids**, not high ids:

| id | token |
|---:|---|
| 0 | `<\|pad\|>` |
| 1 | `<\|startoftext\|>` (BOS) |
| 2 | `<\|endoftext\|>` |
| 6 | `<\|im_start\|>` |
| 7 | `<\|im_end\|>` (**EOS** in `config` / `generation_config`) |
| 8–13 | tool markers |
| 16+ | reserved |

SmolLM3 put BPE first and specials at the **end**, so `02_build_tokenizer.py` had to **reindex specials** (L81–122, L196–198). **LFM does not need that remap** if new Wolof tokens are **appended after `len(tokenizer)-1`**.

Still mandatory on LFM:

- Validate `len(tok)` vs `config.vocab_size` (SmolLM3 warned about using `vocab_size` alone — L271–274).
- Packing must use **EOS = 7** (`<|im_end|>`), not `<|endoftext|>` id 2.
- Chat template uses ChatML-like `<|im_start|>` / `<|im_end|>` (Base ships a simple template; Instruct’s is richer — copy Instruct template into the final SFT artifact).
- Discover whether `ignore_merges` exists before choosing “whole-word only” candidate selection (SmolLM3 L38–46). If false, candidate selection / monotonicity tests may need adjustment.
- **Two tokenizer families** in LFM2.5: 65k (≤1.2B) vs 128k (≥2.6B). Never mix fertility/BPB interpretation without stating the family.

### 1.4 Data assets (investigated Hub cards)

Deep dives: [`docs/DATASETS.md`](./docs/DATASETS.md), synthetic strategy [`docs/SYNTHETIC_DATA.md`](./docs/SYNTHETIC_DATA.md).

| Hub dataset | Role | Key facts |
|---|---|---|
| [`galsenai/cleaned_data_for_clm`](https://huggingface.co/datasets/galsenai/cleaned_data_for_clm) | **Primary CPT train/test** | Already SmolLM3 splits: train **958 646**, test **4 282** (`text`). Code spans stripped, NFC, seed 42, quasi-dup eval removed. **Reuse — do not rebuild.** Wired in `Evaluation/scripts/download_dataset.py`. |
| [`galsenai/wolof_centalized_corpus`](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus) | Raw provenance / domain filters / tokenizer mining | **1 018 520** unique rows; `text` + `sources[]`; ~72 M whitespace tokens; 47 sources; GlotLID ~97% wolof. Code source dominates **~70% of tokens**. Median length **97 chars** → long-context is **not** native. |
| [`galsenai/wolof_sft`](https://huggingface.co/datasets/galsenai/wolof_sft) | **Primary SFT** | ChatML `messages`; train **990 320** / test **2 250**; tasks: translation 57.9%, tool_use 20.9%, classification 12.2%, instruction 6.5%, intent/ner/qa tiny. Test = Belebele/AfriMMLU/AfriMGSM/AfriXNLI (pretrain leakage risk). |
| Evaluation CSV + verbalizer | External CLM leaderboard | Same protocol as Smol/Qwen/Aya |

**Default data policy:** CPT on `cleaned_data_for_clm`; SFT on stratified/`task`-balanced slices of `wolof_sft`; use centralized corpus only for audits, source-aware ablations, and synthetic seed mining.

---

## 2. Target directory layout

Create under `Experiments/LFM2.5-Base-1B/`:

```text
LFM2.5-Base-1B/
├── README.md
├── IMPLEMENTATION_PLAN.md             # this file
├── FILE_TODOS.md
├── requirements.txt
├── common/ …
├── configs/
│   ├── tokenizer-ext4000-65k.yaml     # shared by 230M/350M/1.2B
│   ├── tokenizer-ext4000-128k.yaml    # 2.6B / 8B-A1B family
│   ├── clm/{230m,350m,1.2b,2.6b,8b}/  # per-size YAML grids
│   ├── sft/{230m,350m,1.2b,2.6b,8b}/
│   └── scaling/                       # smoke / mini / full stage configs
├── scripts/
│   ├── clm/ …                         # 00–06 as before
│   ├── sft/ …                         # 10–12 + 13_stratify_sft.py
│   ├── data/
│   │   ├── 20_audit_datasets.py
│   │   ├── 21_build_sft_mixture.py    # task weights + val carve
│   │   └── 22_synth_*.py              # synthetic builders (see SYNTHETIC_DATA)
│   └── family/
│       └── 30_run_size.py             # thin wrapper: size → config set
├── notebooks/ …
├── docs/
│   ├── FOOTGUNS.md
│   ├── FAMILY_EVAL.md
│   ├── PUBLICATION.md
│   ├── DATASETS.md
│   ├── SYNTHETIC_DATA.md
│   ├── MODEL_PLANS.md                 # per-size detailed plans
│   └── SCALING_LADDER.md              # small tests → scale with evals
└── results/{smoke,mini,v1,v2,sft,family}/
```

Also add under repo `Evaluation/`:

```text
Evaluation/configs/clm/lfm_{230m,350m,1.2b,2.6b,8b}_{base,clm,instruct}.yaml
```

Per-size training details: [`docs/MODEL_PLANS.md`](./docs/MODEL_PLANS.md).  
Small→scale gates: [`docs/SCALING_LADDER.md`](./docs/SCALING_LADDER.md).
---

## 3. Phase plan

### Phase A — Scaffold + environment (1–2 days)

1. Create layout above; stub README with pipeline commands (mirror SmolLM3 README L8–41).
2. Pin deps: `transformers>=4.57` (Liquid cards cite 4.57–5.x), `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, `torch` bf16, `pyyaml`, `sacrebleu` (for Evaluation). Prefer aligning with SmolLM3 `requirements.txt` where possible; **verify `Lfm2ForCausalLM` loads** before training.
3. Run `scripts/clm/06_probe_modules.py` on Base → write `docs/lfm_module_names.txt` (attention, MLP, conv). Decide CPT LoRA `target_modules`.

### Phase B — Data (use published Hub splits)

1. **CPT:** `load_dataset("galsenai/cleaned_data_for_clm")` → map to local parquet or stream in `common/data.py`. This **is** the SmolLM3 frozen protocol (code strip + NFC + seed 42 + clean eval). Prefer Hub over re-running `00`/`01` unless regenerating.
2. **Audit only:** load `wolof_centalized_corpus` to verify source balance, length histograms, orthography variants; never train CPT on raw Code-170k-dominated tokens without the cleaned pipeline.
3. **SFT:** `wolof_sft` train/test; carve **validation ~2% stratified by `task`+`source`** from train (card requirement). Apply task mixture weights (see DATASETS.md) — naive full shuffle → translator model.
4. **Leakage honesty:** SFT `test` benchmarks may appear in pretrain raw text — report that limitation; plan a native held-out Wolof set for publication claims.
5. Orthography: train/eval must share the same normalization as `cleaned_data_for_clm` (not the aggressive lowercase-only Data-Preprocessing notebook path unless that matches the Hub split).

### Phase C — Custom tokenizer (critical)

Port `02_build_tokenizer.py` with these LFM deltas:

| SmolLM3 behavior | LFM behavior |
|---|---|
| Reindex specials after BPE (`construire_tokenizer` L81–122) | **Append** new tokens; leave ids 0–N specials untouched |
| Move embedding rows via `id_map` (L196–198) | **No special remap**; only init new rows `[old_V : new_V)` |
| Remap `eos/bos/pad` in config (L220–228) | Keep eos=7, bos=1, pad=0 unless append shifts them (it must not) |
| `ignore_merges=True` whole-word strategy | Probe first; if absent, still rank by `freq*(n_tok-1)` but validate monotonicity harder |

Keep unchanged:

- Candidate ranking by expected token savings (L48–49, L65–77).
- Validations: monotonicity, round-trip, FR/EN/code non-regression (L125–169).
- Mean-pool init **without** norm recalibration (L172–186) — tied embeddings still apply (`tie_embedding: true`).
- Ablation 4k vs 8k (SmolLM3 found 4k enough) — rerun fertility+BPB-before on Wolof sample; default **+4000**.

Deliverables:

- `tokenizers/lfm25_wolof_ext_4000/`
- `models/lfm25-wolof-ext4000-init/`
- `extension_meta.json` (base id, N, fertility before/after, norm ratio, seed)

### Phase D — Continual pretraining grid (CLM)

Mirror SmolLM3 v1 then v2:

**v1 — equal data, 1 epoch** (4 runs):

| Run | Tokenizer | Method |
|---|---|---|
| `wolof-default-lora` | stock 65k | QLoRA |
| `wolof-default-full` | stock 65k | full |
| `wolof-ext4000-lora` | +4k | QLoRA + `modules_to_save` |
| `wolof-ext4000-full` | +4k | full + `lr_embeddings` (+ warmup if needed) |

Suggested starting HPs (adapt from SmolLM3 v2 LoRA yaml; scale batch to VRAM):

- `seq_len: 2048` (or 1024 if memory-bound; LFM context allows more but corpus median is short — packing dominates).
- LoRA: `r=32, alpha=64`, dropout 0.05; CPT targets = attention + MLP (from probe); `modules_to_save: [embed_tokens, lm_head]` when extended.
- LR LoRA ~`2e-4`; full backbone ~`2e-5`; embeddings ~`2e-4`–`4e-4`.
- Effective tokens/step ≈ micro_bs × grad_acc × seq_len (document in journal).
- `--smoke` path mandatory (10 steps) before full runs.

**v2 — best of each**: early stopping on eval loss (`patience`), differentiated LRs, embedding warmup for full+ext (SmolLM3 `03_train.py` L192–228).

Champion CPT checkpoint → Phase E.

### Phase E — Merge + Hub CPT publish

1. `05_merge_adapter.py` with BPB tolerance check.
2. Set `config.tie_embedding` / `tie_word_embeddings` to match weight reality (SmolLM3 L123–130).
3. Push `galsenai/LFM2.5-1.2B-Base_wolof` with subfolders e.g. `v1/...`, `v2/ext4000-lora`.
4. Add `Evaluation/configs/clm/lfm_clm.yaml` pointing at Hub + subfolder.

### Phase F — Instruct (SFT)

1. `21_build_sft_mixture.py`: stratified mixture from `wolof_sft` (downweight translation; upweight `instruction`/`qa`/`ner`/`intent`; keep a tool_use slice for agentic story).
2. `10_prepare_sft.py`: ChatML → LFM Instruct template; parse `meta`/`tool_calls` JSON strings; assistant-only loss.
3. Start from **merged CPT** for headline Wolof models.
4. Ablations: CPT→SFT vs Instruct-only; balanced vs raw task mix; ± synthetic reasoning/long-context packs.
5. TRL `SFTTrainer`; then optional DPO on synthetic preferences.
6. Eval: SFT val loss by task, Generation on native prompts, CLM BPB continuity, Evaluation harness.

### Phase G — Full family (all LFM2.5 text Bases)

Execute per [`docs/MODEL_PLANS.md`](./docs/MODEL_PLANS.md) after 1.2B hero gates pass:

| Size | Base Hub id | Vocab | Context | Priority |
|---|---|---:|---:|---|
| 230M | `LiquidAI/LFM2.5-230M-Base` | 65 536 | 32k | Edge curve; LoRA CPT+SFT only |
| 350M | `LiquidAI/LFM2.5-350M-Base` | 65 536 | 32k | Cheap full recipe + ablations |
| **1.2B** | `LiquidAI/LFM2.5-1.2B-Base` | 65 536 | 32k | **Hero** |
| 2.6B | `LiquidAI/LFM2.5-2.6B-Base` | **128 000** | **128k** | Quality + long-context mid-train |
| 8B-A1B | `LiquidAI/LFM2.5-8B-A1B-Base` | **128 000** | **128k** | MoE; tool/agent focus; last |

Shared 65k extended tokenizer across 230M/350M/1.2B (resize embeddings per size). Separate 128k extension for 2.6B/8B after inspecting special id layout.

### Phase H — Publication & findings presentation

See `docs/PUBLICATION.md`. Emphasize family Pareto (quality vs size vs tok/s) and dataset limitations (SFT translation skew, benchmark pretrain exposure).

---

## 4. Common footguns (LFM-specific + project-wide)

| # | Footgun | Symptom | Mitigation |
|---|---|---|---|
| 1 | Norm-recalibrate new embeddings | BPB explodes; model emits only new tokens | Mean-pool only; expect norm ratio ~0.5 |
| 2 | LoRA without `modules_to_save` on new vocab | Silent failure; new tokens never train | Always save `embed_tokens` (+ `lm_head` if untied copy) |
| 3 | Merge with wrong `tie_*` | Half of embedding learning discarded | `05_merge` BPB gate |
| 4 | Compare PPL across tokenizers | Extended looks worse | Report **BPB**; PPL only within same tok |
| 5 | Train/eval leakage | Inflated BPB | Use `wolof_eval_clean_v1` |
| 6 | Pack with wrong EOS | Broken sequence boundaries | EOS id **7** |
| 7 | SmolLM3-style special reindex on LFM | Corrupt chat/tool tokens | Append-only extension |
| 8 | Start CPT from Instruct | Overwrites alignment; weaker language gain | Use Base |
| 9 | Attention-only LoRA for CPT+new vocab | Slow embedding uptake | CPT: broader targets + modules_to_save |
| 10 | Mixed casing / verbalizer mismatch | Unfair eval | Freeze one normalization; document |
| 11 | Truncate long texts in BPB while counting all bytes | Deflated BPB | Prefer Evaluation `compute_bpb` windowing (`evaluate_clm.py` L191–199); migrate SmolLM3 metric later |
| 12 | 65k vs 128k family confusion | Invalid cross-size claims | State tokenizer family in every table |
| 13 | Flash-attn / transformers version | Load errors on `lfm2` | Smoke-load before multi-hour runs |
| 14 | Code-switch Wolof–French | Model FR-defaults | Mix translation SFT + Wolof-only system prompts; evaluate both |
| 15 | Raw SFT without task weights | Model becomes a translator | Stratified mixture (`21_build_sft_mixture.py`) |
| 16 | CPT on raw centralized corpus | Wasted compute on code tokens | Use `cleaned_data_for_clm` only |
| 17 | Trust SFT test as “unseen” | Inflated Afri* scores | Disclose pretrain overlap; add native Wolof test |
| 18 | Long-context fine-tune on short corpus alone | No length generalization | Synthetic long packs + 2.6B/8B mid-train (§ SCALING / SYNTHETIC) |
| 19 | MoE LoRA without expert targets | Under-adapts 8B-A1B | Probe experts; PEFT MoE guidance |

### Quality levers for a strong Wolof instruct model

1. **Representation**: tokenizer extension with validated fertility drop (~SmolLM3 −36% target).
2. **CPT until val floor** (v2 early stop), not a single under-trained epoch for the release model.
3. **Protect pretrained weights** (LoRA or low LR + embedding warmup).
4. **SFT on native ChatML** with correct template and assistant masking.
5. **Task mixture**: downweight translation; upweight instruction/QA; keep tool_use for LFM agentic narrative.
6. **Synthetic fill** for reasoning, long-context, orthography, preferences (SYNTHETIC_DATA.md).
7. **Human eval** across Latin Wolof orthographies.
8. **Ship GGUF** for edge sizes (230M/350M/1.2B) after quality lock.
9. **Scale only after stage gates** (SCALING_LADDER.md) — never jump smoke → multi-day 2.6B.

---

## 5. Evaluation protocol

### Internal (during training)

- Fertility on eval split.
- BPB before/after each run (`common/metrics.py`).
- Final `04_evaluate.py` recalculation on same N texts for all local runs.

### External (publication)

`Evaluation/scripts/evaluate_clm.py` with verbalizer on:

- stock Base
- CPT best
- Instruct Wolof
- (family) 350M / 2.6B variants

Metrics: BPB, PPL (footnote), BLEU, chrF — same seeds as existing configs (`bleu_seed: 0`, `bleu_n_texts: 100`).

### Cross-project leaderboard

Fill rows alongside existing `Evaluation/metrics/clm/{smol_*,qwen,aya,oolel}.json`.

---

## 6. Success criteria (v1)

- [ ] Extended tokenizer: fertility ↓ ≥20% vs stock on eval; monotonicity 0 regressions; FR/EN/code non-regression pass.
- [ ] CPT LoRA+ext BPB ≪ stock Base BPB (expect large drop; SmolLM3 Base was ~2.16 → ~0.59 at v2).
- [ ] Merged model BPB within 0.002 of adapter BPB.
- [ ] SFT model follows Wolof instructions on held-out set without collapsing to French-only.
- [ ] Evaluation JSON published for Base / CPT / Instruct.
- [ ] Model card + reproducibility configs on Hub + this folder.

---

## 7. Out of scope (v1 hero track)

Still deferred until scaling ladder green on 1.2B:

- VL / Audio LFM2.5
- Distilling `LFM2.5-1.2B-Thinking` into Wolof
- Unsloth as primary path (allowed as optional speed notebook; scripts = TRL/Transformers)
- Replacing Evaluation verbalizer

**In scope for family track (after hero):** 230M, 350M, 2.6B, 8B-A1B Base → CPT → SFT as in MODEL_PLANS.md.

---

## 8. Suggested execution order (checklist)

Follow **SCALING_LADDER** stages; condensed:

1. Audit datasets (`20_audit_datasets.py`) + probe modules on 1.2B and 350M.
2. Wire `cleaned_data_for_clm` as CPT source; fertility baseline.
3. Tokenizer append (65k) + validations.
4. Stage smoke → mini CPT on **350M** (fast) then **1.2B**.
5. 1.2B v1 grid → v2 champion → merge → Evaluation JSON.
6. SFT mixture + smoke SFT → full SFT on 1.2B.
7. Roll family: 230M → 350M full publish → 2.6B (128k tok) → 8B-A1B.
8. Synthetic reasoning/long-context packs → mid-train 2.6B/8B → re-eval.
9. Publication pack (collection, cards, leaderboard, blog).

---

## 9. Index of detailed companion docs

| Doc | Contents |
|---|---|
| [`COMPREHENSIVE_TODOS.md`](./COMPREHENSIVE_TODOS.md) | **Canonical** hierarchical todos + gates + `uv` |
| [`docs/DATASETS.md`](./docs/DATASETS.md) | Full dataset investigation + how to use each |
| [`docs/SYNTHETIC_DATA.md`](./docs/SYNTHETIC_DATA.md) | Where/why synthetic data; recipes; eval gates |
| [`docs/MODEL_PLANS.md`](./docs/MODEL_PLANS.md) | Individual plans for 230M / 350M / 1.2B / 2.6B / 8B-A1B |
| [`docs/SCALING_LADDER.md`](./docs/SCALING_LADDER.md) | Small-scale tests → scale with evals |
| [`docs/FOOTGUNS.md`](./docs/FOOTGUNS.md) | Operator traps |
| [`docs/FAMILY_EVAL.md`](./docs/FAMILY_EVAL.md) | Cross-size tables |
| [`docs/PUBLICATION.md`](./docs/PUBLICATION.md) | Findings presentation |
| [`FILE_TODOS.md`](./FILE_TODOS.md) | SmolLM3 file/line porting anchors |
