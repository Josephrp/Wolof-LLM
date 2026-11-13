# LFM2.5 Wolof — File & line-level TODOs

> **Canonical hierarchical checklist (phases, gates, `uv`, subtasks):**  
> [`COMPREHENSIVE_TODOS.md`](./COMPREHENSIVE_TODOS.md)

This file keeps **SmolLM3 line-anchor** mapping for porting. Use COMPREHENSIVE for execution order.

Status: `[ ]` todo · `[~]` in progress · `[x]` done.

Companion narrative: [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).  
Also: [`docs/DATASETS.md`](./docs/DATASETS.md) · [`docs/SYNTHETIC_DATA.md`](./docs/SYNTHETIC_DATA.md) · [`docs/MODEL_PLANS.md`](./docs/MODEL_PLANS.md) · [`docs/SCALING_LADDER.md`](./docs/SCALING_LADDER.md).

---

## A. Scaffold

- [ ] **A1** Create directory tree from plan §2 (`common/`, `configs/clm/{230m,350m,1.2b,2.6b,8b}/`, `configs/sft/…`, `configs/scaling/`, `scripts/clm|sft|data|family/`, `docs/`, `results/`).
- [ ] **A2** Write `requirements.txt` — start from `../SmolLM3-3B-Base/requirements.txt` L1–86; bump/verify `transformers` loads `Lfm2ForCausalLM`; add `trl`, `sacrebleu` if missing.
- [ ] **A3** Keep `README.md` pipeline commands; link companion docs.
- [ ] **A4** Add `.gitignore` entries for `runs/`, `models/`, `tokenizers/`, large parquet (match SmolLM3 if present).

---

## B. `common/` library

### `common/__init__.py`
- [ ] **B0** Empty or re-export public helpers.

### `common/data.py` ← `SmolLM3-3B-Base/common/data.py`
- [ ] **B1** Support Hub load: `galsenai/cleaned_data_for_clm` train/test **or** local parquet (same schema `text`).
- [ ] **B2** Copy `paqueter` L32–51; document that `tokenizer.eos_token_id` must be **7** (`<|im_end|>`) for LFM 65k family — assert/log in caller (re-check for 128k family).
- [ ] **B3** Add `charger_sft_messages(...)` for `wolof_sft` (`messages`, `task`, `meta` JSON strings).

### `common/metrics.py` ← `SmolLM3-3B-Base/common/metrics.py`
- [ ] **B4** Copy `compute_bpb` L14–49 and `compute_fertility` L52–72 for training-time use.
- [ ] **B5** TODO (later): align windowed BPB with `Evaluation/scripts/evaluate_clm.py` L177–288 so internal/external numbers match (footgun #11).

### `common/journal.py` ← `SmolLM3-3B-Base/common/journal.py`
- [ ] **B6** Copy `ecrire`, `extraire_courbes`, `tracer` unchanged.

### `common/model.py` ← `SmolLM3-3B-Base/common/model.py`
- [ ] **B7** Copy `_attention_dispo` L7–21, `charger` L24–65 — change default `base` comments to LFM; keep vocab assert L57–59.
- [ ] **B8** Adapt `preparer_lora` L68–102:
  - Default `target_modules` from `06_probe_modules.py` output (not SmolLM3 list blindly).
  - Keep `modules_to_save` warning L71–73.
  - Liquid SFT docs use only `q_proj..o_proj` — expose as `target_modules` config key.
- [ ] **B9** Copy `preparer_full` L105–112, `groupes_parametres` L125–162, `geler_backbone` / `degeler_backbone` L165–188.
- [ ] **B10** Extend `_est_embedding` / `NOMS_EMBEDDINGS` L118–122 to also match LFM naming if probe shows different embed module paths.
- [ ] **B11** Add helper `lire_tie_flag(config) -> bool` reading `tie_embedding` **or** `tie_word_embeddings` (1.2B uses `tie_embedding`; 2.6B uses `tie_word_embeddings`).

---

## C. CLM scripts

### `scripts/clm/06_probe_modules.py` (new)
- [ ] **C0** Load `LiquidAI/LFM2.5-1.2B-Base`; print all `nn.Linear` / Conv module suffixes; write `docs/lfm_module_names.txt`.
- [ ] **C0b** Propose CPT vs SFT `target_modules` lists in that file.

### `scripts/clm/00_clean_corpus.py` ← `…/00_clean_corpus.py`
- [ ] **C1** Copy L20–56 (`strip_code`, CLI) unchanged.

### `scripts/clm/01_prepare_data.py` ← `…/01_prepare_data.py`
- [ ] **C2** Copy L22–86 (normalization, leakage, meta JSON).
- [ ] **C3** Document: prefer reusing SmolLM3 `data_root` instead of regenerating.

### `scripts/clm/02_build_tokenizer.py` ← `…/02_build_tokenizer.py` (**major LFM edits**)
- [ ] **C4** Keep `selectionner_candidats` L35–78; after load, print `ignore_merges` / model type from `tokenizer.json`.
- [ ] **C5** **Replace** `construire_tokenizer` L81–122 with append-only:
  - Do **not** reindex specials.
  - `tokenizer.add_tokens(nouveaux)` or edit vocab JSON by assigning ids `len(tok)..`.
  - Return `meta = {V: old_len, N: n_added, id_map: {}}` (empty map).
- [ ] **C6** Keep `valider` L125–169; ensure `clean_up_tokenization_spaces=False` (L148–153).
- [ ] **C7** Rewrite `initialiser_modele` L172–235:
  - Skip special remapping loop L197–198 when `id_map` empty.
  - Mean-pool new rows only L200–208.
  - Keep norm-ratio check L211–213 (**do not recalibrate**).
  - Do **not** remap eos/bos/pad ids (must stay 7/1/0).
  - Set `model.config.vocab_size = len(ext_tok)`; preserve `tie_embedding`.
- [ ] **C8** Update defaults: `base_id=LiquidAI/LFM2.5-1.2B-Base`, paths `tokenizers/lfm25_wolof_ext_*`, `models/lfm25-wolof-ext*-init` (L260–303).
- [ ] **C9** Write `extension_meta.json` fields L305–314 with LFM method string `"append after specials (no reindex)"`.

### `scripts/clm/03_train.py` ← `…/03_train.py`
- [ ] **C10** Copy structure L47–320 (`TrainerLRDifferencie`, smoke, phase A/B, journal).
- [ ] **C11** `resoudre_artefact` L36–44: also accept paths starting with `LiquidAI/`.
- [ ] **C12** Ensure journal records `base_id`, fertility family (`65k`), EOS id.
- [ ] **C13** Memory warning L153–158: retune for 1.2B (full FT needs less than 3B).

### `scripts/clm/04_evaluate.py` ← `…/04_evaluate.py`
- [ ] **C14** Copy L26–135; default `--base-id LiquidAI/LFM2.5-1.2B-Base`.
- [ ] **C15** Fix hardcoded `data_root` L58 — prefer config/env like train script.

### `scripts/clm/05_merge_adapter.py` ← `…/05_merge_adapter.py`
- [ ] **C16** Copy merge + BPB gate L63–167.
- [ ] **C17** Replace `tie_word_embeddings` reads L98–130 with `common.model.lire_tie_flag` / write both keys if present.
- [ ] **C18** Keep tolerance default 0.002 (L71–72).

---

## D. CLM configs

### `configs/tokenizer-ext4000.yaml` ← `configs/v1_donnees_egales/tokenizer-ext4000.yaml`
- [ ] **D1** Set `base_id: LiquidAI/LFM2.5-1.2B-Base`, LFM output paths, `n_tokens: 4000`, seed 42.

### `configs/clm/v1_equal_data/*.yaml` ← SmolLM3 `configs/v1_donnees_egales/*.yaml`
- [ ] **D2** Four YAMLs: default/ext × lora/full; update `base_id`, `tokenizer`, `model`, `run` names (`lfm-…`).
- [ ] **D3** Ext-LoRA: `modules_to_save: [embed_tokens, lm_head]` (as SmolLM3 v2 L20).
- [ ] **D4** Ext-full: include `lr_embeddings`, optional `warmup_embeddings_steps` (SmolLM3 v2 learnings).

### `configs/clm/v2_best_of_each/*.yaml` ← SmolLM3 `configs/v2_best_of_each/*.yaml`
- [ ] **D5** Port patience/early-stop fields (v2 LoRA yaml L32–33).
- [ ] **D6** Fill `data_root` / `output_root` placeholders for the training machine.

---

## E. SFT scripts & configs

### `scripts/sft/10_prepare_sft.py` (new)
- [ ] **E1** Load `galsenai/wolof_sft` (fallback: `vonewman/alpaca-sharegpt-wolof` — Qwen notebook uses this).
- [ ] **E2** Normalize to `messages` list; map ShareGPT `from/value` → `role/content`.
- [ ] **E3** Write parquet under `data/sft/` with train/test; never touch CLM eval texts for instruct test if they overlap — prefer dataset’s own test split (2 250).
- [ ] **E4** Save formatting metadata (template source: Instruct `chat_template.jinja`).

### `scripts/sft/11_train_sft.py` (new; Liquid TRL pattern)
- [ ] **E5** `SFTTrainer` + `SFTConfig` + `LoraConfig` per [Liquid TRL docs](https://docs.liquid.ai/lfm/fine-tuning/trl).
- [ ] **E6** CLI `--config configs/sft/*.yaml`; support `model:` = merged CPT path.
- [ ] **E7** Assistant-only loss; packing optional; bf16; seed 42.
- [ ] **E8** Save adapter + tokenizer + `journal_sft.json`.

### `scripts/sft/12_eval_instruct.py` (new)
- [ ] **E9** Held-out SFT metrics (loss and/or generation chrF on translations).
- [ ] **E10** Qualitative prompt suite (Wolof-only system prompt).
- [ ] **E11** Call into Evaluation CLM config for continuity BPB on instruct checkpoint.

### `configs/sft/wolof-sft-lora.yaml`
- [ ] **E12** LR ~2e-4 LoRA, r=16–32, attention targets (Liquid default), `modules_to_save` if ext tok.
- [ ] **E13** `wolof-sft-from-merged.yaml` pointing at Phase E merge output.

---

## F. Evaluation/ (repo root)

- [ ] **F1** Add `Evaluation/configs/clm/lfm_base.yaml` — `base_model_id: LiquidAI/LFM2.5-1.2B-Base`, `output_json: ../metrics/clm/lfm_base.json` (mirror `smol_base.yaml` / `qwen.yaml`).
- [ ] **F2** Add `lfm_clm.yaml` — Hub CPT subfolder (after publish), mirror `smol_clm.yaml` L28–34.
- [ ] **F3** Add `lfm_instruct.yaml` — Wolof instruct Hub id.
- [ ] **F4** Run `evaluate_clm.py` for each; commit JSON under `Evaluation/metrics/clm/`.
- [ ] **F5** (Family) configs for 350M / 2.6B when trained.

---

## G. Notebooks (optional, non-blocking)

- [ ] **G1** `notebooks/clm/tokenizer_probe.ipynb` — fertility curves, sample encodings (inspired by SmolLM3 `tokenizer_extension.ipynb`).
- [ ] **G2** `notebooks/sft/sft_smoke.ipynb` — small TRL run (Qwen SFT notebook pattern: `apply_chat_template`, `SFTTrainer`).
- [ ] **G3** `notebooks/README.md` — “exploration only; scripts are source of truth” (SmolLM3 notebooks README).

---

## H. Docs & publication

### `docs/FOOTGUNS.md`
- [ ] **H1** Expand plan §4 into operator checklist with “if you see X, you forgot Y”.

### `docs/FAMILY_EVAL.md`
- [ ] **H2** Table template: size × tok family × method × BPB × BLEU × chrF × hours.
- [ ] **H3** Rules: never compare PPL across 65k vs 128k; mark tokenizer family column.

### `docs/PUBLICATION.md`
- [ ] **H4** Model card sections: intended use, training data, eval, carbon/hardware, limitations (code-switch, orthography).
- [ ] **H5** Hub layout: `v1/`, `v2/ext4000-lora`, `instruct/v1`.
- [ ] **H6** Blog outline: (1) why Base not Instruct, (2) append-vs-reindex tokenizer, (3) BPB story, (4) family curves, (5) edge GGUF teaser.
- [ ] **H7** HF Collection + link from org `galsenai`.
- [ ] **H8** Cite LFM2 paper `arXiv:2511.23404` and SmolLM3 Wolof results for cross-family narrative.

---

## I. Family extensions (see `docs/MODEL_PLANS.md`)

- [ ] **I1** 230M: resize 65k ext; LoRA CPT+SFT; GGUF note.
- [ ] **I2** 350M: full scientific grid + vocab ablation host.
- [ ] **I3** 1.2B: hero publish.
- [ ] **I4** 2.6B: **new** 128k tokenizer; verify specials near ~124893 before append.
- [ ] **I5** 2.6B: short CPT gate → longctx mid-train → SFT.
- [ ] **I6** 8B-A1B: PEFT expert targeting; tool-heavy SFT; last in queue.
- [ ] **I7** Evaluation YAMLs: `lfm_{230m,350m,1.2b,2.6b,8b}_{base,clm,instruct}.yaml`.

---

## J. Acceptance gates (block publish)

- [ ] **J1** Tokenizer validations pass (`valider` OK).
- [ ] **J2** Smoke CPT completes both default + ext LoRA.
- [ ] **J3** v1 or v2 table written to `results/comparaison.csv`.
- [ ] **J4** Merge BPB delta ≤ 0.002.
- [ ] **J5** Evaluation JSON for base + CPT on Hub corpus.
- [ ] **J6** Instruct smoke generations in Wolof reviewed by a native speaker (process gate, not code).
- [ ] **J7** Scaling ladder stages 0–4 logged green before 1.2B multi-day jobs.
- [ ] **J8** SFT mixture uses task weights (not raw 57.9% translation).

---

## K. Datasets & synthetic

- [ ] **K1** `scripts/data/20_audit_datasets.py` → `results/data_audit.json`.
- [ ] **K2** `scripts/data/21_build_sft_mixture.py` + val carve 2% stratified.
- [ ] **K3** `configs/sft/mixture_v1.yaml` (translation ≤35%, boost instruction/qa).
- [ ] **K4** `22_synth_reasoning.py` 5k pilot + SYNTHETIC gates G0–G2.
- [ ] **K5** `22_synth_longctx.py` for 2.6B stage.
- [ ] **K6** `22_synth_ortho.py` / `22_synth_prefs.py` after SFT exists.
- [ ] **K7** Model cards disclose SFT Afri* pretrain-overlap caveat.

---

## L. Scaling ladder configs

- [ ] **L1** Create `configs/scaling/smoke_*.yaml` and `mini_*.yaml` (SCALING_LADDER.md).
- [ ] **L2** Run stages 0–4 on 350M; `results/scaling_log.md`.
- [ ] **L3** Promote 1.2B only after 350M mini BPB gate.
- [ ] **L4** 2.6B 128k smoke before longctx.
- [ ] **L5** 8B-A1B MoE smoke before mini.

---

## Line-anchor quick map (SmolLM3 → LFM action)

| Anchor | Action |
|---|---|
| `02_build_tokenizer.py` L81–122 | Replace reindex with append (65k); re-check for 128k |
| `02_build_tokenizer.py` L172–186 | Keep no-norm-recalibrate doctrine |
| `02_build_tokenizer.py` L196–198 | Skip if no `id_map` |
| `common/model.py` L71–78 | Keep `modules_to_save` doctrine |
| `common/model.py` L125–162 | Keep dual LR |
| `03_train.py` L192–228 | Keep embedding warmup for full+ext |
| `05_merge_adapter.py` L10–33, L123–145 | Port; generalize `tie_*` |
| `04_evaluate.py` L69–112 | Port BPB comparison narrative |
| `Evaluation/.../evaluate_clm.py` L177–288 | Canonical external BPB |
| `Evaluation/.../download_dataset.py` | Default `cleaned_data_for_clm` |
| `Evaluation/.../smol_clm.yaml` L28–34 | Template for `lfm_*.yaml` |
| Liquid TRL / Unsloth LoRA targets | SFT default; CPT may widen + MLP |

---

## Priority order (do in sequence)

Follow **[`COMPREHENSIVE_TODOS.md`](./COMPREHENSIVE_TODOS.md)** phases P0→P11.

Quick map:
1. P0 data audit + P1 `uv` (`pyproject.toml` / `uv.lock`)  
2. P2 `common/` + P3 mixtures  
3. P4 tokenizer 65k  
4. P5–P6 scaling smoke/mini on **350M** → 1.2B CPT  
5. P7 SFT → P9 Evaluation  
6. P8 synthetic (gated) → P10 family → P11 publish  
