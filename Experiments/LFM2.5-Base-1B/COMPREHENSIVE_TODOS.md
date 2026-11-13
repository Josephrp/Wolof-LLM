# LFM2.5 Wolof — Comprehensive TODO list

Canonical hierarchical checklist. Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[GATE]` must pass before children.

**Order:** data investigations → `uv` envs/builds → shared libraries → scripts/configs → scaling gates → family → publish.

Companions: [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) · [`docs/DATASETS.md`](./docs/DATASETS.md) · [`docs/SYNTHETIC_DATA.md`](./docs/SYNTHETIC_DATA.md) · [`docs/MODEL_PLANS.md`](./docs/MODEL_PLANS.md) · [`docs/SCALING_LADDER.md`](./docs/SCALING_LADDER.md) · [`FILE_TODOS.md`](./FILE_TODOS.md) (SmolLM3 line anchors).

**Run convention (after §1):** prefer `uv run …` inside this experiment dir; never mix system Python with training deps.

---

## 0. Master phases (top-level)

| ID | Phase | Blocks |
|---|---|---|
| **P0** | Data investigations & audits | Everything training |
| **P1** | `uv` project, envs, builds, CUDA wheels | All scripts |
| **P2** | Scaffold + `common/` library | CLM/SFT scripts |
| **P3** | Data scripts + mixtures | CPT/SFT runs |
| **P4** | Tokenizer (65k, then 128k) | Ext CPT |
| **P5** | CLM train/eval/merge scripts + configs | Scaling ladder |
| **P6** | Scaling tests Stages 0–7 (350M → 1.2B CPT) | Hero CPT |
| **P7** | SFT scripts + mixture training | Instruct |
| **P8** | Synthetic data (gated) | Reasoning / longctx / DPO |
| **P9** | Evaluation harness + Hub metrics | Publish claims |
| **P10** | Family 230M / 350M / 2.6B / 8B-A1B | Collection |
| **P11** | Publication & findings | External |

---

## P0 — Data investigations

### P0.1 File: `scripts/data/20_audit_datasets.py` *(create)*

- [x] **P0.1.0** CLI: `--clm`, `--sft`, `--raw`, `--out`, `--hf-token-env HF_TOKEN`, `--max-rows` for sampling.
- [x] **P0.1.1** Load `galsenai/cleaned_data_for_clm` via `datasets`.
  - [x] **P0.1.1.a** Assert splits: train ≈ 958 646, test ≈ 4 282.
  - [x] **P0.1.1.b** Column check: only/`text` present; dtypes; null rate = 0.
  - [x] **P0.1.1.c** Length histogram (chars, whitespace tokens): p50/p90/p99; write to JSON.
  - [x] **P0.1.1.d** Sample 20 rows to `results/audit/clm_samples.jsonl` for human skim.
- [x] **P0.1.2** Load `galsenai/wolof_centalized_corpus`.
  - [x] **P0.1.2.a** Assert ~1 018 520 rows; columns `text`, `sources`.
  - [x] **P0.1.2.b** Source frequency table (top 20); flag `Code-170k-wolof` token mass.
  - [x] **P0.1.2.c** Estimate % rows with `len(text) > 2000` / `> 8000` (long-context seed pool).
  - [x] **P0.1.2.d** Orthography / GlotLID note (document; optional sample LID later).
- [x] **P0.1.3** Load `galsenai/wolof_sft` (`data/train.jsonl`, `data/test.jsonl`).
  - [x] **P0.1.3.a** Assert train ≈ 990 320, test ≈ 2 250.
  - [x] **P0.1.3.b** Task histogram; fail soft-warn if translation > 50%.
  - [x] **P0.1.3.c** `prompt_lang` histogram; multi-turn count; tool_call count.
  - [x] **P0.1.3.d** Validate `meta` / `tool_calls[].arguments` parse as JSON strings on N=1000 sample.
  - [x] **P0.1.3.e** List test `source` / benchmark ids (Belebele, AfriMMLU, …); write leakage caveat stub.
- [x] **P0.1.4** Cross-checks.
  - [x] **P0.1.4.a** Exact-overlap sample: SFT assistant text vs CLM test (report rate).
  - [x] **P0.1.4.b** Write `results/data_audit.json` + `results/data_audit.md` summary.
- [x] **[GATE G-DATA]** PASS 2026-09-16: full audit artifacts exist; CLM test = 4,282 exactly; SFT task counts match the card; null text = 0.

### P0.2 File: `docs/DATASETS.md` *(already drafted — verify)*

- [x] **P0.2.1** Reconcile audit numbers with doc tables; fix any drift.
- [x] **P0.2.2** Add “how to load with `uv run`” snippet.

### P0.3 File: `results/scaling_log.md` *(create stub)*

- [x] **P0.3.1** Template rows for Stages 0–14 + G-DATA / G-TOK / …

### P0.4 Manual / process (sub)

- [x] **P0.4.1** Confirm HF access to private `galsenai/*` (all three datasets loaded in the full audit on 2026-09-16).
- [x] **P0.4.2** Decide cache dir (`HF_HOME` / `UV_CACHE_DIR`) on training machine; document in README.

---

## P1 — `uv` environments, builds, dependency groups

> No `requirements.txt`-only workflow for training. **`uv` is source of truth.**

### P1.1 File: `pyproject.toml` *(create)*

- [x] **P1.1.0** Project name `galsenai-lfm25-wolof`; Python `>=3.11,<3.13` (pin after CUDA wheel check).
- [x] **P1.1.1** `[project.dependencies]` core: `pyyaml`, `pandas`, `pyarrow`, `numpy`, `tqdm`, `huggingface_hub`, `datasets`, `jinja2`.
- [x] **P1.1.2** Optional dependency groups:
  - [x] **P1.1.2.a** `train`: `torch`, `transformers>=4.57` (or `>=5` if required by LFM cards), `accelerate`, `peft`, `trl`, `bitsandbytes`, `safetensors`, `sentencepiece`, `protobuf`.
  - [x] **P1.1.2.b** `eval`: `sacrebleu`, `torch`, `transformers`, `accelerate` (share or duplicate pins).
  - [x] **P1.1.2.c** `data`: `datasets`, `pandas`, `pyarrow`, `fasttext`/`glotlid` optional later.
  - [x] **P1.1.2.d** `dev`: `ruff`, `pytest`, `ipython`.
  - [x] **P1.1.2.e** `notebook`: `jupyter`, `ipywidgets`, `matplotlib`.
- [x] **P1.1.3** `[tool.uv]` / `[tool.uv.sources]` for PyTorch CUDA index (e.g. cu128) matching cluster GPUs.
- [x] **P1.1.4** Scripts entry points (optional): `[project.scripts]` or document `uv run python scripts/...`.
- [x] **P1.1.5** Package layout: treat `common/` as local package via
  ```toml
  [tool.uv]
  package = true
  ```
  or `pythonpath` in `uv run --directory`.

### P1.2 File: `uv.lock` *(generate)*

- [x] **P1.2.1** `uv lock` on Linux+CUDA machine (or export platform markers).
- [ ] **P1.2.2** Commit `uv.lock` for reproducibility.
- [ ] **P1.2.3** Document Windows-dev vs Linux-train lock strategy if dual-platform.

### P1.3 File: `.python-version` *(create)*

- [x] **P1.3.1** Pin exact CPython (e.g. `3.12.8`).

### P1.4 File: `README.md` — uv section

- [x] **P1.4.1** Bootstrap:
  ```bash
  cd Experiments/LFM2.5-Base-1B
  uv python install
  uv sync --group train --group eval --group data
  uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
  ```
- [x] **P1.4.2** Auth: `uv run huggingface-cli login` or `HF_TOKEN` in env.
- [x] **P1.4.3** GPU check script command.

### P1.5 File: `scripts/env/00_verify_env.py` *(create)*

- [x] **P1.5.1** Assert CUDA available (or `--allow-cpu` for data-only).
- [x] **P1.5.2** Assert `transformers` can resolve `Lfm2ForCausalLM`.
- [x] **P1.5.3** Print torch CUDA capability, bf16 support, free VRAM.
- [x] **P1.5.4** Exit non-zero on failure (CI-friendly).

### P1.6 Builds / extras

- [x] **P1.6.1** Document flash-attn as **optional** extra (`uv sync --extra flash`); never hard-require.
- [ ] **P1.6.2** bitsandbytes version pin compatible with torch CUDA.
- [ ] **P1.6.3** (Optional later) Unsloth as `--extra unsloth` for notebooks only — not default path.

- [x] **[GATE G-UV]** PASS 2026-09-16 on NVIDIA H200: full group sync, CUDA env verification, 350M load, and finite tiny forward succeeded.

### P1.7 File: `.gitignore`

- [x] **P1.7.1** Ignore `.venv/`, `runs/`, `models/`, `tokenizers/`, large parquet, `*.pt`, HF cache symlinks; **keep** `uv.lock`, `pyproject.toml`.

---

## P2 — Scaffold + `common/` library

### P2.1 Directories

- [x] **P2.1.1** Create tree from IMPLEMENTATION_PLAN §2:
  - [ ] `common/`, `configs/{tokenizer,clm,sft,scaling}/`, `scripts/{clm,sft,data,family,env}/`
  - [ ] `notebooks/{clm,sft}/`, `docs/`, `results/{audit,smoke,mini,v1,v2,sft,family,scaling}/`

### P2.2 File: `common/__init__.py`

- [x] **P2.2.1** Re-export public helpers or leave empty package marker.

### P2.3 File: `common/data.py` ← port SmolLM3 `common/data.py`

- [x] **P2.3.1** `charger_splits_hub(dataset_id="galsenai/cleaned_data_for_clm")`.
- [x] **P2.3.2** `charger_splits_local(data_root)` parquet fallback (SmolLM3 names).
- [x] **P2.3.3** `paqueter(...)` + EOS assert/log (65k: id 7; configurable).
- [x] **P2.3.4** `charger_sft_messages(path_or_hub)` + `json.loads` for `meta` / tool args.
- [x] **P2.3.5** Unit-ish smoke: `uv run pytest tests/test_data_smoke.py` (tiny mocked rows).

### P2.4 File: `common/metrics.py`

- [x] **P2.4.1** Port `compute_bpb`, `compute_fertility`.
- [x] **P2.4.2** Align windowed BPB with `Evaluation/scripts/evaluate_clm.py` (offline boundary/byte tests).

### P2.5 File: `common/journal.py`

- [x] **P2.5.1** Port `ecrire`, `extraire_courbes`, `tracer`.

### P2.6 File: `common/model.py`

- [x] **P2.6.1** Port load / LoRA / full / dual-LR / freeze helpers.
- [x] **P2.6.2** `lire_tie_flag` / `ecrire_tie_flag` for `tie_embedding` ∪ `tie_word_embeddings`.
- [x] **P2.6.3** Configurable `target_modules` + `modules_to_save`.
- [x] **P2.6.4** Embedding name matcher flexible for LFM module paths.

### P2.7 File: `tests/test_imports.py`

- [x] **P2.7.1** `uv run pytest -q` imports `common.*`.

- [x] **[GATE G-COMMON]** Package imports under `uv run`; prior Hub dry-run returned non-empty train/test.

---

## P3 — Data scripts & mixtures

### P3.1 File: `scripts/data/20_audit_datasets.py` — see P0.1 (implement under uv)

- [x] **P3.1.1** `uv run python scripts/data/20_audit_datasets.py --out results/data_audit.json`
- [x] **[GATE]** = **G-DATA**

### P3.2 File: `scripts/data/21_build_sft_mixture.py` *(create)*

- [x] **P3.2.1** Read `configs/sft/mixture_v1.yaml` (caps, seeds, val %).
- [x] **P3.2.2** Stratified sample by `task` (+ `source` if possible).
- [x] **P3.2.3** Carve val = 2% from train; never touch Hub `test`.
- [x] **P3.2.4** Write `data/sft/mixture_v1/{train,val}.parquet` + `mixture_meta.json` (counts per task).
- [x] **P3.2.5** Sub: dry-run mode `--max-per-task 100` for CI.

### P3.3 File: `configs/sft/mixture_v1.yaml` *(create)*

- [x] **P3.3.1** Default: translation ≤ 35%; tool_use ~20%; boost instruction/qa/ner/intent; seed 42.
- [x] **P3.3.2** Document Afri* test caveat pointer.

### P3.4 Files: optional clean/prepare (fallback only)

- [x] **P3.4.1** `scripts/clm/00_clean_corpus.py` port (only if regenerating cleaned CLM).
- [x] **P3.4.2** `scripts/clm/01_prepare_data.py` port (only if regenerating).
- [x] **P3.4.3** Prefer Hub `cleaned_data_for_clm` — mark these scripts “regen only”.

### P3.5 File: `scripts/data/21_build_sft_mixture.py` gate

- [x] **[GATE G-MIX]** PASS 2026-09-16 on the authorized VM: 345,314 selected rows → 338,408 train / 6,906 val; every task represented; translation 34.75%; Hub test untouched.

---

## P4 — Tokenizer (65k family first)

### P4.1 File: `scripts/clm/06_probe_modules.py` *(create)*

- [x] **P4.1.1** Load Base; list Linear/Conv suffixes → `docs/lfm_module_names.txt`.
- [x] **P4.1.2** Propose CPT vs SFT `target_modules`.
- [ ] **P4.1.3** Run for 350M and 1.2B; note diffs.
- [x] **P4.1.4** `uv run python scripts/clm/06_probe_modules.py --base-id LiquidAI/LFM2.5-350M-Base` (H200 smoke, 2026-09-16).

### P4.2 File: `scripts/clm/02_build_tokenizer.py` *(create, LFM append-only)*

- [x] **P4.2.1** Port candidate selection; print `ignore_merges` / tokenizer model type.
- [x] **P4.2.2** Append-only construction (no special reindex for 65k).
- [x] **P4.2.3** Validations: monotonicity, round-trip, FR/EN/code.
- [x] **P4.2.4** Mean-pool init; **no** norm recalibration; preserve eos/bos/pad ids.
- [x] **P4.2.5** Write `tokenizers/lfm25_wolof_ext_<N>/`, `models/lfm25-wolof-ext<N>-init/`, `extension_meta.json`.
- [x] **P4.2.6** CLI + YAML: `--config configs/tokenizer-ext4000-65k.yaml`.

### P4.3 File: `configs/tokenizer-ext4000-65k.yaml` *(create)*

- [x] **P4.3.1** `base_id: LiquidAI/LFM2.5-1.2B-Base` (or 350M for cheap build); `n_tokens: 4000`; seed 42.

### P4.4 Sub: fertility / ablation configs

- [x] **P4.4.1** `configs/tokenizer-ext2000-65k.yaml`
- [x] **P4.4.2** `configs/tokenizer-ext8000-65k.yaml`

### P4.5 File: `configs/tokenizer-ext4000-128k.yaml` *(later; after 65k hero)*

- [x] **P4.5.1** Inspect 2.6B special id layout before coding append vs reindex. → [`docs/lfm_128k_specials.md`](./docs/lfm_128k_specials.md) (pad=124893, bos=124894, eos=124900).
- [x] **P4.5.2** `--family 128k` append path with blocking contiguous-layout/special-ID validation; real T-128 model proof remains gated.

### P4.6 Resize helper for other 65k sizes

- [x] **P4.6.1** File: `scripts/clm/02b_resize_ext_to_size.py` — preserve append-only IDs/specials; mean-pool target-size rows or copy compatible source rows; write resize metadata; no norm recalibration.

- [ ] **[GATE G-TOK-65K]** Validations pass; fertility drop recorded; norm ratio ~0.5; eos remains 7 on 65k; `extension_meta.json` committed or archived.
  - GPU micro-smoke 2026-09-16: temporary ext32 preserved eos=7 and passed append-only/round-trip checks, but showed no fertility drop on the tiny sample and norm ratio 0.714; ext4000 gate remains open.

---

## P5 — CLM scripts & configs (code)

### P5.1 File: `scripts/clm/03_train.py`

- [x] **P5.1.1** Port Trainer dual-LR, smoke, phase A/B, journal (SmolLM3).
- [x] **P5.1.2** Resolve `LiquidAI/` Hub ids + local `output_root`.
- [x] **P5.1.3** `--smoke` → 10 steps; write under `results/smoke/`.
- [x] **P5.1.4** Journal: `size`, `tokenizer_family`, `stage`, EOS id, `uv`/pkg versions.

### P5.2 File: `scripts/clm/04_evaluate.py`

- [x] **P5.2.1** Recalc BPB all runs; CSV + plot under `results/`.
- [x] **P5.2.2** No hardcoded machine paths.

### P5.3 File: `scripts/clm/05_merge_adapter.py`

- [x] **P5.3.1** Merge bf16; tie-flag fix; BPB Δ ≤ 0.002 gate (exit non-zero).

### P5.4 Configs — scaling

- [x] **P5.4.1** `configs/scaling/smoke_350m_default_lora.yaml`
- [x] **P5.4.2** `configs/scaling/smoke_350m_ext_lora.yaml`
- [x] **P5.4.3** `configs/scaling/mini_350m_ext_lora.yaml` (250–500 steps)
- [x] **P5.4.4** `configs/scaling/smoke_1.2b_ext_lora.yaml`
- [x] **P5.4.5** `configs/scaling/mini_1.2b_ext_lora.yaml`
- [x] **P5.4.6** `configs/scaling/smoke_230m_ext_lora.yaml`
- [x] **P5.4.7** `configs/scaling/smoke_2.6b_ext_lora.yaml` (gated placeholder)
- [x] **P5.4.8** `configs/scaling/smoke_8b_moe_lora.yaml` (gated placeholder)

### P5.5 Configs — size grids

- [x] **P5.5.1** `configs/clm/350m/v1_{default,ext4000}_{lora,full}.yaml`
- [x] **P5.5.2** `configs/clm/350m/v2_*.yaml` (patience, lr_embeddings)
- [x] **P5.5.3** `configs/clm/1.2b/v1_*.yaml` + `v2_*.yaml`
- [x] **P5.5.4** `configs/clm/230m/v2_ext_lora.yaml` (LoRA only)
- [x] **P5.5.5** `configs/clm/2.6b/...` schema-valid gated placeholder
- [x] **P5.5.6** `configs/clm/8b/...` schema-valid gated placeholder

### P5.6 File: `scripts/family/30_run_family.py`

- [x] **P5.6.1** Config-driven size → ordered configs; invoke train via subprocess `uv run`; dry-run and fail-fast.

- [ ] **[GATE G-SCRIPTS-CLM]** `uv run python scripts/clm/03_train.py --config configs/scaling/smoke_350m_default_lora.yaml --smoke` completes without NaN.
  - GPU micro-smoke 2026-09-16: temporary 2-step default LoRA and 1-step ext32 LoRA completed with finite losses; canonical 10-step gate remains open.

---

## P6 — Scaling tests (CPT) — detailed gates

Log every gate in `results/scaling_log.md`.

### Stage 0 — Env *(depends G-UV)*

- [x] **S0.1** `uv run python scripts/env/00_verify_env.py` (NVIDIA H200, CUDA 12.8, bf16; 2026-09-16).
- [ ] **S0.2** Forward pass 350M-Base + 1.2B-Base (1 batch random).
- [ ] **S0.3** `06_probe_modules` for both.
- [ ] **[GATE S0]** Loads OK; module lists saved.

### Stage 1 — Data *(depends G-DATA)*

- [ ] **S1.1** Re-run audit if data version bumped.
- [ ] **[GATE S1]** = G-DATA fresh.

### Stage 2 — Tokenizer *(depends G-TOK-65K)*

- [ ] **S2.1** Build ext4000 on chosen base.
- [ ] **S2.2** Fertility table before/after on CLM test.
- [ ] **[GATE S2]** Validations green; meta archived.

### Stage 3 — Smoke CPT 350M

- [ ] **S3.1** Smoke default-LoRA 10 steps.
- [ ] **S3.2** Smoke ext-LoRA 10 steps (`modules_to_save` on).
- [ ] **S3.3** Confirm journals + finite BPB.
- [ ] **[GATE S3]** Both smokes green; no OOM (adjust micro_bs if needed).

### Stage 4 — Mini CPT 350M ext-LoRA

- [ ] **S4.1** 250–500 steps; plot train/val loss.
- [ ] **S4.2** Compare BPB vs step-0 and vs stock base (quick 200–500 texts).
- [ ] **[GATE S4]** BPB clearly improves vs init; if not → debug tok/LoRA before v1.

### Stage 5 — Vocab ablation 350M

- [ ] **S5.1** Mini for N=2000, 4000, 8000 (shared recipe).
- [ ] **S5.2** Pick N by post-mini BPB + fertility; default expect 4000.
- [ ] **[GATE S5]** N frozen in `extension_meta.json` + README.

### Stage 6 — v1 equal-data 350M (4 runs)

- [ ] **S6.1** default-lora, default-full, ext-lora, ext-full (1 epoch equal text).
- [ ] **S6.2** `04_evaluate.py` → `results/v1_350m_comparaison.csv`.
- [ ] **[GATE S6]** Ranking interpretable; ext-LoRA competitive or diagnosed.

### Stage 7 — 1.2B CPT

- [ ] **S7.1** Smoke 1.2B ext-LoRA.
- [ ] **S7.2** v1 best recipe from 350M (or full 4-run if budget).
- [ ] **S7.3** v2 early-stop champion.
- [ ] **[GATE S7]** 1.2B BPB ≪ stock Base on clean test; journal complete.

### Stage 8 — Merge 1.2B

- [ ] **S8.1** `05_merge_adapter.py` with tolerance 0.002.
- [ ] **S8.2** Save `models/<run>-merged/` + `fusion_meta.json`.
- [ ] **[GATE S8]** ΔBPB ≤ 0.002; tie flags correct.

---

## P7 — SFT code & scaling

### P7.1 File: `scripts/sft/10_prepare_sft.py`

- [x] **P7.1.1** Consume mixture parquet or Hub; apply LFM Instruct chat template.
- [x] **P7.1.2** Prepare assistant token spans and TRL-compatible assistant-only text.
- [x] **P7.1.3** Save formatted parquet + template hash.

### P7.2 File: `scripts/sft/11_train_sft.py`

- [x] **P7.2.1** TRL `SFTTrainer` + LoRA; config-driven.
- [x] **P7.2.2** `--smoke` path (10 steps, ≤1k rows).
- [x] **P7.2.3** Fail-fast merged-CPT start path.
- [x] **P7.2.4** `journal_sft.json` with mixture id, weights, lock hash and versions.

### P7.3 File: `scripts/sft/12_eval_instruct.py`

- [x] **P7.3.1** Val loss by `task`.
- [x] **P7.3.2** Qualitative 20 WO prompts → `results/sft/qualitative.jsonl`.
- [x] **P7.3.3** Optional chrF when reference/prediction columns exist.
- [x] **P7.3.4** Optional shared windowed CLM BPB continuity hook.

### P7.4 Configs

- [x] **P7.4.1** `configs/sft/1.2b/smoke.yaml`
- [x] **P7.4.2** `configs/sft/1.2b/mini_mixture_v1.yaml`
- [x] **P7.4.3** `configs/sft/1.2b/full_mixture_v1.yaml`
- [x] **P7.4.4** Ablation: `full_raw_unweighted.yaml`
- [x] **P7.4.5** Ablation: `from_stock_instruct.yaml`

### Stage 9 — Smoke SFT

- [ ] **S9.1** Smoke on merged 1.2B.
- [ ] **[GATE S9]** Loss↓; ≥15/20 qualitative replies stay Wolof-primary.

### Stage 10 — Mini SFT

- [ ] **S10.1** 50k–100k mixture.
- [ ] **[GATE S10]** Val `instruction`/`qa` improve vs CPT-only chat baseline.

### Stage 11 — Full SFT

- [ ] **S11.1** Full mixture_v1.
- [ ] **S11.2** Save adapter/merged instruct checkpoint.
- [ ] **[GATE S11]** Publish candidate; Evaluation JSON draft; human spot-check scheduled.

---

## P8 — Synthetic data (gated)

### P8.1 File: `scripts/data/22_synth_reasoning.py`

- [x] **P8.1.1** Offline CoT pilot filtering (answer/steps/degeneracy; generation and LID remain external).
- [x] **P8.1.2** Machine-readable 50-sample G0 review manifest generation.
- [ ] **[GATE SYN-G0]** Human OK.
- [ ] **[GATE SYN-G1]** Mini-SFT +5k vs control improves val qa/instruction.
- [ ] **P8.1.3** Scale to 20k–50k only if G1 pass.

### P8.2 File: `scripts/data/22_synth_longctx.py`

- [x] **P8.2.1** Deterministic 4k/8k/16k/32k word-pack builder.
- [x] **P8.2.2** Needle task input generator.
- [ ] **[GATE SYN-G4]** Needle@8k on 2.6B mid-train beats short-only (Stage 13).

### P8.3 File: `scripts/data/22_synth_ortho.py`

- [x] **P8.3.1** Diacritic pair builder (mixture execution remains gated).

### P8.4 File: `scripts/data/22_synth_prefs.py`

- [x] **P8.4.1** Preference validator + real config-driven `14_train_dpo.py`; generation/run remains gated after S11.
- [ ] **[GATE SYN-G3]** CLM BPB not degraded >2% after pref step.

### P8.5 File: `scripts/data/22_synth_tool_wo.py`

- [x] **P8.5.1** Localized tool-use schema validation (translation generation remains external).

### P8.6 File: `docs/SYNTHETIC_DATA.md`

- [x] **P8.6.1** Document actual offline script CLIs.

- [ ] **[GATE]** Synthetic ≤ 30% of SFT tokens in v1 hero mix.

---

## P9 — Evaluation harness (repo `Evaluation/`)

### P9.1 Configs *(create)*

- [x] **P9.1.1** `Evaluation/configs/clm/lfm_1.2b_base.yaml`
- [x] **P9.1.2** `Evaluation/configs/clm/lfm_1.2b_clm.yaml`
- [x] **P9.1.3** `Evaluation/configs/clm/lfm_1.2b_instruct.yaml`
- [x] **P9.1.4** Per-size base/CLM/instruct YAMLs for 230M–8B; unpublished variants explicitly marked non-runnable placeholders.

### P9.2 Runs

- [ ] **P9.2.1** Download test via `uv run` + existing `download_dataset.py` (or CSV already present).
- [ ] **P9.2.2** `uv run python Evaluation/scripts/evaluate_clm.py --config ...`
- [ ] **P9.2.3** Commit/write `Evaluation/metrics/clm/lfm_*.json`.

### P9.3 Optional: env group for Evaluation root

- [x] **P9.3.1** Added `Evaluation/pyproject.toml` and documented `uv --directory` runner.

- [ ] **[GATE G-EVAL]** Base / CPT / Instruct JSONs exist for hero; BPB comparable protocol (verbalizer on, same seed).

---

## P10 — Family rollout (per MODEL_PLANS)

### P10.1 230M

- [ ] **P10.1.1** Resize 65k ext → 230M init (`02b_resize…`).
- [ ] **P10.1.2** Smoke + mini + short v2 LoRA CPT.
- [ ] **P10.1.3** Mini/full SFT (lighter).
- [ ] **P10.1.4** Evaluation YAML + metrics.
- [ ] **[GATE F-230]** Smoke+mini green; Hub draft card.

### P10.2 350M publish

- [ ] **P10.2.1** Promote best v2 CPT + SFT (from ablation host).
- [ ] **P10.2.2** Evaluation metrics.
- [ ] **[GATE F-350]** Published alongside hero narrative.

### P10.3 2.6B (128k)

- [ ] **P10.3.1** Tokenizer family 128k — inspect specials → build ext.
- [ ] **P10.3.2** **[GATE T-128]** Validations on 128k.
- [ ] **P10.3.3** Smoke CPT short seq.
- [ ] **P10.3.4** Short CPT until BPB gate.
- [ ] **P10.3.5** Longctx mid-train curriculum (4k→16k) with synth packs.
- [ ] **P10.3.6** Needle eval @ 8k/16k.
- [ ] **P10.3.7** SFT mixture + reasoning.
- [ ] **[GATE F-26]** Short CPT + needle + Evaluation JSON.

### P10.4 8B-A1B MoE

- [ ] **P10.4.1** Probe expert module names; PEFT plan doc snippet in `docs/lfm_moe_lora.md`.
- [ ] **P10.4.2** Smoke MoE LoRA (tiny batch).
- [ ] **P10.4.3** Mini CPT → SFT tool-heavy.
- [ ] **P10.4.4** Tool schema validity metric.
- [ ] **[GATE F-8B]** Stable train; tool eval; Evaluation JSON.

### P10.5 File: `scripts/family/30_run_family.py`

- [x] **P10.5.1** Orchestrate manifest-defined size→configs with `uv run`, `--dry-run`, and fail-fast execution.

---

## P11 — Publication

### P11.1 Hub

- [ ] **P11.1.1** Push hero CPT/instruct under `galsenai/LFM2.5-1.2B-Base_wolof`.
- [ ] **P11.1.2** Subfolders `v1/`, `v2/`, `instruct/v1/`.
- [ ] **P11.1.3** Model cards from PUBLICATION.md + leakage caveats.
- [ ] **P11.1.4** Collection “Wolof LFM2.5”.

### P11.2 Repo docs

- [ ] **P11.2.1** Fill FAMILY_EVAL tables with real numbers.
- [ ] **P11.2.2** Blog outline → draft.
- [ ] **P11.2.3** Cite `arXiv:2511.23404` + SmolLM3 Wolof.

### P11.3 Human

- [ ] **P11.3.1** Native-speaker review panel (process).
- [ ] **[GATE PUB]** J-gates + G-EVAL + human OK + license check.

---

## P12 — Notebooks (optional, non-blocking)

### P12.1 Files

- [x] **P12.1.1** `notebooks/clm/tokenizer_probe.ipynb` — executed successfully on the authorized VM (2026-09-16).
- [x] **P12.1.2** `notebooks/sft/sft_smoke.ipynb` — executed successfully on the authorized VM (2026-09-16).
- [x] **P12.1.3** `notebooks/README.md` — scripts are SoT

---

## Cross-cutting quality / footgun checklist (run continuously)

- [ ] **X.1** Never CPT on raw centralized without cleaned pipeline.
- [ ] **X.2** Never LoRA-ext without `modules_to_save`.
- [ ] **X.3** Never recalibrate embedding norms after mean-pool.
- [ ] **X.4** Never compare PPL across tokenizers; use BPB.
- [ ] **X.5** Never promote stage if previous GATE red.
- [ ] **X.6** Always `uv run` for train/eval (reproducible lock).
- [ ] **X.7** Log `uv.lock` hash / `transformers` version in every `journal.json`.

## P13 — Production orchestration and Trackio

- [x] **P13.1** Durable atomic stage state, PID lock, dependency gates, resource
  checks, per-stage logs, command/environment hashes, `--resume`, `--dry-run`,
  and `--only-size`.
- [x] **P13.2** Private-only Hub write-scope smoke and verified publisher with
  secret scan, checksums, remote listing/readback, revisions, and publication
  manifest.
- [x] **P13.3** Shared secret-safe Trackio adapter, deterministic names/groups,
  structured alerts, durable degradation state, and Trainer/TRL integration.
- [x] **P13.4** Trackio added to `uv` dependencies and lock; offline mocked
  tracking/state-machine tests implemented.
- [x] **P13.5** Production manifest requires private Trackio Bucket persistence
  before GPU training and exports only `TRACKIO_SPACE_ID`; live dashboard
  availability is tracked separately.
- [x] **[GATE G-TRACKIO-REMOTE]** Private Space created; tiny run synced and
  queried successfully on the authorized VM.
  - Passed 2026-09-16: the authenticated Bucket was made private, the exact
    smoke metric was downloaded and parsed, SHA-256 and object revision were
    confirmed, and the durable local database query passed. The live Space is
    separately degraded as `paused_quota`; local-only tracking cannot satisfy
    this gate.
- [x] **[GATE G-PRODUCTION-QUEUE]** Detached queue launched only after
  G-TRACKIO-REMOTE and Hub write-scope verification pass.
  - Launched 2026-09-16 with PID/session `1061998`; lock/state are active and
    the queue progressed through preflight, environment, data mixture, and the
    first model probe. Final training and quality gates remain pending.
  - Monitor intervention 2026-09-16: fixed padded-vocabulary semantic-prefix
    validation in `02b_resize_ext_to_size.py`, passed 31 tests/Ruff/compileall
    locally and remotely, and resumed from the failed `resize_350m_ext4000`
    checkpoint. The retry and all resize stages completed. The next gate
    rejected stale generic LoRA targets; all three 350M recipes were corrected
    to the probe-approved `q_proj`, `k_proj`, and `v_proj` set and locked by a
    regression test. After 32 tests/Ruff/compileall and dry-run validation, the
    queue resumed with 13 stages complete and the tokenizer-ablation training
    active. No loss/BPB or publication manifest existed at the observation.
  - Monitor 2026-09-16 20:43 UTC: still healthy at 13 complete stages;
    ext2000 tokenizer-ablation reached step 39/400 with finite loss 4.298 and
    baseline BPB 3.6607. Private Bucket telemetry was fresh, no red gate or
    publication change appeared, and no intervention was needed.
  - Monitor 2026-09-16 20:58 UTC: ext2000 completed 400/400 with train loss
    2.584, final eval loss 2.315, and BPB 3.6607→1.5845 (56.7%); ext4000 was
    roughly halfway with latest loss 2.469. The top-level ablation stage
    remained active, private Bucket data was fresh, and no intervention or
    publication change was needed.
  - Monitor 2026-09-16 21:13 UTC: ext4000 completed 400/400 with train loss
    2.587, final eval loss 2.312, and BPB 3.7926→1.5790 (58.4%); ext8000
    reached approximately 200/400 with latest loss 2.390. Queue and private
    Bucket persistence remained healthy with no red gate, publication, or
    intervention.
  - Monitor 2026-09-16 21:28 UTC: tokenizer ablation completed; ext8000 was the
    metric winner with final eval loss 2.306 and BPB 3.9775→1.5612 (60.8%).
    `cpt_350m` started and completed both 10-step smoke runs before advancing to
    its 1,500-step default-tokenizer run. Fourteen top-level stages were
    complete; no red gate, publication, or intervention appeared.
  - Monitor 2026-09-16 21:43 UTC: `v1_350m_default_lora` reached approximately
    600/1500 with latest train loss 3.177 and eval loss 3.245 at step 500.
    Current Bucket and durable local Trackio state were verified; the queue
    remained healthy with no red gate, publication, or intervention.
  - Monitor 2026-09-16 21:58 UTC: `v1_350m_default_lora` advanced to
    approximately 1300/1500 with latest train/eval losses 3.023/2.961 and
    checkpoints at steps 500 and 1000. The queue and durable tracking state
    remained healthy; no red gate, publication, or intervention appeared.
  - Monitor 2026-09-16 22:13 UTC: `v1_350m_default_lora` completed with final
    train/eval losses 2.9237/2.9433 and BPB 3.3941→2.1247. The queue advanced to
    `v1_350m_default_full`, observed at step 505/1500. Fixed an idempotency-only
    Trackio finalization metadata defect after confirming private-Bucket
    persistence; 10 focused tests and Ruff passed locally and remotely.
  - Monitor 2026-09-16 22:28 UTC: `v1_350m_default_full` advanced to
    approximately 1300/1500 with latest train/eval losses 2.120/2.136 and
    checkpoints through step 1000. Queue, private-Bucket persistence, and local
    durable tracking remained healthy; no red gate, publication, or intervention
    appeared.
  - Monitor 2026-09-16 22:43 UTC: `v1_350m_default_full` completed with final
    train/eval losses 2.1086/2.1354 and BPB 3.3436→1.4400. The queue advanced to
    `v1_350m_ext4000_lora`, observed at step 302/1500 with latest train/eval
    losses 2.321/2.421. Tracking remained current; no red gate, publication, or
    intervention appeared.
  - Monitor 2026-09-16 22:58 UTC: `v1_350m_ext4000_lora` reached 989/1500 with
    latest emitted train/eval losses 2.107/2.123 and `checkpoint-500` durable.
    Queue and tracking remained healthy; no red gate, publication, or
    intervention appeared.
  - Monitor 2026-09-16 23:13 UTC: `v1_350m_ext4000_lora` completed with final
    train/eval losses 2.0261/2.0483 and BPB 3.7926→1.4784. Its Trackio state
    finished as `synced`. The queue advanced to `v1_350m_ext4000_full`, which
    was loading/preparing data; no red gate, publication, or intervention
    appeared.
  - Monitor 2026-09-16 23:28 UTC: `v1_350m_ext4000_full` reached 720/1500 with
    latest emitted train/eval losses 2.102/2.118 and `checkpoint-500` durable.
    Queue and tracking remained healthy; no red gate, publication, or
    intervention appeared.
  - Monitor 2026-09-16 23:43 UTC: `v1_350m_ext4000_full` completed with final
    train/eval losses 2.0366/2.0626 and BPB 3.7629→1.3869. Trackio finalized as
    `synced`; the queue advanced to `v2_350m_default_lora`, which was
    loading/preparing data. No red gate, publication, or intervention appeared.
  - Monitor 2026-09-16 23:58 UTC: `v2_350m_default_lora` reached 603/3000 with
    latest train/eval losses 3.455/3.449 and checkpoints through step 500.
    Queue and private-Bucket tracking remained healthy; no red gate,
    publication, or intervention appeared.
  - Monitor 2026-09-17 00:13 UTC: `v2_350m_default_lora` reached 1240/3000 with
    latest train/eval losses 3.174/3.134 and checkpoints through step 1200.
    Queue and tracking remained healthy; no red gate, publication, or
    intervention appeared.
  - Monitor 2026-09-17 00:28 UTC: `v2_350m_default_lora` reached 1930/3000 with
    latest train/eval losses 3.005/3.000 and checkpoints through step 1900.
    Queue and tracking remained healthy; no red gate, publication, or
    intervention appeared.
  - Monitor 2026-09-17 00:43 UTC: `v2_350m_default_lora` reached 2601/3000 with
    latest train/eval losses 2.948/2.956 and checkpoints through step 2600.
    Queue and tracking remained healthy; no red gate, publication, or
    intervention appeared.
  - Resumed monitor 2026-09-17 02:39 UTC: `v2_350m_default_full` completed by
    configured early stopping at 2500/3000 with train/eval losses 2.3400/2.3403
    and BPB 3.3436→1.6338. The queue advanced to `v2_350m_ext4000_lora`,
    observed at 2462/3000 with train/eval losses 2.020/2.017 and checkpoints
    through step 2400. No red gate, publication, or recovery appeared.
  - Resumed monitor 2026-09-17 04:49 UTC:
    `v2_350m_ext4000_lora` completed with losses 2.0064/2.0062 and BPB
    3.7926→1.3622; `v2_350m_ext4000_full` early-stopped at 2500/3000 with
    losses 2.2518/2.2518 and BPB 3.7629→1.5667. `cpt_350m` completed.
    Recovered `merge_cpt_350m` after adding nested journal-ID adapter
    resolution; 23 local and 13 remote focused tests plus Ruff passed. Merge
    attempt 2 was alive, with no gate bypass or publication yet.
  - Recovery follow-through 2026-09-17 05:02 UTC: merge passed, SFT input
    preparation passed, and private CPT publication/readback completed at
    `Tonic/LFM2.5-350M-Wolof-CPT` revision
    `d16bb3dbd1938a16fc0026aab29332bdb28e6a92`. The first SFT launch then
    exposed zero-row prepared parquet caused by array-like message columns and
    unstable template-prefix masking. Preparation now normalizes those columns,
    uses native assistant masks, and fails closed on empty output. Fourteen
    local/remote tests and a 1,000-row-per-split remote smoke passed with zero
    rejects. The queue resumed full preparation as attempt 2; no red gate was
    bypassed.
  - Monitor 2026-09-17 05:14 UTC: `prepare_sft_65k` attempt 2 completed with
    338,408 train / 6,906 validation rows, zero rejects, and explicit
    assistant-only labels. `sft_350m` attempt 2 then reached 29/8,000; no loss,
    eval, or checkpoint had been emitted yet. Queue/SFT processes were alive,
    with 19 complete stages, one running stage, and no failure. The private
    Trackio Bucket readback returned 214 objects and fresh metadata at
    05:10:52 UTC; the 350M CPT publication remained verified with no new
    publication. H200 usage was 12,061/143,771 MiB and disk free was about
    1,045 GiB. No recovery or gate bypass was needed; exactly one recurring
    15-minute monitor was armed. Fully completed/private-published sizes: 0/4.
  - Monitor 2026-09-17 05:27 UTC: `sft_350m` advanced to 200/8,000; latest
    train loss was 2.273 (3.185 at step 25) with mean token accuracy 0.5592.
    The first scheduled eval/checkpoint had not emitted yet. Queue and SFT
    processes remained alive with 19 complete stages, one active stage, and no
    failure. H200 usage was 12,693/143,771 MiB, instantaneous utilization 2%,
    and disk free 1,044.52 GiB. Private Trackio Bucket readback still passed
    with 214 objects / 6,710,545 bytes; Bucket metadata remained at 05:10:52
    UTC while local run tracking was active. The verified 350M CPT publication
    was unchanged; no recovery or gate bypass was needed. Fully
    completed/private-published sizes remain 0/4, so the sole monitor remains
    armed.
  - Monitor 2026-09-17 05:42 UTC: `sft_350m` advanced to 411/8,000 with
    train loss 2.026 at step 400 and eval loss 2.160 at step 200; checkpoints
    200 and 400 were durable. Queue/SFT processes remained alive with 19
    complete stages, one active stage, and no failure. H200 usage was
    12,875/143,771 MiB, instantaneous utilization 7%, and disk free
    1,042.93 GiB. Private Trackio Bucket readback still passed with 214
    objects / 6,710,545 bytes; Bucket metadata remained at 05:10:52 UTC while
    local tracking stayed active. The verified private 350M CPT publication
    was unchanged. No recovery or gate bypass was needed; completed/private-
    published sizes remain 0/4 and the sole monitor remains armed.
  - Monitor 2026-09-17 06:03 UTC: `sft_350m` advanced to 1,010/8,000 with
    train/eval losses 1.907/1.918 at step 1,000; retained checkpoints were
    800 and 1,000. Queue/SFT processes remained alive with 19 complete stages,
    one active stage, and no failure. H200 usage was 13,077/143,771 MiB,
    instantaneous utilization 16%, and disk free 1,042.92 GiB. Authenticated
    private Trackio Bucket readback still passed with 214 objects / 6,710,545
    bytes; Bucket metadata remained at 05:10:52 UTC while local tracking stayed
    active. The verified private 350M CPT publication was unchanged. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 0/4 and the sole monitor remains armed.
  - Monitor 2026-09-17 06:18 UTC: `sft_350m` advanced to 1,610/8,000 with
    train/eval losses 1.940/1.827 at step 1,600; retained checkpoints were
    1,400 and 1,600. Queue/SFT processes remained alive with 19 complete
    stages, one active stage, and no failure. H200 usage was
    13,169/143,771 MiB, instantaneous utilization 0%, and disk free
    1,042.92 GiB. Authenticated private Trackio Bucket readback still passed
    with 214 objects / 6,710,545 bytes; Bucket metadata remained at 05:10:52
    UTC while local tracking stayed active. The verified private 350M CPT
    publication was unchanged. No recovery or gate bypass was needed;
    completed/private-published sizes remain 0/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 06:33 UTC: `sft_350m` advanced to 2,466/8,000 with
    train loss 1.676 at step 2,400 and latest emitted eval loss 1.763 at step
    2,200; retained checkpoints were 2,200 and 2,400. Queue/SFT processes
    remained alive with 19 complete stages, one active stage, and no failure.
    H200 usage was 13,239/143,771 MiB, instantaneous utilization 0%, and disk
    free 1,042.92 GiB. Authenticated private Trackio Bucket readback still
    passed with 214 objects / 6,710,545 bytes; Bucket metadata remained at
    05:10:52 UTC while local tracking stayed active. The verified private 350M
    CPT publication was unchanged. No recovery or gate bypass was needed;
    completed/private-published sizes remain 0/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 06:48 UTC: `sft_350m` advanced to 3,534/8,000 with
    train loss 1.724 at step 3,400 and latest emitted eval loss 1.712 at step
    3,200; retained checkpoints were 3,200 and 3,400. Queue/SFT processes
    remained alive with 19 complete stages, one active stage, and no failure.
    H200 usage was 13,287/143,771 MiB, instantaneous utilization 0%, and disk
    free 1,042.92 GiB. Authenticated private Trackio Bucket readback still
    passed with 214 objects / 6,710,545 bytes; Bucket metadata remained at
    05:10:52 UTC while local tracking stayed active. The verified private 350M
    CPT publication was unchanged. No recovery or gate bypass was needed;
    completed/private-published sizes remain 0/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 07:03 UTC: `sft_350m` advanced to 4,655/8,000 with
    train loss 1.688 at step 4,600 and latest emitted eval loss 1.677 at step
    4,400; retained checkpoints were 4,400 and 4,600. Queue/SFT processes
    remained alive with 19 complete stages, one active stage, and no failure.
    H200 usage was 13,323/143,771 MiB, instantaneous utilization 14%, and disk
    free 1,042.91 GiB. Authenticated private Trackio Bucket readback still
    passed with 214 objects / 6,710,545 bytes; Bucket metadata remained at
    05:10:52 UTC while local tracking stayed active. The verified private 350M
    CPT publication was unchanged. No recovery or gate bypass was needed;
    completed/private-published sizes remain 0/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 07:18 UTC: `sft_350m` advanced to 5,962/8,000 with
    train loss 1.670 at step 5,800 and latest emitted eval loss 1.657 at step
    5,600; retained checkpoints were 5,600 and 5,800. Queue/SFT processes
    remained alive with 19 complete stages, one active stage, and no failure.
    H200 usage was 13,337/143,771 MiB, instantaneous utilization 13%, and disk
    free 1,042.91 GiB. Authenticated private Trackio Bucket readback still
    passed with 214 objects / 6,710,545 bytes; Bucket metadata remained at
    05:10:52 UTC while local tracking stayed active. The verified private 350M
    CPT publication was unchanged. No recovery or gate bypass was needed;
    completed/private-published sizes remain 0/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 07:33 UTC: `sft_350m` advanced to 7,053/8,000 with
    train loss 1.638 at step 7,000 and latest emitted eval loss 1.646 at step
    6,800; retained checkpoints were 6,800 and 7,000. Queue/SFT processes
    remained alive with 19 complete stages, one active stage, and no failure.
    H200 usage was 25,678/143,771 MiB, instantaneous utilization 99%, and disk
    free 1,042.88 GiB. Authenticated private Trackio Bucket readback still
    passed with 214 objects / 6,710,545 bytes; Bucket metadata remained at
    05:10:52 UTC while local tracking stayed active. The verified private 350M
    CPT publication was unchanged. No recovery or gate bypass was needed;
    completed/private-published sizes remain 0/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 07:48 UTC: `sft_350m` completed 8,000/8,000 with
    aggregate train loss 1.7383, final-batch loss 1.744, final eval loss
    1.642, and checkpoint 8,000. `merge_sft_350m` then passed its BPB gate:
    1.609118→1.609487, delta 0.000369 within tolerance 0.002.
    `eval_sft_350m` became active and was loading the merged artifact; final
    evaluation metrics remained pending. Queue/eval processes were alive with
    21 complete stages, one active stage, and no failure. H200 usage was
    20,419/143,771 MiB, utilization 39%, and disk free 1,041.80 GiB. Private
    Trackio Bucket readback refreshed to 221 objects / 9,090,564 bytes at
    07:47:55 UTC. The verified private 350M CPT publication was unchanged. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 0/4 and the sole monitor remains armed.
  - Monitor/recovery 2026-09-17 08:08 UTC: final 350M evaluation completed;
    per-task mean losses were tool-use 2.5363, NER 4.6138, instruction 2.9086,
    translation 5.8490, intent 4.5308, classification 6.0467, and QA 4.5695.
    `Tonic/LFM2.5-350M-Wolof-Instruct` was privately published and
    authenticated readback verified revision
    `0142968c16f430230aca095077a794b489d76600`; CPT readback remained valid.
    The next gate rejected four stale 1.2B LoRA suffixes absent from its probe.
    The recipe now uses probe-approved `q_proj`, `k_proj`, and `v_proj`; five
    local and remote tests, Ruff, and the exact validator passed. The first
    resume lacked the non-interactive `uv` path and failed closed; relaunching
    with the corrected environment passed `validate_1_2b_targets` on attempt 3
    and advanced to active `cpt_1_2b` smoke training. Queue/training processes
    were alive with 25 complete stages, one active stage, and no failure. H200
    usage was 19,556/143,771 MiB, utilization 43%, and disk free 1,041.80 GiB.
    Private Trackio Bucket readback refreshed to 231 objects / 9,127,364 bytes
    at 08:07:56 UTC. No gate was bypassed; completed/private-published sizes
    are now 1/4 and the sole monitor remains armed.
  - Monitor 2026-09-17 08:18 UTC: 1.2B CPT smoke completed 10/10 with final
    train/eval losses 4.4313/4.3361, mean train loss 4.8159, and BPB
    3.6259→2.6830 (26.0% gain); peak allocated VRAM was 8,413 MiB. The
    400-step 1.2B mini run then started and was loading. Queue/training
    processes remained alive with 25 complete top-level stages, one active
    stage, and no failure. H200 usage was 9,505/143,771 MiB, instantaneous
    utilization 0% during loading, and disk free 1,038.19 GiB. Private Trackio
    Bucket readback refreshed to 234 objects / 9,224,377 bytes at 08:14:49
    UTC. Both private 350M Hub readbacks remained unchanged and verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the sole monitor remains armed.
  - Monitor 2026-09-17 08:33 UTC: the 1.2B mini run advanced to 319/400;
    latest train loss was 2.121 at step 300, eval loss 2.206 at step 200, and
    checkpoint 200 was durable. Queue/training processes remained alive with
    25 complete top-level stages, one active stage, and no failure. H200 usage
    was 25,483/143,771 MiB, utilization 43%, and disk free 1,036.08 GiB.
    Private Trackio Bucket readback returned 235 objects / 9,293,726 bytes,
    updated at 08:19:46 UTC. Both private 350M Hub readbacks remained verified.
    No recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the sole monitor remains armed.
  - Monitor 2026-09-17 08:48 UTC: the 1.2B mini completed 400/400 with final
    train/eval losses 2.0740/2.0955, mean train loss 2.3861, and BPB
    3.6259→1.3716 (62.17% gain); checkpoint 400 was durable. The queue advanced
    to `v1_1.2b_default_lora`, observed at 166/1,500 with train loss 3.880 at
    step 150; eval and checkpoint were pending. Queue/training processes
    remained alive with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 22,821/143,771 MiB, utilization 59%, and disk free
    1,030.73 GiB. Private Trackio Bucket readback returned 238 objects /
    9,613,673 bytes, updated at 08:42:09 UTC. Both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed;
    completed/private-published sizes remain 1/4 and the sole monitor remains
    armed.
  - Monitor 2026-09-17 09:03 UTC: `v1_1.2b_default_lora` advanced to
    610/1,500 with train loss 2.926 at step 600, eval loss 2.937 at step 500,
    and checkpoint 500. Queue/training processes remained alive with 25
    complete top-level stages, one active stage, and no failure. H200 usage
    was 22,821/143,771 MiB, utilization 47%, and disk free 1,030.71 GiB.
    Private Trackio Bucket readback remained at 238 objects / 9,613,673 bytes,
    updated at 08:42:09 UTC, while local run tracking stayed active. Both
    private 350M Hub readbacks remained verified. No recovery or gate bypass
    was needed; completed/private-published sizes remain 1/4 and the sole
    monitor remains armed.
  - Monitor 2026-09-17 09:18 UTC: `v1_1.2b_default_lora` advanced to
    971/1,500 with train loss 2.812 at step 900, eval loss 2.809 at step 800,
    and checkpoint 500. Queue/training processes remained alive with 25
    complete top-level stages, one active stage, and no failure. H200 usage
    was 38,724/143,771 MiB at 100% utilization; disk free remained 1,030.71
    GiB. The private Trackio Bucket remained at 238 objects / 9,613,673 bytes,
    updated at 08:42:09 UTC, while local run tracking stayed active. Both
    private 350M Hub readbacks remained verified. No recovery or gate bypass
    was needed; completed/private-published sizes remain 1/4 and the sole
    monitor remains armed.
  - Monitor 2026-09-17 09:33 UTC: `v1_1.2b_default_lora` advanced to
    1,333/1,500 with train loss 2.739 at step 1,300, eval loss 2.749 at step
    1,200, and checkpoint 1,000. The run remains active, so final BPB and the
    next family stage are not yet available. Queue/training processes remained
    alive with 25 complete top-level stages, one active stage, and no failure.
    H200 usage was 22,821/143,771 MiB at a 4% instantaneous sample; disk free
    was 1,030.69 GiB. The private Trackio Bucket remained at 238 objects /
    9,613,673 bytes, updated at 08:42:09 UTC, while local run tracking stayed
    active. Both private 350M Hub readbacks remained verified. No recovery or
    gate bypass was needed; completed/private-published sizes remain 1/4 and
    the sole monitor remains armed.
  - Monitor 2026-09-17 09:48 UTC: `v1_1.2b_default_lora` completed
    1,500/1,500 with aggregate train loss 3.0093499, final-batch loss
    2.7180896, final eval 2.7415144, and BPB 3.2216063 -> 2.0560082 (36.18%
    gain); checkpoint 1,500 is durable and Trackio synced. The family advanced
    cleanly to `v1_1.2b_default_full`, observed at 110/1,500 with train loss
    4.700 at step 100 and eval/checkpoint pending. Queue/training processes
    remained alive with 25 complete top-level stages, one active stage, and
    no failure. H200 usage was 21,665/143,771 MiB at 57% utilization; disk
    free was 1,030.66 GiB. Private Trackio readback advanced to 241 objects /
    10,526,120 bytes at 09:45:31 UTC. Both private 350M Hub readbacks remained
    verified. No recovery or gate bypass was needed; completed/private-
    published sizes remain 1/4 and the sole monitor remains armed.
  - Monitor 2026-09-17 10:03 UTC: `v1_1.2b_default_full` advanced to
    642/1,500 with train loss 1.954 at step 600, eval loss 1.960 at step 500,
    and checkpoint 500. Queue/training processes remained alive with 25
    complete top-level stages, one active stage, and no failure. H200 usage
    was 31,907/143,771 MiB at 58% utilization; disk free was 1,024.12 GiB.
    Private Trackio readback remained at 241 objects / 10,526,120 bytes,
    updated at 09:45:31 UTC, with the current run locally active. Both private
    350M Hub readbacks remained verified. No recovery or gate bypass was
    needed; completed/private-published sizes remain 1/4 and the sole monitor
    remains armed.
  - Monitor 2026-09-17 10:18 UTC: `v1_1.2b_default_full` advanced to
    1,200/1,500 with train loss 1.935 at step 1,200, eval loss 1.881 at step
    1,100, and checkpoint 1,000. Queue/training processes remained alive with
    25 complete top-level stages, one active stage, and no failure. H200 usage
    was 31,907/143,771 MiB at 91% utilization; disk free was 1,017.58 GiB.
    Private Trackio readback remained at 241 objects / 10,526,120 bytes,
    updated at 09:45:31 UTC, with the current run locally active. Both private
    350M Hub readbacks remained verified. No recovery or gate bypass was
    needed; completed/private-published sizes remain 1/4 and the sole monitor
    remains armed.
  - Immediate pass 2026-09-17 10:24 UTC: `v1_1.2b_default_full` advanced to
    1,451/1,500 with train loss 1.861 at step 1,400, eval loss 1.879 at step
    1,300, and checkpoint 1,000. Final BPB and the next stage remain pending.
    Queue/training processes stayed alive with 25 complete top-level stages,
    one active stage, and no failure. H200 usage was 31,907/143,771 MiB at 58%
    utilization; disk free was 1,017.58 GiB. Private Trackio readback remained
    at 241 objects / 10,526,120 bytes, updated at 09:45:31 UTC, with the
    current run locally active. Both private 350M Hub readbacks remained
    verified. No recovery or gate bypass was needed; completed/private-
    published sizes remain 1/4 and monitor PID 27132 remains armed.
  - Immediate pass 2026-09-17 10:31 UTC: `v1_1.2b_default_full` completed
    1,500/1,500 with aggregate train loss 2.0433457, final-batch loss
    1.8557368, final eval 1.8790327, and BPB 3.2045671 -> 1.2552539 (60.83%
    gain); checkpoint 1,500 is durable and Trackio synced. The family advanced
    cleanly to `v1_1.2b_ext4000_lora`, observed at 8/1,500 before its first
    metric/checkpoint. Queue/training processes stayed alive with 25 complete
    top-level stages, one active stage, and no failure. H200 usage was
    13,705/143,771 MiB at 53% utilization; disk free was 1,008.85 GiB. Private
    Trackio readback advanced to 244 objects / 11,190,497 bytes at 10:31:07
    UTC. Both private 350M Hub readbacks remained verified. No recovery or
    gate bypass was needed; completed/private-published sizes remain 1/4 and
    monitor PID 27132 remains armed.
  - Lightweight monitor 2026-09-17 10:32 UTC: the just-started
    `v1_1.2b_ext4000_lora` advanced from 8 to 41/1,500; its first scheduled
    train/eval/checkpoint metrics remain pending. Queue and all five relevant
    processes were alive, with 25 complete top-level stages, one active stage,
    and no failure. H200 usage was 14,791 MiB at 40% utilization and disk free
    remained 1,008.85 GiB. The Trackio and private-readback results from the
    immediately preceding pass remain current. No recovery was needed;
    completed/private-published sizes remain 1/4 and the monitor remains
    armed.
  - Immediate pass 2026-09-17 10:37 UTC: `v1_1.2b_ext4000_lora` advanced
    from 41 to 146/1,500 and emitted its first train metric, loss 2.499 at
    step 100; eval and checkpoint remain pending. Queue and all five relevant
    processes stayed alive, with 25 complete top-level stages, one active
    stage, and no failure. H200 usage was 25,481 MiB at 44% utilization and
    disk free remained 1,008.85 GiB. No recovery was needed; completed/private-
    published sizes remain 1/4 and the existing monitor remains armed.
  - Monitor 2026-09-17 10:48 UTC: `v1_1.2b_ext4000_lora` advanced to
    393/1,500 with train loss 2.145 at step 300 and eval loss 2.230 at step
    200; checkpoint remains pending. Queue and all five relevant processes
    stayed alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 25,481/143,771 MiB at 48% utilization and disk
    free remained 1,008.85 GiB. Private Trackio remained at 244 objects /
    11,190,497 bytes, updated at 10:31:07 UTC, and both private 350M Hub
    readbacks remained verified. No recovery or gate bypass was needed;
    completed/private-published sizes remain 1/4 and the monitor remains
    armed.
  - Monitor 2026-09-17 11:03 UTC: `v1_1.2b_ext4000_lora` advanced to
    756/1,500 with train loss 1.987 at step 700, eval loss 1.999 at step 600,
    and checkpoint 500. Queue and all five relevant processes stayed alive,
    with 25 complete top-level stages, one active stage, and no failure. H200
    usage was 25,481/143,771 MiB at 42% utilization; disk free was 1,006.75
    GiB. Private Trackio remained at 244 objects / 11,190,497 bytes, updated
    at 10:31:07 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 11:18 UTC: `v1_1.2b_ext4000_lora` advanced to
    1,107/1,500 with train loss 1.895 at step 1,100, eval loss 1.899 at step
    1,000, and checkpoint 1,000. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 25,481/143,771 MiB at 43% utilization; disk free
    was 1,004.64 GiB. Private Trackio remained at 244 objects / 11,190,497
    bytes, updated at 10:31:07 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 11:33 UTC: `v1_1.2b_ext4000_lora` advanced to
    1,475/1,500 with train loss 1.848 at step 1,400, eval loss 1.865 at step
    1,300, and checkpoint 1,000. Final BPB and transition remain pending.
    Queue and all five relevant processes stayed alive, with 25 complete
    top-level stages, one active stage, and no failure. H200 usage was
    25,481/143,771 MiB at 42% utilization; disk free remained 1,004.64 GiB.
    Private Trackio remained at 244 objects / 11,190,497 bytes, updated at
    10:31:07 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 11:48 UTC: `v1_1.2b_ext4000_lora` completed
    1,500/1,500 with aggregate train loss 2.0745277, final-batch loss
    1.8477289, final eval 1.8613154, and BPB 3.6259127 -> 1.2210235 (66.33%
    gain); checkpoint 1,500 is durable and Trackio synced. The family advanced
    cleanly to `v1_1.2b_ext4000_full`: its 100-step phase A completed and the
    main run reached 295/1,500 with train loss 2.098 at step 200, eval loss
    2.330 at step 100, and phase-A checkpoint 100. Queue and all five relevant
    processes stayed alive, with 25 complete top-level stages, one active
    stage, and no failure. H200 usage was 32,771/143,771 MiB at 66%
    utilization; disk free was 998.76 GiB. Private Trackio advanced to 249
    objects / 12,191,356 bytes at 11:41:23 UTC, and both private 350M Hub
    readbacks remained verified. No recovery or gate bypass was needed;
    completed/private-published sizes remain 1/4 and the monitor remains
    armed.
  - Monitor 2026-09-17 12:03 UTC: the main
    `v1_1.2b_ext4000_full` run advanced to 839/1,500 with train loss 1.834 at
    step 800, eval loss 1.851 at step 700, checkpoint 500, and its durable
    phase-A checkpoint 100. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 32,771/143,771 MiB at 55% utilization; disk free
    was 992.19 GiB. Private Trackio remained at 249 objects / 12,191,356
    bytes, updated at 11:41:23 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 12:18 UTC: the main
    `v1_1.2b_ext4000_full` run advanced to 1,400/1,500 with train loss 1.800
    at step 1,400, eval loss 1.817 at step 1,300, checkpoint 1,000, and its
    durable phase-A checkpoint 100. Queue and all five relevant processes
    stayed alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 32,771/143,771 MiB at 96% utilization; disk free
    was 985.62 GiB. Private Trackio remained at 249 objects / 12,191,356
    bytes, updated at 11:41:23 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 12:33 UTC: `v1_1.2b_ext4000_full` completed its
    100-step phase A and 1,500-step main run with aggregate train loss
    1.9554434, final-batch loss 1.7950310, final eval 1.8169287, and BPB
    3.5953855 -> 1.1561285 (67.84% gain); checkpoint 1,500 and phase-A
    checkpoint 100 are durable and Trackio synced. The family advanced
    cleanly to `v2_1.2b_default_lora`, observed at 180/3,000 with train loss
    4.172 at step 100 and checkpoint 100; eval remains pending. Queue and all
    five relevant processes stayed alive, with 25 complete top-level stages,
    one active stage, and no failure. H200 usage was 22,821/143,771 MiB at 87%
    utilization; disk free was 976.83 GiB. Private Trackio advanced to 252
    objects / 12,853,400 bytes at 12:25:51 UTC, and both private 350M Hub
    readbacks remained verified. No recovery or gate bypass was needed;
    completed/private-published sizes remain 1/4 and the monitor remains
    armed.
  - Monitor 2026-09-17 12:48 UTC: `v2_1.2b_default_lora` advanced to
    529/3,000 with train loss 3.207 at step 500, eval loss 3.296 at step 400,
    and checkpoint 500. Queue and all five relevant processes stayed alive,
    with 25 complete top-level stages, one active stage, and no failure. H200
    usage was 22,821/143,771 MiB at 41% utilization; disk free was 976.81 GiB.
    Private Trackio remained at 252 objects / 12,853,400 bytes, updated at
    12:25:51 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 13:03 UTC: `v2_1.2b_default_lora` advanced to
    900/3,000 with train loss 2.998 at step 900, eval loss 3.005 at step 800,
    and checkpoint 900. Queue and all five relevant processes stayed alive,
    with 25 complete top-level stages, one active stage, and no failure. H200
    usage was 22,821/143,771 MiB at 91% utilization; disk free remained 976.81
    GiB. Private Trackio remained at 252 objects / 12,853,400 bytes, updated
    at 12:25:51 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Immediate pass 2026-09-17 13:18 UTC: `v2_1.2b_default_lora` advanced to
    1,257/3,000 with train loss 2.947 at step 1,200, eval loss 2.906 at step
    1,100, and checkpoint 1,200. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 22,821/143,771 MiB at 98% utilization; disk free
    remained 976.81 GiB. Private Trackio remained at 252 objects / 12,853,400
    bytes, updated at 12:25:51 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Lightweight monitor 2026-09-17 13:20 UTC: `v2_1.2b_default_lora`
    advanced from 1,257 to 1,301/3,000, with train loss 2.858 at step 1,300,
    eval loss 2.882 at step 1,200, and checkpoint 1,300. The queue remained
    alive and no stage failed. No recovery was needed; completed/private-
    published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 13:33 UTC: `v2_1.2b_default_lora` advanced to
    1,604/3,000 with train loss 2.846 at step 1,600, eval loss 2.826 at step
    1,500, and checkpoint 1,600. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 22,821/143,771 MiB at 99% utilization; disk free
    remained 976.81 GiB. Private Trackio remained at 252 objects / 12,853,400
    bytes, updated at 12:25:51 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 13:48 UTC: `v2_1.2b_default_lora` advanced to
    1,961/3,000 with train loss 2.792 at step 1,900, eval loss 2.790 at step
    1,800, and checkpoint 1,900. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 22,821/143,771 MiB at 45% utilization; disk free
    remained 976.81 GiB. Private Trackio remained at 252 objects / 12,853,400
    bytes, updated at 12:25:51 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 14:03 UTC: `v2_1.2b_default_lora` advanced to
    2,323/3,000 with train loss 2.766 at step 2,300, eval loss 2.762 at step
    2,200, and checkpoint 2,300. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 22,821/143,771 MiB at 47% utilization; disk free
    remained 976.81 GiB. Private Trackio remained at 252 objects / 12,853,400
    bytes, updated at 12:25:51 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 14:18 UTC: `v2_1.2b_default_lora` advanced to
    2,683/3,000 with train loss 2.745 at step 2,600, eval loss 2.753 at step
    2,500, and checkpoint 2,600. Queue and all five relevant processes stayed
    alive, with 25 complete top-level stages, one active stage, and no
    failure. H200 usage was 22,821/143,771 MiB at 45% utilization; disk free
    remained 976.81 GiB. Private Trackio remained at 252 objects / 12,853,400
    bytes, updated at 12:25:51 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 14:33 UTC: `v2_1.2b_default_lora` completed
    3,000/3,000 with aggregate train loss 2.9987998, final-batch loss
    2.7516602, final eval 2.7490308, and BPB 3.2216063 -> 2.0551098 (36.21%
    gain); checkpoint 3,000 is durable and Trackio synced. The family advanced
    cleanly to `v2_1.2b_default_full`, which was loading training inputs with
    step metrics/checkpoint pending. Queue and all five relevant processes
    stayed alive, with 25 complete top-level stages, one active stage, and no
    failure. The transition sample showed 10,829/143,771 MiB H200 usage at 0%
    utilization; disk free was 976.79 GiB. Private Trackio advanced to 254
    objects / 13,776,339 bytes at 14:31:36 UTC, and both private 350M Hub
    readbacks remained verified. No recovery or gate bypass was needed;
    completed/private-published sizes remain 1/4 and the monitor remains
    armed.
  - Monitor 2026-09-17 14:48 UTC: `v2_1.2b_default_full` advanced from
    loading to 443/3,000 with train loss 2.297, eval loss 2.312, and
    checkpoint 400. Queue and all five relevant processes stayed alive, with
    25 complete top-level stages, one active stage, and no failure. H200 usage
    was 31,907/143,771 MiB at 60% utilization; disk free was 963.71 GiB.
    Private Trackio advanced to 255 objects / 14,733,422 bytes at 14:36:24
    UTC, and both private 350M Hub readbacks remained verified. No recovery or
    gate bypass was needed; completed/private-published sizes remain 1/4 and
    the monitor remains armed.
  - Monitor 2026-09-17 15:03 UTC: `v2_1.2b_default_full` advanced to
    990/3,000 with train loss 2.154, eval loss 2.143, and checkpoint 900.
    Queue and all five relevant processes stayed alive, with 25 complete
    top-level stages, one active stage, and no failure. H200 usage was
    31,907/143,771 MiB at 58% utilization; disk free remained 963.71 GiB.
    Private Trackio remained at 255 objects / 14,733,422 bytes, updated at
    14:36:24 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 15:18 UTC: `v2_1.2b_default_full` advanced to
    1,525/3,000 with train loss 2.059, eval loss 2.085, and checkpoint 1,500.
    Queue and all five relevant processes stayed alive, with 25 complete
    top-level stages, one active stage, and no failure. H200 usage was
    31,907/143,771 MiB at 58% utilization; disk free remained 963.71 GiB.
    Private Trackio remained at 255 objects / 14,733,422 bytes, updated at
    14:36:24 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Monitor 2026-09-17 15:33 UTC: `v2_1.2b_default_full` advanced to
    2,070/3,000 with train loss 2.101, eval loss 2.073, and checkpoint 2,000.
    Queue and all five relevant processes stayed alive, with 25 complete
    top-level stages, one active stage, and no failure. H200 usage was
    31,907/143,771 MiB at 62% utilization; disk free remained 963.71 GiB.
    Private Trackio remained at 255 objects / 14,733,422 bytes, updated at
    14:36:24 UTC, and both private 350M Hub readbacks remained verified. No
    recovery or gate bypass was needed; completed/private-published sizes
    remain 1/4 and the monitor remains armed.
  - Immediate pass 2026-09-17 15:46 UTC: `v2_1.2b_default_full`
    early-stopped at 2,500/3,000 after patience 5, with aggregate train loss
    2.2376881, final-batch loss 2.0738036, final eval 2.0708070, and BPB
    3.2045671 -> 1.4411784 (55.03% gain); checkpoint 2,500 is durable and
    Trackio synced. The family advanced cleanly to
    `v2_1.2b_ext4000_lora`, which was loading inputs with step metrics and
    checkpoint pending. Queue and all five relevant processes stayed alive,
    with 25 complete top-level stages, one active stage, and no failure. The
    transition sample showed 9,505/143,771 MiB H200 usage at 0% utilization;
    disk free was 961.53 GiB. Private Trackio advanced to 257 objects /
    15,256,780 bytes at 15:45:19 UTC, and both private 350M Hub readbacks
    remained verified. No recovery or gate bypass was needed; completed/
    private-published sizes remain 1/4 and monitor PID 27132 remains armed.
  - Monitor 2026-09-17 00:58 UTC: `v2_350m_default_lora` completed with final
    train/eval losses 2.9541/2.9517 and BPB 3.3941→2.1579. Trackio finalized as
    `synced`; the queue advanced to `v2_350m_default_full`, observed at step
    39/3000. No red gate, publication, or intervention appeared.
  - Recovery 2026-09-17 16:18 UTC: recreated the private Trackio Space after
    the legacy deployment rejected live APIs, restored the 4.8 MiB local
    database, and migrated 264 fragments between private Buckets. Authenticated
    live readback now passes with 18 recovered runs. Enabled four persistent
    pinned-memory dataloader workers without changing batch/LR/steps, passed
    10 focused tests and Ruff locally and remotely, and resumed
    `v2_1.2b_ext4000_lora` from checkpoint 400. It reached 550/3,000 with
    train/eval 2.07454/2.07309 at step 500, checkpoint 500, and 91% H200
    utilization. Three tokenizer families were uploaded to separate private
    model repositories with authenticated readback. Upload of all 54 currently
    durable checkpoints (116.19 GB) to one private versioned repository is
    active, and promoted-model publication now blocks on verified checkpoint
    and tokenizer backup. No gate was bypassed; completed/private-published
    sizes remain 1/4. Exactly one 15-minute monitor is armed (PID 30260).
  - Final recovery verification 2026-09-17 16:35 UTC: explicit
    `TRACKIO_BUCKET_ID` wiring was added to CLM, SFT, DPO, stage tracking, and
    pipeline environment exports. Twenty focused tests and Ruff passed both
    locally and remotely. The queue restarted from checkpoint 600 and reached
    635/3,000; train/eval were 2.07497/2.03956 at step 600. Trackio live
    readback verified this exact active run against the new private Bucket.
    Throughput stabilized near 2.13 seconds/step versus roughly 2.4 before the
    recovery. The private checkpoint repository has authenticated readback for
    9 of 54 inventoried checkpoint folders while the low-priority upload
    continues; future promoted-model publication is hard-gated on completing
    this backup. No gate was bypassed and the single monitor remains active.

---


  - Recovery 2026-09-17 17:42 UTC: dedicated private Trackio Space `Tonic/lfm25-wolof-trackio` RUNNING with history dataset `Tonic/lfm25-wolof-trackio-history` (297 files; 88 imported runs / 43 live). Legacy Buckets preserved. `causal_conv1d` kernel OK on torch 2.11.0+cu128 (no transformers fallback). Production queue resumed `v2_ext4000_lora` from checkpoint-600 under the new Trackio env. Published sizes remain 1/4; monitor PID 27132 remains armed. No gate bypassed.

## Suggested first 10 commands (after files exist)

```bash
cd Experiments/LFM2.5-Base-1B
uv python install
uv sync --group data --group train --group eval --group dev
uv run python scripts/env/00_verify_env.py
uv run python scripts/data/20_audit_datasets.py --out results/data_audit.json
uv run python scripts/clm/06_probe_modules.py --base-id LiquidAI/LFM2.5-350M-Base
uv run python scripts/clm/02_build_tokenizer.py --config configs/tokenizer-ext4000-65k.yaml
uv run python scripts/clm/03_train.py --config configs/scaling/smoke_350m_ext_lora.yaml --smoke
# … only then mini / v1 …
```

---

## Dependency graph (summary)

```text
P0 audit ──► G-DATA
P1 uv ─────► G-UV ──► S0
P2 common ─► G-COMMON
P3 mixture ► G-MIX ──► P7
P4 tok ────► G-TOK-65K ► S2–S8
P5 scripts ► G-SCRIPTS-CLM ► S3+
P6 scaling ► S0…S8 ► P7
P7 SFT ────► S9…S11 ► P9/P11
P8 synth ──► SYN-G* ──► P7/P10.3
P9 eval ───► G-EVAL ──► P11
P10 family ► F-* gates
P11 publish ► PUB
```
