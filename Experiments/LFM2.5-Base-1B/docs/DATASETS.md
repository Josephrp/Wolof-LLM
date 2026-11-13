## 0. Card reconciliation note

Hub card sizes verified (MCP / dataset READMEs): `cleaned_data_for_clm` train **958 646** / test **4 282**; `wolof_centalized_corpus` **1 018 520**; `wolof_sft` train **990 320** / test **2 250**. Tables below match those cards — no drift at last check.

1. [`galsenai/wolof_centalized_corpus`](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus)
2. [`galsenai/cleaned_data_for_clm`](https://huggingface.co/datasets/galsenai/cleaned_data_for_clm)
3. [`galsenai/wolof_sft`](https://huggingface.co/datasets/galsenai/wolof_sft)

---

## 1. `wolof_centalized_corpus` — raw centralized CLM source

### Facts (from Hub card)

| | |
|---|---|
| Rows | **1 018 520** unique |
| Columns | `text`, `sources` (list — multi-provenance merges) |
| ~Tokens (whitespace) | **72.5 M** |
| Sources | **47** |
| Wolof (GlotLID sample) | **~97%** |
| License umbrella | cc-by-4.0 (per-source licenses still apply) |

Full audit: **2.69%** of rows exceed 2,000 chars and **0.094%** exceed
8,000 chars. `michsethowusu/Code-170k-wolof` accounts for **344,137 rows**
(33.8%); token-mass skew still needs a tokenizer-based measurement.  
Orthography: standard + non-standard mixed. Dedup = exact only.

### Best uses

| Use | How |
|---|---|
| Provenance audits | Filter by `sources` for domain ablations (news vs religious vs code) |
| Tokenizer candidate mining | Sample non-code sources for whole-word candidates |
| Synthetic seed text | Authentic Wolof spans for long-context packing / QA passage pools |
| **Not** default CPT | Too code-heavy; use cleaned split instead |

### Language-specific issues

- Heterogeneous Latin Wolof spelling → fertility metrics must sample diversely.
- Residual FR/EN/AR noise (~3%) — acceptable for CPT; strip URLs/code if rebuilding.
- Multi-source rows (26.6%) — good signal that phrase is “canonical”; optional upweight.

---

## 2. `cleaned_data_for_clm` — **default CPT dataset**

### Facts

| Split | Rows | Column |
|---|---:|---|
| `train` | **958 646** | `text` |
| `test` | **4 282** | `text` |

Pipeline already applied (card):

1. Strip code fences/spans from Code-170k-style lines (−19% chars, 0 rows dropped).
2. NFC normalization.
3. Random split seed **42**, 5 000 eval drawn then **718** near-dupes removed → 4 282 clean test.
4. Source: cleaned export of centralized corpus (`data_v1.parquet`).

This is exactly what SmolLM3 CPT used. `Evaluation/scripts/download_dataset.py` defaults to this dataset.

### Best uses

| Use | How |
|---|---|
| All LFM CPT runs | `load_dataset(...)` train/test; pack with model EOS |
| Fertility / BPB | Always use `test` (clean) |
| Cross-family parity | Same bytes as SmolLM3 → BPB comparable across Smol vs LFM |
| Rebuild | Only if changing cleaning policy; then re-publish Hub version |

### Do not

- Mix with lowercase-only Data-Preprocessing notebook output.
- Evaluate on raw centralized `train` (leakage + code).

---

## 3. `wolof_sft` — **default instruct dataset**

### Facts

| | |
|---|---|
| Version | 0.4.1 (2026-08-31) |
| Train / test | **990 320** / **2 250** |
| Format | JSONL ChatML: `messages`, `task`, `source`, `prompt_lang`, `meta` (JSON **string**) |
| Generator | [galsenai-sft-data](https://github.com/SEYDINA04/galsenai-sft-data) |

### Task mix (train)

| task | n | % | Notes |
|---|---:|---:|---|
| `translation` | 573 832 | **57.9** | Bidirectional WO↔FR/EN — dominates |
| `tool_use` | 206 557 | 20.9 | Function calling + code QA; many EN-native tools |
| `classification` | 120 701 | 12.2 | Labels often English |
| `instruction` | 63 899 | 6.5 | Native WO replies; diacritic restore |
| `intent` | 15 012 | 1.5 | Banking + general; JSON slots |
| `ner` | 7 505 | 0.8 | JSON entities; types in EN |
| `qa` | 2 814 | **0.3** | Extractive + memory — **too small for reasoning** |

Prompt language: wo **479,177** / fr **364,586** / en **146,557**.
The full audit found **81,014** multi-turn rows and **184,541** rows with tool
calls. A 1,000-row JSON sample parsed all `meta` values successfully; no tool
arguments occurred in that sample.

Test split = `belebele` + `afrimmlu` + `afrimgsm` + `afrixnli` held out of SFT train — but card warns **source sentences may exist in pretrain corpus**.

### Best uses

| Goal | Mixture recipe |
|---|---|
| General Wolof chat | Cap translation ≤25–35%; boost `instruction`+`qa` |
| Agentic / LFM story | Keep `tool_use` ~15–25%; ensure Wolof user turns |
| Structured NLP | Upsample ner/intent/classification |
| Reasoning | **Cannot** rely on `qa` alone → synthetic (AfriMGSM-style CoT in WO) |
| Hyperparams | Carve val = 2% of train, stratify by `task`+`source` |
| Final report | Run official `test` **and** a native Wolof human set |

### Dataset-specific footguns

1. **`meta` / `tool_calls[].arguments` are JSON strings** — always `json.loads`.
2. Training without weights → translator.
3. Tool schemas/labels in English while user text is Wolof — model must tolerate bilingual schemas (matches LFM tool tokens).
4. No `validation` split on Hub.
5. Optimistic Afri* numbers if pretrain saw the text.

---

## 4. Recommended data flow

```text
wolof_centalized_corpus ──(audit / synth seeds)──┐
                                                 │
cleaned_data_for_clm ──► CPT (all sizes) ──► merge ──► SFT init
                                                 │
wolof_sft ──► stratify + weights ──► (+ synthetic packs) ──► SFT
                                                 │
Evaluation cleaned test CSV ──► publish BPB/BLEU/chrF
```

### Script todos

- [x] `scripts/data/20_audit_datasets.py` — length, task, source histograms; writes `results/data_audit.json`
- [x] `scripts/data/21_build_sft_mixture.py` — YAML mixture → train/val parquet
- [x] Update `common/data.py` — Hub CLM path and local SFT parquet mixture directories
- [x] `scripts/data/24_split_orthography.py` — procedural standard/non_standard/mixed/not_wolof split + Hub upload
- [x] `scripts/data/25_submit_ortho_job.py` — submit the same pipeline on HF Jobs (`cpu-xl`)

---

## 6. Orthography partitions (procedural)

Wolof Latin spelling is dual (school/CLAD vs informal). Heuristic labels
(`common/orthography.py`) partition the three Hub sources into:

| Output (default namespace `Tonic`) | Contents |
|---|---|
| `cleaned_data_for_clm_ortho_labeled` | CLM train/test + `ortho_label` (+ scores/reason) |
| `wolof_centalized_corpus_ortho_labeled` | Raw corpus + labels |
| `wolof_sft_ortho_labeled` | SFT + labels (from flattened `messages`) |
| `wolof_ortho_partitions` | Configs `{clm,raw,sft}_{standard,non_standard,mixed,not_wolof}` |

Labels match the crowdsource Space (`standard` / `non_standard` / `mixed` /
`not_wolof`). **Procedural ≠ gold** — use consensus votes when available.

```bash
# Local / remote VPS (needs HF_TOKEN with read on galsenai/* and write on namespace)
uv run python scripts/data/24_split_orthography.py --push --private

# HF Jobs (recommended for full ~1M-row corpora)
uv run python scripts/data/25_submit_ortho_job.py
uv run python scripts/data/25_submit_ortho_job.py --smoke   # 500 rows/split dry run
```

Load a partition:

```python
from datasets import load_dataset
ds = load_dataset("Tonic/wolof_ortho_partitions", "clm_standard", split="train")
```

### Load with `uv run`

From `Experiments/LFM2.5-Base-1B` (requires `HF_TOKEN` for private `galsenai/*`):

```bash
uv sync --group data --group dev
uv run python -c "from datasets import load_dataset; ds=load_dataset('galsenai/cleaned_data_for_clm'); print(ds)"
uv run python scripts/data/20_audit_datasets.py --out results/data_audit.json --max-rows 500
```

Or via the shared helper:

```bash
uv run python -c "from common.data import charger_splits_hub; tr, te = charger_splits_hub(max_rows=100); print(len(tr), len(te))"
```

---

## 5. Capability matrix (what each dataset can / cannot teach)

| Capability | cleaned CLM | centralized | wolof_sft | Needs synthetic? |
|---|---|---|---|---|
| Fluent Wolof LM | ✓ primary | raw only | weak | No |
| Translation | incidental | some | ✓ heavy | Optional downweight |
| Instruction following | ✗ | ✗ | ✓ (small) | Upsample + synth |
| Tool / agent | ✗ | ✗ | ✓ EN-heavy | Wolof-wrap tools |
| Math / reasoning | ✗ | ✗ | AfriMGSM in **test** only | **Yes** (train CoT) |
| Long context (4k–32k+) | packing only; sources short | few long docs | short turns | **Yes** (packs) |
| Orthography | mixed | mixed | diacritic subset | Synth pairs |
| NER / intent | ✗ | ✗ | ✓ small | Optional |
| Preference alignment | ✗ | ✗ | ✗ | **Yes** (DPO) |
