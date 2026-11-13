# Synthetic data — where and how

Natural Wolof data is short, translation-heavy, and thin on reasoning/QA. Synthetic data is **not** a replacement for `cleaned_data_for_clm` / authentic `instruction` text; it fills capability holes before we claim “reasoning” or “long-context” Wolof models.

---

## 1. Priority gaps (from dataset audit)

| Gap | Evidence | Synthetic product |
|---|---|---|
| Reasoning / math | `qa` = 0.3% of SFT; AfriMGSM only in test | Wolof chain-of-thought SFT (+ optional GRPO later) |
| Long context | Corpus median 97 chars; SFT turns short | Multi-doc packs, needle-in-haystack WO, book-chunk QA |
| Balanced instruct | Translation 57.9% | Reverse-instruction & paraphrase from CPT text |
| Orthography | Mixed spellings | Diacritic restore / normalize pairs (extend existing task) |
| Preference | No DPO set | Chosen/rejected pairs (fluency, faithfulness, language) |
| Wolof tool use | tool_use mostly EN schemas | Rewrite user/system to WO; keep JSON args EN |
| Cultural / local knowledge | Thin vs translation | Seeded generation from news/agri/finance sources in centralized corpus |

---

## 2. Where synthetic belongs in the pipeline

```text
Stage CPT (real cleaned_data_for_clm only)
        │
        ▼
Stage SFT mixture (real wolof_sft, reweighted)
        │
        ├── + synth_reasoning   (after mini-SFT gate)
        ├── + synth_longctx     (2.6B/8B mid-train & SFT)
        ├── + synth_ortho       (all instruct sizes)
        └── + synth_prefs       (DPO after SFT)
```

**Rules**

1. Never mix unverified synthetic into CPT until a **mini CPT** on real data already beats base BPB.
2. Cap synthetic ≤ **30%** of SFT tokens in v1 mixtures (avoid model collapse to generatorese).
3. Every synth pack has a **held-out real** eval (AfriMGSM test, native prompts, CLM BPB) — synthetic must not be the only score source.
4. Log generator model, prompt, temperature, filter rates in `meta`.

---

## 3. Recipes

### 3.1 Reasoning (`scripts/data/22_synth_reasoning.py`)

- Seed: AfriMGSM / GSM8K-style items (FR/EN) + existing `qa` rows.
- Generate: step-by-step **Wolof** solution + final answer; verify answer match with regex/numeric check.
- Filter: reject empty CoT, FR-only answers when prompt is WO, answer mismatch.
- Target size: start **5k–20k** for mini; scale to **50k+** if val AfriMGSM-style improves.
- Use on: 1.2B+ instruct; optional 350M distilled shorter CoT.

### 3.2 Long-context (`22_synth_longctx.py`)

- Seed: longer centralized docs (FineWeb-WO, HPLT, wikipedia, religious chapters) + cleaned CLM tails.
- Build packs at 4k / 8k / 16k / 32k tokens (tokenize with **target** LFM tokenizer).
- Tasks: (a) continue document, (b) answer question about paragraph k, (c) needle fact in WO haystack.
- Use on: **2.6B-Base / 8B-A1B** mid-training after short CPT; light exposure on 1.2B (max 4k–8k) only if memory allows.
- Eval: needle accuracy @ lengths; BPB on long eval subset.

### 3.3 Orthography (`22_synth_ortho.py`)

- Seed: `instruction` diacritic examples + Non-Standard Orthography source.
- Pairs: stripped ↔ restored; standardize to a chosen convention for *eval* while training both.
- Use on: all instruct models; report both conventions in human eval.

### 3.4 Preference / DPO (`22_synth_prefs.py`)

- Prompt from val mixture; sample two completions from CPT-SFT checkpoint (T>0).
- Rank with: stronger teacher (FR/WO bilingual judge) + rules (stays in WO, no empty, follows format).
- Start **10k** pairs; DPO β~0.1 (Liquid docs).
- Use on: 1.2B hero then 2.6B.

### 3.5 Wolof tool-calling wrappers

- Take existing `tool_use` rows; translate user/system to WO; keep tool names/args in EN (matches LFM special tokens).
- Filter with schema validation.
- Use on: 1.2B / 2.6B / 8B-A1B (agentic narrative).

### 3.6 Reverse instructions from CPT

- Sample cleaned CLM spans; generate “write an instruction that yields this text” (bilalfaye/wori-style).
- Increases authentic WO assistant text without new translationese.

---

## 4. Quality filters (mandatory)

| Filter | Fail if |
|---|---|
| Language ID | GlotLID not wol_Latn on WO-intended fields (threshold TBD) |
| Length | CoT < 2 steps; longctx pack shorter than target by >20% |
| Answer check | Math final answer ≠ gold |
| Degeneracy | High n-gram repeat / empty / English-only when WO required |
| Leakage | Exact match to SFT `test` or CLM `test` |

---

## 5. Eval gates before scaling synthetic volume

| Gate | Metric | Action if fail |
|---|---|---|
| G0 | Manual 50-sample review | Fix prompts |
| G1 | Mini-SFT +5k synth vs no-synth on val `instruction`/`qa` | Drop or regenerate |
| G2 | AfriMGSM-style / math subset | Only scale if ↑ |
| G3 | CLM BPB on cleaned test | Must not degrade >1–2% |
| G4 | Needle @ 8k (2.6B) | Must beat short-only SFT |

---

## 6. What not to synthesize

- Do not fabricate **evaluation** benchmarks used for publication claims.
- Do not replace `cleaned_data_for_clm` with LLM-generated Wikipedia-style CPT (distribution shift + hallucinations).
- Do not synthesize tool JSON that fails schema validation.

## 7. Implemented offline CLIs

These commands validate or transform supplied fixtures. They never call a paid inference API:

```bash
uv run python scripts/data/22_synth_reasoning.py --input candidates.jsonl --output reasoning.jsonl --review results/sft/reasoning_g0_review.json
uv run python scripts/data/22_synth_longctx.py --input documents.jsonl --output needles_8k.jsonl --target-words 8000 --needles
uv run python scripts/data/22_synth_ortho.py --input authentic_text.jsonl --output ortho.jsonl
uv run python scripts/data/22_synth_prefs.py --input preference_candidates.jsonl --output preferences.jsonl
uv run python scripts/data/22_synth_tool_wo.py --input localized_tools.jsonl --output tools_valid.jsonl
uv run python scripts/data/23_check_synthetic_budget.py --real real.jsonl --synthetic synthetic.jsonl
```

The budget command uses a documented whitespace-token estimate and exits non-zero above 30%. Before training, use the target tokenizer for a final exact token count and retain that report with mixture metadata.
