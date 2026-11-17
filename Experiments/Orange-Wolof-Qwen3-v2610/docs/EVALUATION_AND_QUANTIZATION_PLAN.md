# Orange Wolof-Qwen3 v2610 — Complete Evaluation & Quantization Plan

> Scope: `Orange/Wolof-Qwen3-{0.6B,1.7B,4B,8B,14B}-v2610`
> ([arXiv:2610.20824](https://arxiv.org/abs/2610.20824), *Wolof-Adapted Industry-Ready
> Open-Weight Language Models*, Lecorvé et al., Orange, Oct 2026).
> This plan **reuses this repository's existing evaluation code and HF Jobs patterns** —
> it does not introduce a new evaluation framework.
> Companion document: [`TRAINING_IMPROVEMENT_PLAN.md`](TRAINING_IMPROVEMENT_PLAN.md).
> Machine-readable source of truth: [`../configs/family.yaml`](../configs/family.yaml).

---

## 0. Gate 0 — Hub audit (done 2026-11-17, live via `hf models info`, user `Tonic`)

Verified against the Hub, not just the model cards:

| Size | Hub repo | SHA (short) | Weights on Hub | Dtype | Params | Recipe (per card) |
|---|---|---|---|---|---|---|
| 0.6B | `Orange/Wolof-Qwen3-0.6B-v2610` | `a296360960` | ✅ `model.safetensors` 1.19 GB | BF16 | 596,049,920 | SFT only (DPO/merge tried, rejected) |
| 1.7B | `Orange/Wolof-Qwen3-1.7B-v2610` | `f83fe208ca` | ❌ **NO WEIGHTS — 11 MB total, tokenizer/config only** | BF16 (config) | — (config only) | SFT + DPO, no merge |
| 4B | `Orange/Wolof-Qwen3-4B-v2610` | `64314c62da` | ✅ 2 shards, 8.06 GB | BF16 | 4,022,468,096 | SFT + DPO, no merge |
| 8B | `Orange/Wolof-Qwen3-8B-v2610` | `eacd4f3428` | ✅ 4 shards, 16.39 GB | **FP16** (merged) | 8,190,735,360 | SFT + DPO + 3-way linear merge |
| 14B | `Orange/Wolof-Qwen3-14B-v2610` | `cc3d985809` | ✅ 6 shards, 29.55 GB | **FP16** (merged) | 14,768,307,200 | SFT + DPO + 3-way linear merge |

All sizes: `Qwen3ForCausalLM`, vocab 151,936 (stock Qwen3 tokenizer, **no Wolof vocab
extension**), `max_position_embeddings: 40960`, generation defaults
`temperature 0.6 / top_p 0.95 / top_k 20`, chat template with `<tools>` support.
License: **CC-BY-NC-SA-4.0 (non-commercial) + separate Orange commercial terms** —
every derived artifact (GGUF, ONNX, AWQ, fine-tune) inherits this; nothing in this
plan may be relicensed Apache-2.0.

### Blocking findings

1. **🔴 `Wolof-Qwen3-1.7B-v2610` cannot be evaluated or quantized — the repo contains
   no weight file at all** (no `model.safetensors`, no shards; `safetensors: null` in
   the Hub API). Action: open a Hub discussion on the repo asking Orange to complete
   the upload; until then the 1.7B is `BLOCKED` in every matrix below and must not be
   silently dropped from reports — it is reported as *missing*, not as zero.
2. **🟡 Dtype split**: 8B/14B ship FP16 (merge output), 4B/0.6B ship BF16. GGUF masters
   and ONNX exports must record source dtype; do not "normalize" 8B/14B to BF16 before
   quantization without logging it (it changes Q4_K_M rounding inputs).
3. **🟡 Context claim vs training**: config advertises 40,960 tokens but SFT used
   2,048 and DPO 1,024. Long-context behaviour is therefore an *open question this
   plan must answer* (§4.5), not a spec to trust.
4. **🟢 Good news**: standard `qwen3` architecture (not `qwen3_5`). Unlike the
   Qwen3.5 family, **no optimum-onnx fork, no Gated-DeltaNet patcher, and no custom
   llama.cpp build are needed** — stock `llama.cpp convert_hf_to_gguf.py` and
   upstream `optimum` ONNX export both support Qwen3.

---

## 1. What the paper/cards claim — and what is independently reproducible

Paper protocol: 63 tasks (35 Wolof, 13 in-house/private), macro-averaged normalized
scores (random = 0, max = 1). Headline aggregates (Orange vs stock Qwen3):

| Aggregate | 0.6B | 1.7B | 4B | 8B | 14B |
|---|---|---|---|---|---|
| Overall | 0.20 → 0.20 | 0.23 → **0.31** | 0.28 → **0.40** | 0.29 → **0.44** | 0.32 → **0.47** |
| Wolof understanding | 0.06 → 0.12 | 0.04 → 0.22 | 0.07 → 0.30 | 0.07 → 0.33 | 0.10 → 0.36 |
| Wolof generation | 0.03 → 0.17 | −0.04 → 0.22 | −0.02 → 0.27 | −0.03 → 0.28 | 0.00 → 0.31 |
| Instruction following | 0.38 → **0.13** 🔻 | 0.50 → **0.25** 🔻 | 0.59 → 0.49 🔻 | 0.63 → 0.57 🔻 | 0.67 → 0.58 🔻 |
| Function calling | 0.53 → **0.23** 🔻 | 0.61 → **0.24** 🔻 | 0.68 → 0.37 🔻 | 0.67 → 0.46 🔻 | 0.70 → 0.35 🔻 |
| Safety | 0.30 → 0.22 🔻 | 0.32 → 0.30 🔻 | 0.38 → 0.45 | 0.45 → 0.64 | 0.54 → 0.73 |

Reproducibility boundary (state this in every published table):

- **Not reproducible**: the 13 in-house tasks, the exact 63-task macro-average, the
  LLM-judge configuration, and the human arena (12 annotators × 9 tasks × 100 samples,
  8B only; Wolof-Qwen3-8B Elo 1051 vs Gemini-2.5-Flash 1112, Qwen3-8B 742).
- **Reproducible with this repo**: Wolof language-modelling quality (BPB — the repo's
  cross-tokenizer gold metric), continuation BLEU/chrF, tokenizer fertility, GGUF
  quantization loss, plus a *new* public chat-task tier (§4.3) that this repo
  currently lacks (`Evaluation/scripts/evaluate_sft.py` is an empty file — see §4.3).
- Therefore our headline deliverable is **not** "we reproduced 0.47" but:
  *same-corpus, same-protocol BPB/BLEU/chrF for all 5 Orange sizes vs their stock
  Qwen3 counterparts and vs the repo's own Wolof families, + quant-loss deltas,
  + a public, re-runnable chat benchmark where Orange's private suite is opaque.*

---

## 2. Reuse map — existing code this plan runs on

| Need | Existing asset (do not rewrite) | Path |
|---|---|---|
| HF CLM eval (BPB/PPL/BLEU/chrF) | `evaluate_clm.py` — BPB is tokenizer-independent; BLEU budget is per-model-tokenizer normalized | `Evaluation/scripts/evaluate_clm.py` |
| Wolof normalization | `verbalize_text()` applied before all metrics (`apply_verbalization: true`) | `Evaluation/scripts/verbalizer.py` |
| Eval corpus | 4,282 Wolof texts, `text` column | `Evaluation/data/corpus_test_clm.csv` |
| Config format | One YAML per run (`base_model_id`, `corpus_csv`, `output_json`, …) | `Evaluation/configs/clm/qwen.yaml`, `oolel.yaml` |
| Prior baselines on same corpus | Qwen3.5-2B: BPB 2.2576 / chrF 21.79; Oolel-v0.1: BPB 1.4183 / chrF 20.79 | `Evaluation/metrics/clm/{qwen,oolel}.json` |
| Family eval runner pattern | matrix runner + `--resume` state file + `FAMILY_EVAL.md` generation | `Experiments/Qwen3.5-Wolof-Family/scripts/publication/55_run_eval_family.py`, `56_publish_hub_metrics.py`, `docs/FAMILY_EVAL.md` |
| Golden checkpoint selection | BPB-select top candidates, prune pool | `…/Qwen3.5-Wolof-Family/scripts/publication/61_golden_eval_select.py` |
| GGUF eval (llama.cpp PPL + BLEU/chrF) | `evaluate_gguf.py` — protocol tag `gguf-llama-cpp`, explicitly *not* HF-BPB-equivalent | `Evaluation/scripts/evaluate_gguf.py` |
| GGUF conversion (local/H200) | `70_convert_gguf.py` → `71_publish_gguf.py` → `72_run_gguf_family.py`, quant set `BF16,F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0`, SHA256SUMS | `…/Qwen3.5-Wolof-Family/scripts/publication/70–72_*.py`, `configs/publication/gguf_family.yaml` |
| GGUF on HF Jobs (CPU) | PEP 723 `uv` script, `cpu-xl`, builds llama.cpp, uploads `-GGUF` repos | `…/Qwen3.5-Wolof-Family/scripts/jobs/job_gguf_family.py` |
| ONNX/WebGPU on HF Jobs | PEP 723 script, ORT `fp16 / quantized(QInt8) / q4 / q4f16`, `transformers.js_config` external-data handling | `…/Qwen3.5-Wolof-Family/scripts/jobs/job_onnx_family.py`, `configs/publication/onnx_family.yaml` |
| Tokenizer fertility probe | word-list + corpus tokens/char, unk rate (adapt Cohere→Qwen tokenizer) | `Experiments/TinyAya-Wolof-Family/scripts/clm/02c_fertility_probe.py` |
| Hub metrics publication | `metrics.json` + README `model-index` injection between eval markers | `…/scripts/publication/56_publish_hub_metrics.py` |

New code in this tree is limited to: family configs, three HF Jobs entrypoints
(adapted, §6), a local family runner, and an aggregator — mirroring the Qwen3.5
family layout file-for-file so maintainers already know where everything lives.

---

## 3. Evaluation matrix

### 3.1 Models under test (15 runs for Tier 1)

| # | Model | Role |
|---|---|---|
| 1–5 | `Orange/Wolof-Qwen3-{0.6B,1.7B,4B,8B,14B}-v2610` | Systems under test (1.7B blocked, §0) |
| 6–10 | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B}` (stock instruct) | Matched-size baselines — the exact comparison the cards make |
| 11 | `soynade-research/Oolel-v0.1` | Prior Wolof specialist (already in repo metrics) |
| 12 | Repo champions on same corpus: Qwen3.5-Wolof Instruct (0.8B/2B/4B/9B), TinyAya-Wolof, LFM2.5-Wolof best-per-size | Cross-family positioning ("which Wolof family wins per size class?") |

### 3.2 Tier 1 — HF CLM (gold, mandatory, runs first)

Protocol = `evaluate_clm.py` exactly as configured in `../configs/eval/clm/*.yaml`:
verbalized `corpus_test_clm.csv` (4,282 texts, full corpus for BPB; 100 texts,
seed 0, prompt-ratio 0.5, length-ratio 1.3 for BLEU/chrF), `bpb_max_len 1024`.

- **Primary metric: BPB ↓** (only metric valid across tokenizers — Orange uses stock
  Qwen3 vocab, repo families use extended vocabs).
- PPL reported but **only compared within identical tokenizers** (Orange-vs-Qwen3
  pairs are valid; Orange-vs-Qwen3.5-Wolof PPL is not).
- Output: `Evaluation/metrics/clm/orange_wolof_qwen3_{size}.json` +
  `qwen3_{size}.json`, aggregated into `results/FAMILY_EVAL.md` by
  `scripts/publication/aggregate_eval.py` (pattern: Qwen3.5 `56_publish_hub_metrics.py`).
- Smoke first: `--smoke` equivalent = 32 texts / 16 BLEU pairs per model to catch
  load/chat-template errors before paying for the full corpus.

Expected sanity ranges (flag, don't fail, if outside): Orange BPB should land well
below stock Qwen3 at each size (cards imply large Wolof gains) and in the
neighbourhood of the repo's CPT models (Qwen3.5-Wolof CPT BPB 0.85–1.05 on the
ortho-filtered corpus — note: *different corpus slice*, so cross-family BPB is only
strictly comparable on the shared `corpus_test_clm.csv` run).

### 3.3 Tier 2 — Chat / instruction evaluation (new, required, currently missing)

`evaluate_clm.py` scores *raw continuation*, not chat behaviour — it cannot detect
the cards' known regressions (instruction following, function calling) nor validate
translation quality. `Evaluation/scripts/evaluate_sft.py` and
`evaluate_tokenizer.py` are **empty files** today; Tier 2 fills that gap with a
public suite built only from releasable data:

| Task block | Construction | Metric | Why |
|---|---|---|---|
| WO→FR / FR→WO translation | Held-out slice of public bilingual data **disjoint from any training split we control**; never Orange's private data | chrF (primary), BLEU | Cards' bilingual SFT block is 182K pairs; lexical metrics + spot human check |
| Wolof QA / FAQ | Public Wolof QA + repo SFT *test* split (stratified, already published by repo pipelines) | LLM-judge absolute score (document judge model + prompt; validate on 100 native-speaker labels, report agreement) | Mirrors paper's judge protocol transparently |
| Instruction following (WO/FR/EN) | IFEval-style verifiable constraints translated/adapted to Wolof + FR/EN originals | Constraint pass rate | Directly probes the 🔻 regression |
| Function calling | BFCL-style single-turn calls + **no-tool negative set** (prompts where the correct action is *no call*) | AST/exact match; **false-positive call rate** | Paper attributes regression to excessive tool calls — measure exactly that |
| Code-switching / orthography robustness | Same prompts in standard vs non-standard Wolof orthography (repo `ortho` labelling) | Score delta standard↔non-standard | Repo's core thesis; Orange cards flag this as untested |
| Long context | Needle + summarization at 4K / 8K / 16K / 32K (config claims 40,960; trained at 2,048) | Retrieval accuracy by depth/length | §0 finding 3 |
| Safety (Wolof-context) | Wolof + FR harmful/benign prompt pairs, Senegal-context scenarios | Refusal precision/recall (judge + human spot-check) | Cards: English safety ≠ Wolof safety |

Rules: chat template from each repo's own `chat_template.jinja`; generation
`temperature 0` for classification/extraction/function-calling, card defaults
(0.6/0.95/20) for open generation; every prompt set versioned as a Hub dataset
(`Tonic/wolof-qwen3-v2610-chatbench-v1`) so results are re-runnable by third parties —
the property Orange's private 13 tasks lack.

### 3.4 Tier 3 — Human evaluation (replicate the 8B arena, extend down)

Blind arena, native Wolof speakers, correctness/formatting/Wolof-quality axes
(paper protocol): 8B first (direct comparability with published Elo), then 4B and
0.6B (deployment-relevant sizes). Minimum 6 annotators, 50 samples/task to bound
cost; report Elo + win probability with bootstrap CIs. Gate: do not publish
"human-preferred" claims below this sample floor.

### 3.5 Cross-cutting measurements (cheap, always collected)

- **Tokenizer fertility**: tokens/char + tokens/word on the CLM corpus for stock
  Qwen3 tokenizer (adapt `02c_fertility_probe.py`) — quantifies what a vocab
  extension (Training Plan §3) would buy.
- **Latency/throughput**: tokens/s + peak VRAM at batch 1/8 on the eval GPU per size
  (needed for the quantization Pareto, §5.4).
- **Carbon/cost log**: wall-clock + HF Jobs flavor per run, appended to metrics JSON
  (paper reports CodeCarbon for training; we report it for eval/quant too).

---

## 4. Execution — where each tier runs

### 4.1 HF Jobs (primary, per user request; prices live-checked 2026-11-17)

`hf jobs uv run` with the PEP 723 scripts in `../scripts/jobs/` (adapted from the
Qwen3.5 family jobs; `hf auth whoami` = `Tonic`, orgs include `galsenai`):

| Job | Script | Flavor | Est. wall-clock | Est. cost | Output |
|---|---|---|---|---|---|
| CLM eval, 14B + baseline | `job_eval_clm_family.py` (`ONLY=14b`) | `h200` ($5.00/h) or `a100-large` ($2.50/h) | 1.5–3 h | $4–15 | metrics JSON → `Tonic/Wolof-Qwen3-v2610-eval` (dataset) |
| CLM eval, 8B/4B pairs | same (`ONLY=8b,4b`) | `l40sx1` ($1.80/h) / `a10g-large` ($1.50/h) | 1–2 h | $2–4 | same |
| CLM eval, 0.6B (+1.7B when unblocked) | same | `l4x1` ($0.80/h) / `t4-medium` ($0.60/h) | <1 h | <$1 | same |
| GGUF matrix (per size) | `job_gguf_family.py` | `cpu-xl` ($1.00/h, 124 GB RAM) | 1–3 h (14B worst) | $1–3 / size | `Tonic/Wolof-Qwen3-{size}-v2610-GGUF` |
| ONNX matrix (0.6B/4B only) | `job_onnx_family.py` | `cpu-xl` ($1.00/h) | 1–2 h / size | $1–2 / size | `Tonic/Wolof-Qwen3-{size}-v2610-ONNX` |
| Tier-2 chat bench | `job_eval_clm_family.py --suite chat` (same image) | size-matched GPU as CLM | +50% of CLM time | +$3–10 total | same eval dataset repo |

Total programme estimate: **≈ $25–60** for the full Tier 1 + GGUF + ONNX + Tier 2
pass. Always launch with `--detach`, then `hf jobs logs <id> --follow`; set
`--timeout` ≥ 2× estimate; pass the token as `--secrets HF_TOKEN` (never bake it in).

Local/H200 alternative (repo's established path, zero marginal cost if the host is
up): `scripts/publication/run_eval_family.py --resume` +
`72_run_gguf_family.py`-equivalent flow. Use HF Jobs when the H200 host is busy or
for the CPU-only GGUF/ONNX stages (they never needed the GPU).

### 4.2 Sequencing & gates

```
G0 Hub audit (§0) ──► G1 smoke CLM (32 texts, all sizes)
   ──► G2 Tier-1 full CLM (15 runs) ──► G3 Tier-2 chat bench ──► G4 human arena (8B→4B→0.6B)
   └──► G5 GGUF conversion ──► G6 GGUF eval (Δ vs HF) ──► G7 ONNX export+eval ──► publish
```

- **G1 exit**: every loadable model produces finite BPB on smoke; 1.7B recorded BLOCKED.
- **G2 exit**: `FAMILY_EVAL.md` table complete; any Orange size with BPB ≥ its stock
  Qwen3 baseline triggers a config/chat-template investigation before proceeding
  (would contradict the cards and indicate a protocol bug on our side first).
- **G6/G7 exit**: quant acceptance thresholds (§5.3) met, else quant is published
  labelled `experimental` with the failing delta in its README.

---

## 5. Quantization plan

### 5.1 Format strategy (matches repo precedent)

| Format | Audience | Sizes | Precedent |
|---|---|---|---|
| **GGUF (llama.cpp)** — primary | Local/desktop/server CPU+GPU, Ollama/LM Studio | All loadable (0.6/4/8/14B) | Qwen3.5/TinyAya/LFM GGUF families + `evaluate_gguf.py` |
| **ONNX (Transformers.js/WebGPU)** | Browser demo (`apps/qwen35-wolof-webgpu` pattern) | 0.6B, 4B only (8B/14B excluded: browser download/RAM; 14B FP16 ≈ 30 GB) | `job_onnx_family.py`, `apps/qwen35-wolof-webgpu` |
| AWQ / FP8 (vLLM serving) — *phase 2, optional* | GPU serving | 8B, 14B | No repo precedent; only if a serving need is confirmed. Not in the first publish wave. |

Target namespaces (cannot publish under `Orange/`):
`Tonic/Wolof-Qwen3-{0.6B,4B,8B,14B}-v2610-GGUF` and `…-ONNX`, each README carrying
`base_model: Orange/Wolof-Qwen3-…`, the CC-BY-NC-SA-4.0 license tag, SHA256SUMS,
and the quant-eval table. Quant repos of the 1.7B are **not created** until weights exist.

### 5.2 GGUF matrix (per size)

Master: convert from the Hub dtype as-is (FP16 for 8B/14B, BF16 for 4B/0.6B) to an
F16 master, then quantize — exactly `70_convert_gguf.py` semantics
(`QUANTS = [F16, Q8_0, Q5_K_M, Q4_K_M, Q4_0]`, lean set from the HF-Jobs variant;
add `Q6_K` for 8B/14B where the size/quality point is attractive):

| Quant | Bits/weight (approx) | 0.6B | 4B | 8B | 14B | Role |
|---|---|---|---|---|---|---|
| F16 | 16 | 1.2 GB | 8.0 GB | 16.4 GB | 29.5 GB | Reference master |
| Q8_0 | 8.5 | 0.64 GB | 4.3 GB | 8.7 GB | 15.7 GB | Near-lossless serving |
| Q6_K | 6.6 | — | — | 6.8 GB | 12.2 GB | Quality desktop (8B/14B) |
| Q5_K_M | 5.7 | 0.43 GB | 2.9 GB | 5.9 GB | 10.6 GB | Balanced |
| Q4_K_M | 4.8 | 0.36 GB | 2.4 GB | 4.9 GB | 8.8 GB | **Default recommendation** (repo precedent) |
| Q4_0 | 4.5 | 0.34 GB | 2.3 GB | 4.6 GB | 8.3 GB | Legacy/mobile compat |

(Sizes ≈ params × bits/8 + metadata/KV overhead; measured sizes replace these
estimates in each repo's `gguf_meta.json` after conversion.)

Conversion notes specific to these checkpoints:
- Stock `qwen3` → current llama.cpp `convert_hf_to_gguf.py` works unmodified
  (contrast Qwen3.5 hybrid cache, which needed patches).
- Preserve `chat_template.jinja` in the GGUF metadata (llama.cpp reads it);
  verify with the repo smoke prompt (`Salaam aleekum, naka nga def?`, 48 tokens)
  plus one Wolof tool-call prompt per size to catch template loss.
- Record source SHA (`cc3d985809`, …) in `gguf_meta.json` so a silent Orange
  re-upload invalidates our quants loudly.

### 5.3 Quant acceptance — HARD GATE before any publishing (per model, per quant)

> Added 2026-11-17: **no model repo — private or public — is created until that
> model and each of its quant files/dtypes have been evaluated and the results
> are rendered into its model card.** Enforcement lives in
> `scripts/publication/publish_quant.py`, which fails closed (exit 2, manifest
> entry `blocked_eval_gate`) when any required artifact is missing.

Required artifacts per model, all SHA-matched to the pinned Orange revision:

1. **Tier-1 HF JSON** (`orange_wolof_qwen3_{size}.json` with `src_sha` stamped by
   `job_eval_clm_family.py`) — must pre-exist in `Tonic/Wolof-Qwen3-v2610-eval`;
   the GGUF/ONNX jobs fetch it *before* converting, so an unevaluated size
   blocks early. Conversion of an unevaluated model may happen, publishing may not.
2. **Per-quant GGUF eval JSON** (`{QUANT}_gguf_eval.json`, from
   `Evaluation/scripts/evaluate_gguf.py` on CPU, `n_gpu_layers: 0`, plus
   `file_sha256` stamped by the job) — including the F16 master as the ΔPPL
   reference (evaluated, never uploaded). A quant whose eval crashes = `fail`.
3. **Per-dtype ONNX eval JSON** (`{dtype}_onnx_eval.json` from
   `scripts/publication/eval_onnx_dtype.py`: chrF subset + generation parity +
   `file_sha256`).
4. **Verdicts** computed in `publish_quant.py` from the thresholds below:
   `pass`/`compat-only` files are uploaded; `fail` files are **not uploaded**
   (listed on the card as evaluated-but-rejected, recorded in the manifest).
   The card is generated *from these JSONs*; an empty eval section raises.

Flavor consequence: because artifacts must be retained until the gate passes,
the cheapest-feasible assignment is 0.6B GGUF/ONNX → `cpu-basic` ($0.01/h),
4B GGUF → `cpu-upgrade` ($0.03/h), 8B/14B GGUF and 4B ONNX → `cpu-xl` ($1.00/h).
Quantization and per-quant eval are CPU-only throughout; the only GPU spend in
the whole quantization programme is the upstream Tier-1 HF reference eval.

For every GGUF quant, run `Evaluation/scripts/evaluate_gguf.py`
(`../configs/eval/gguf/*_q4_k_m.yaml` pattern, extended per quant) on the same
verbalized corpus and compare against the Tier-1 HF numbers for that size.
Protocol tag `gguf-llama-cpp` — PPL is compared **within** the GGUF family, chrF/BLEU
are compared across to HF:

| Quant tier | Δ chrF vs HF (max drop) | Δ llama.cpp PPL vs F16 GGUF | Verdict rule |
|---|---|---|---|
| Q8_0 / Q6_K | ≤ 0.5 pt | ≤ +1% | Publish as quality tier |
| Q5_K_M | ≤ 1.0 pt | ≤ +3% | Publish |
| Q4_K_M | ≤ 1.5 pt | ≤ +5% | Publish as default if met |
| Q4_0 | ≤ 2.5 pt | ≤ +8% | Publish labelled `compat`, never default |

Also: 20-prompt Wolof generation spot-check per quant (side-by-side with HF output
in the quant repo README), tokenizer/chat-template smoke, and load tests in
llama.cpp + Ollama (GGUF metadata compat). Any Wolof-specific collapse (e.g. chrF
holds but diacritics `ñ/ŋ/à` degrade) blocks the default label — check a
diacritic-heavy slice separately; aggregate chrF can hide it.

### 5.4 ONNX / WebGPU matrix (0.6B, 4B)

Export with upstream `optimum` (`main_export`, task `text-generation-with-past`,
dynamo) — the Qwen3.5 PR fork in `job_onnx_family.py` is **not** needed for `qwen3`;
the adapted script in this tree drops it. Dtypes per repo precedent:
`fp16` (export dtype for 4B), `quantized` (QInt8 dynamic), `q4`, `q4f16`
(MatMul4Bits, block 32) with `transformers.js_config.use_external_data_format`
and `kv_cache_dtype` written into `config.json`.
Acceptance: (a) ORT generation parity smoke vs HF on 20 Wolof prompts,
(b) Tier-1-style chrF on 100-pair subset with the `q4f16` build (the one the demo
loads), drop ≤ 2 pts vs HF, (c) the static WebGPU app
(`apps/qwen35-wolof-webgpu` cloned → pointed at the new ONNX repo) loads and
generates in-browser. 1.7B joins this matrix the day its weights exist — it is
arguably the best browser size, so its absence is called out in the demo README.

### 5.5 Deliverables checklist (per size)

- [ ] Tier-1 CLM JSON in `Evaluation/metrics/clm/` + row in `FAMILY_EVAL.md`
- [ ] Tier-2 chat-bench JSON + prompt-set version recorded
- [ ] GGUF repo: all quants + `SHA256SUMS.json` + `gguf_meta.json` (source SHA) + eval table
- [ ] ONNX repo (0.6B/4B): `onnx/` dtypes + `onnx_meta.json` + WebGPU smoke result
- [ ] `metrics.json` + eval block injected into the *quant* repos' READMEs (never edit Orange's repos)
- [ ] Cost/carbon line appended (flavor, wall-clock, est. $)

---

## 6. Risks & open questions

| Risk | Mitigation |
|---|---|
| 1.7B weights never appear | Report as blocked; optionally note in a Hub discussion; family tables keep the empty row visible |
| Orange re-uploads a checkpoint silently | Pin SHAs (§0) in `family.yaml`; aggregator warns on SHA mismatch |
| Our CLM corpus ≠ Orange's domains (customer-care) | Tier-2 FAQ/conversation blocks + human arena cover conversational quality; state corpus provenance in every table |
| LLM-judge bias for Wolof | Native-speaker validation subset + report judge/annotator agreement, as the paper did |
| CC-BY-NC-SA contamination | All derived repos carry the same license + Orange attribution; no commercial-use claims anywhere |
| HF Jobs cost creep on 14B | Smoke gate G1, `--timeout`, `ONLY=` filters, detach + log polling; hard budget cap ≈ $60 for wave 1 |
