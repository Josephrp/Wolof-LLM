# Orange Wolof-Qwen3 v2610 — Complete Training Improvement Plan

> Goal: a **v2611** Wolof-Qwen3 family, trained with this repository's pipeline,
> that keeps Orange's Wolof gains and repairs their documented regressions.
> Orange recipe (from the cards/paper): Wolof-centric SFT (446K samples / 109.4M
> tokens, 3 ep, lr 2e-5, seq 2048, DeepSpeed ZeRO-3) → DPO on DOLCI (233,804 pairs,
> ~98% English, **0% Wolof**, β 0.2, lr 5e-6, seq 1024) → for 8B/14B only, an
> equal-weight (⅓/⅓/⅓) linear merge of {stock, SFT, DPO}. 4B/1.7B ship post-DPO;
> 0.6B ships SFT-only because DPO/merge did not help it.
> Evaluation gates for every stage below are defined in
> [`EVALUATION_AND_QUANTIZATION_PLAN.md`](EVALUATION_AND_QUANTIZATION_PLAN.md).

---

## 1. Diagnosis — what to fix, with evidence

| # | Weakness | Evidence (cards) | Root-cause hypothesis |
|---|---|---|---|
| D1 | **Function calling collapses at every size** | 0.70→0.35 (14B), 0.67→0.46 (8B), 0.68→0.37 (4B), 0.61→0.24 (1.7B), 0.53→0.23 (0.6B); "mainly excessive tool calls when no tool is required" | English-only FC SFT (44K) with no no-tool negatives, then Wolof-heavy training shifts the call/no-call prior; nothing in DPO (English general) restores calibration |
| D2 | **Instruction following drops at every size** | −0.09 (14B) … −0.25 (1.7B), −0.25 (0.6B) | Machine-translated instruction data (translationese, source-language artefacts — cards admit this) + catastrophic forgetting with no FR/EN replay discipline |
| D3 | **Small models barely net-gain** | 0.6B overall 0.20→0.20; EN/FR understanding, safety all *down*; only Wolof gen up | Capacity: full-parameter SFT on 109M tokens overwrites general skills; DPO/merge rejected for 0.6B because they made it worse — recipe doesn't scale down |
| D4 | **DPO contains zero Wolof** | "~98% English, none in Wolof… a general-capability alignment stage" | Alignment tax is paid in English while Wolof behaviour is left unaligned — Wolof preference quality (politeness, register, orthography) never optimized |
| D5 | **No continued pretraining (CPT)** | SFT is "the only adaptation stage that uses Wolof data"; recipe starts from instruct Qwen3 | Wolof knowledge is injected only through instruction format; raw Wolof modelling (idiom, morphology, code-switching statistics) is never learned at LM level. Contrast: this repo's CPT gives BPB 2.78→1.05 (0.8B) before any SFT |
| D6 | **Stock tokenizer, no Wolof extension** | vocab 151,936 = stock Qwen3 on all sizes | High Wolof fertility → more tokens/word, worse effective context, higher latency/cost; repo's ext-tokenizer work (fertility-gated append + mean-pool init) is directly applicable |
| D7 | **Context: trained 2,048 / advertised 40,960** | SFT seq 2048, DPO 1024, "only limited long-context testing" | Long-context ability is inherited-but-eroded; never trained or measured in Wolof |
| D8 | **Conversation data is tiny** | 7K samples / 5.7M tokens = 1.6% of SFT samples, yet customer care is the stated use case | Multi-turn Wolof dialogue underweighted vs 215K translated single-turn instructions |
| D9 | **Merge is a single fixed point** | ⅓/⅓/⅓ equal merge, 8B/14B only; "evaluated but not retained" for 4B/1.7B/0.6B | No merge-weight search, no per-layer merging, selection on the private suite only — merge benefit left unmeasured on public metrics |
| D10 | **Single seed, single checkpoint** | "Results are reported for a single final checkpoint" | No checkpoint-selection protocol (repo's golden-BPB pool does exactly this), no variance estimate |
| D11 | Orthography/dialect coverage unknown | Cards: data "cannot represent every dialect, register, spelling convention"; repo thesis: real Senegalese Wolof is largely non-standard/mixed | Orange mixture is likely formal/standard-leaning (MT output); everyday/code-switched Wolof is this repo's `non_standard + mixed` speciality |

Non-goals (explicitly): beating Gemini-class API models overall; autonomous-agent
readiness (cards correctly scope this out); commercial licensing (CC-BY-NC-SA).

---

## 2. Strategy — repo pipeline, Orange starting points

This repo already operates the exact pipeline needed, on three families
(LFM2.5, TinyAya, Qwen3.5). The v2611 recipe **starts from Orange's released
checkpoints** (they are the strongest Wolof starting point available) and applies
the repo stages Orange skipped, in repo order:

```
Orange Wolof-Qwen3 v2610 (per size)
  └─► Stage A  Fertility probe + optional vocab extension (G2b-style gate)
  └─► Stage B  CPT on repo Wolof corpus (NTP 70/20/10, ortho-labelled) — 3 ep LoRA
  └─► Stage C  Dual SFT: (C1) from Orange checkpoint, (C2) from CPT-merged
  └─► Stage D  Wolof+FR+EN preference alignment (DPO) with replay & FC repair
  └─► Stage E  Merge sweep (incl. Orange's ⅓/⅓/⅓ point) + golden-BPB selection
  └─► Stage F  Long-context extension (progressive 8K→16K) [8B/14B first]
  └─► Publish: HF + GGUF + ONNX per Evaluation & Quantization Plan §5
```

Per-size differentiation (fixes D3 — one recipe does not fit 0.6B and 14B):

| Size | Start | CPT | SFT | DPO | Merge | Notes |
|---|---|---|---|---|---|---|
| 0.6B | Orange SFT ckpt | LoRA CPT, lr lower, 2 ep | LoRA SFT (not full) — protect general skills | **Skip or Wolof-only tiny DPO** (Orange showed EN-DPO hurts here) | Merge sweep incl. high stock weight | Success = overall > 0.20 with Wolof gains *retained* |
| 1.7B | ⛔ blocked until weights exist | — | — | — | — | Fallback if permanently missing: train from `Qwen/Qwen3-1.7B` with full repo recipe |
| 4B | Orange post-DPO | LoRA CPT | Dual SFT | Wolof-mix DPO | Sweep (Orange skipped merge here — test it properly) | Best effort/quality ratio; do this size **first** as recipe pilot |
| 8B | Orange merged | LoRA CPT | Dual SFT | Wolof-mix DPO | Sweep around ⅓/⅓/⅓ | Human-arena comparability with paper |
| 14B | Orange merged | LoRA CPT (QLoRA if VRAM-bound) | Dual SFT | Wolof-mix DPO | Sweep | Last; only after 4B/8B gates pass |

Pilot order: **4B → 0.6B → 8B → 14B**. 4B validates the recipe cheaply; 0.6B
validates the small-model protections before expensive sizes run.

---

## 3. Stage specifications

### Stage A — Tokenizer (addresses D6)

- Run the fertility probe (adapted `TinyAya…/02c_fertility_probe.py`) on stock
  Qwen3 tokenizer over the repo CLM corpus: tokens/char, tokens/word on the repo
  Wolof word-list (`jefandikoo`, `wooral`, …), unk rate (=0 expected for BPE; the
  signal is fertility, not unk).
- If Wolof fertility ≥ 1.25× French fertility on the same-concept texts, build the
  repo-style extension: pretok-chunk append into `tokenizer.json` BPE vocab
  (LFM/Qwen3.5 method — **not** naive `add_tokens`), +2K/+4K variants, per-token
  mean-pool embedding init, specials preserved; gate = fertility improvement with
  zero regression on FR/EN fertility. Config precedent:
  `Qwen3.5-Wolof-Family/configs/tokenizer/ext_wolof_anchors.yaml`.
- Inject the repo anchors `<|lang_wo|>` / `<|ortho_ns|>` at data-prep stage (not as
  vocab surgery) so prompts can condition language/orthography explicitly.
- **Decision point**: extension changes the tokenizer → all downstream BPB stays
  comparable (BPB is tokenizer-independent) but PPL comparisons vs Orange break;
  record `tokenizer: stock|ext2000|ext4000` in every metrics JSON.

### Stage B — Continued pretraining (addresses D5, D11)

- Data: repo NTP `70_20_10` Wolof corpus, ortho-labelled; **two arms** —
  (B-std) keep all orthographies (closer to Orange), (B-ns) repo default: drop
  `standard`, keep `non_standard + mixed` + FR/EN retention slice. Run B-std first
  (less distribution shock from Orange's starting point), B-ns as ablation on 4B.
  Precedent: `Qwen3.5-Wolof-Family/scripts/data/01_materialize_sources.py`,
  `02_ortho_label_and_filter.py`.
- Method: LoRA (r 16 @0.6–1.7B, r 32 @4B+, α 64, all-linear + `modules_to_save:
  [embed_tokens, lm_head]` when tokenizer extended), 3 epochs, seq 2048, lr 2e-4
  (LoRA) with cosine — repo CPT configs (`configs/clm/*/v1_ntp70_ortho_lora.yaml`)
  transfer almost verbatim; only `base_id` changes to the Orange repo.
- Gate G-CPT: BPB on `corpus_test_clm.csv` improves ≥ 10% relative vs the Orange
  starting checkpoint, FR/EN BPB regression ≤ 3% (retention slice doing its job).

### Stage C — SFT v2 mixture (addresses D1, D2, D8)

Rebalance Orange's mixture; target ≈ 450K samples but re-proportioned:

| Block | Orange | v2611 target | Change |
|---|---|---|---|
| Bilingual FR↔WO translation | 182K / 10.0M tok | Keep, **human-verified subset upsampled ×2** | Quality over raw count |
| Translated general instructions | 215K / 78.8M tok | **Filter**: back-translation chrF + language-ID + translationese classifier; drop worst quartile | Fixes D2 at the source |
| Conversation (multi-turn WO) | 7K / 5.7M tok | **≥ 30K** via repo stratified SFT + synthetic multi-turn expansion from FAQ/ customer-care patterns (no private data) | Fixes D8 |
| Function calling | 44K EN only | 44K EN + **WO/FR tool prompts** + **≥ 20% no-tool negatives** | Fixes D1 false positives |
| FR/EN replay (general instruct) | implicit | Explicit 20–25% of mixture, stock-Qwen-quality sources | Forgetting brake (D2/D3) |
| Code-switching / non-standard ortho | ~absent | Repo `ortho_ns_mixed` SFT slice with `<|ortho_ns|>` anchor | Fixes D11 |

- Dual-track per repo habit: **C1** SFT from Orange checkpoint, **C2** SFT from
  CPT-merged (Stage B output merged via `05_merge_adapter.py` pattern).
  3 epochs, assistant-only loss, seq 2048 (4096 for conversation block if budget
  allows), lr 2e-4 LoRA / 2e-5-equivalent behaviour preserved by LoRA scaling.
- Checkpoint discipline (fixes D10): save pool (keep 15), **golden-BPB select**
  (`61_golden_eval_select.py` pattern) on 200-text subset, then Tier-1/Tier-2 eval
  only on the selected checkpoint per track.

### Stage D — Preference alignment with Wolof (addresses D1, D2, D4)

- Replace DOLCI-only DPO with a **mixed preference set** (~120K pairs):
  - Wolof pairs (~40%): chosen = verified/human or best-of-n from the Stage-C
    model ranked by native-speaker spot-checks; rejected = on-policy samples from
    the same model (repo `05_build_prefs.py` builds *weak shuffled* negatives —
    explicitly upgrade this: on-policy rejected, length-controlled, no shuffled
    chosen-as-rejected shortcuts for production runs).
  - FR/EN pairs (~40%): DOLCI subset retained for general alignment.
  - Function-calling pairs (~20%): chosen = correct call **or correct no-call**;
    rejected = spurious call / wrong arguments — directly optimizes D1's metric.
- Hyperparameters: β sweep {0.1, 0.2} (Orange used 0.2), lr 5e-6 full-model
  equivalent (LoRA: 5e-5–1e-4, r 32), 1–2 epochs, seq 1024→2048 (Wolof pairs are
  longer than DOLCI average; do not truncate chosen responses).
- Gate G-DPO: Tier-2 instruction-following and FC false-positive rate must
  *improve vs Stage C* while Wolof judge score does not drop > 1 pt; else revert
  to Stage-C checkpoint (this is exactly the decision Orange faced at 0.6B —
  make it per-size, by measurement, not by family-wide rule).

### Stage E — Merge sweep (addresses D9)

- Candidates per size: {Orange v2610, Stage-C best, Stage-D best} (+ stock Qwen3
  as 4th ingredient for small sizes). Grid: Orange's (⅓,⅓,⅓) + Wolof-heavy
  (0.5 SFT / 0.25 DPO / 0.25 stock) + conservative (0.5 Orange / 0.5 Stage-D) +
  per-layer (TIES/DARE via `mergekit` if linear sweep shows interference).
- Selection: golden BPB (200 texts) → Tier-1 + Tier-2 on top-2 merges only.
  Publish the sweep table — Orange published one point; the sweep *is* the contribution.

### Stage F — Long context (addresses D7)

- Progressive: continue best checkpoint at seq 8192 (1 ep, long Wolof documents +
  conversation histories + FR/EN long-context replay), then 16384 if 8K gate passes.
- Gate: Tier-2 long-context block (needle/summarization at 4K/8K/16K) ≥ stock
  Qwen3 at same lengths; advertise only validated lengths in the model card
  (do not repeat the 40,960 claim unvalidated).

---

## 4. Training infrastructure — repo patterns, HF Jobs sizing

Repo training runs on the H200 volume host (`scripts/remote/ssh_exec.py`,
`sync_and_start.py`); HF Jobs is the fallback/overflow with live-checked flavors:

| Run | Flavor (HF Jobs) | Cost/h | Notes |
|---|---|---|---|
| 0.6B/1.7B LoRA CPT+SFT | `l40sx1` (48 GB) | $1.80 | Full LoRA fits comfortably |
| 4B pilot (all stages) | `a100-large` (80 GB) | $2.50 | Pilot budget ≈ 30–60 GPU-h ≈ $75–150 |
| 8B | `h200` (141 GB) | $5.00 | QLoRA/full-LoRA both viable |
| 14B | `h200` / `rtx-pro-6000` (96 GB, $2.75) | $2.75–5.00 | QLoRA r32; ZeRO-3 only if multi-GPU (`h200x2`) |

Tracking: Trackio Space pattern (`44_setup_trackio_space.py`) per family;
checkpoints to Hub as **private** `Tonic/Wolof-Qwen3-{size}-v2611-*` until gates
pass (repo publication habit: `06_publish_artifacts.py` after eval, not before).
Data publishing: version every mixture as a Hub dataset with its filter report
(repo `04_publish_datasets.py` pattern) — the filtered-translation quartile
report is a publishable artefact in itself.

Indicative budget (order-of-magnitude, LoRA, repo epoch counts): 4B pilot
complete ≈ $75–150; full family ≈ $600–1,200 on HF Jobs, materially less on the
existing H200 host. Human preference annotation (Stage D Wolof pairs + Tier-3
arena) is the real bottleneck — budget annotator hours before GPU hours.

---

## 5. Release definition of done (per size)

1. Tier-1 BPB < Orange v2610 on shared corpus (or parity + Tier-2 wins documented).
2. Tier-2: function-calling false-positive rate and instruction-following
   **≥ stock Qwen3 baseline** (the two Orange regressions, closed), Wolof judge
   score ≥ Orange v2610.
3. Merge sweep + golden-select tables published; ≥ 2 seeds on the pilot size (4B)
   to bound variance (fixes D10 for the recipe, if not every size).
4. GGUF Q4_K_M passes quant acceptance (Eval Plan §5.3); ONNX for ≤4B passes WebGPU smoke.
5. Model card: validated context length only, license CC-BY-NC-SA-4.0 inherited,
   data/filter reports linked, known-limitations section carried over from Orange
   cards and updated (dialect coverage, judge agreement, seed count).

## 6. Risks

| Risk | Mitigation |
|---|---|
| Starting from CC-BY-NC-SA checkpoints taints all derivatives | Accept + propagate license; if a commercial path is ever needed, re-run the same recipe from stock `Qwen/Qwen3-*` (Apache) — configs keep `base_id` swappable for exactly this |
| CPT on repo corpus shifts away from Orange's formal register | B-std vs B-ns ablation (§Stage B); ortho-delta metric in Tier-2 |
| Wolof preference data scarcity | Best-of-n + native spot-checks; cap DPO pairs at verified quality, mix FR/EN to fill volume |
| 1.7B weights never released | Fallback row in §2 table; do not block the family on it |
| Recipe wins on BPB but loses chat quality (or vice-versa) | Both gates mandatory (§5.1–5.2); golden-select uses BPB for *pool pruning only*, final selection is Tier-2 |
