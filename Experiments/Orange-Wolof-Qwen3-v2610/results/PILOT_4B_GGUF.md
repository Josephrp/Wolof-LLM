# 4B GGUF pilot — results (2026-11-18)

First end-to-end run of the eval-gated quantization pipeline: convert →
evaluate every artifact → gate against the evaluated F16 master → publish
private with a card generated from the measurements.

- **HF Job**: `6afcdaad4643bac7ac62f263` (`job_gguf_family.py`, branch
  `orange-wolof-qwen3-v2610`, llama.cpp **b8999**, flavor `cpu-upgrade`)
- **Source**: `Orange/Wolof-Qwen3-4B-v2610` @ `64314c62da`
- **Published**: [`Tonic/Wolof-Qwen3-4B-v2610-GGUF`](https://huggingface.co/Tonic/Wolof-Qwen3-4B-v2610-GGUF)
  — **private**, revision `f54361ca7bc167bc27ba35e8ab4af4c637cb21a1`,
  post-upload smoke test passed (Q4_K_M generated 63 chars)

## Protocol (pilot subset)

GGUF protocol (`Evaluation/scripts/evaluate_gguf.py`, llama.cpp, CPU):
perplexity via `llama-perplexity` (ctx 1024) and chrF via `llama-completion`
on the first **128 texts** of the verbalized `corpus_test_clm`; generation
sample = 32 prompts (seed 0, temp 0) restricted to texts of ≤200 words
(`bleu_max_text_words`; an uncapped draw contained a ~10k-word text and
collapsed chrF to 7.06 for the F16 master — a measurement artifact, fixed
before this run). **This is a subset pilot: a full-corpus (4,282-text)
evaluation is required before any public promotion.**

## Measured results (same subset & protocol for every row)

| Artifact | PPL | ΔPPL vs F16 | chrF | ΔchrF vs F16 | Verdict |
|---|---:|---:|---:|---:|---|
| F16 (reference, not uploaded) | 10.4037 | — | 20.41 | — | reference |
| Q8_0 | 10.4075 | +0.04% | 20.35 | 0.06 pt | **pass** |
| Q4_K_M plain | 10.7722 | +3.54% | 19.80 | 0.61 pt | (superseded) |
| Q4_K_M + Wolof imatrix | 10.6517 | +2.38% | 19.88 | 0.53 pt | **pass** |
| Q4_0 | 11.1572 | +7.24% | 19.89 | 0.52 pt | compat-only |

Tier-1 HF reference (full 4,282 texts, `evaluate_clm.py`): BPB 1.3889,
PPL 12.36, BLEU 5.55, chrF 24.22. The F16 GGUF sanity ratio
(10.40/12.36 = 0.84×) is inside the ≤1.5× band. Absolute GGUF chrF is not
comparable to Tier-1 chrF (different protocol/subset); only the paired
deltas above are gated.

## Imatrix decision: **adopted**

The Wolof-corpus imatrix Q4_K_M beat the plain build on both metrics
(chrF 19.88 vs 19.80; PPL 10.6517 vs 10.7722, the latter computed over all
128 texts). The imatrix file is the canonical `Q4_K_M` in the published
repo (recorded in its `gguf_meta.json`). Gains are small; adopt imatrix
for the remaining sizes, re-deciding per size with the same A/B protocol.

## Job status note

The HF Job shows `ERROR` although the pilot succeeded: after publishing
the 4B repo, the old substring `ONLY` filter (`"4b" in "…14b…"`) also
selected the 14B model, which then failed on disk space (50 GB flavor).
Fixed in all three family jobs (token-based matching); no eval was or
will be re-run for this — the numbers above are final for the pilot.

## Remaining rollout (not part of this pilot)

Tier-1 eval + GGUF for 0.6B/8B/14B (14B needs `cpu-xl` disk), ONNX for
0.6B/4B, full-corpus GGUF re-eval before any `--promote-public`. The 1.7B
remains blocked: the Orange repo contains no weights.
