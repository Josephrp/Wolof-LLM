# Publication status — Orange Wolof-Qwen3 v2610 (2026-11-18)

All target private repos under `Tonic/` are live (GGUF + ONNX for 0.6B/4B/8B/14B), including ONNX external-data weights. 1.7B remains BLOCKED (no Orange weights).

## Private Hub repos

| Repo | Private | Uploaded artifacts | Notes |
|---|---|---|---|
| [Tonic/Wolof-Qwen3-0.6B-v2610-GGUF](https://huggingface.co/Tonic/Wolof-Qwen3-0.6B-v2610-GGUF) | yes | **Q8_0** only | Q5/Q4_K_M/Q4_0 failed ΔPPL gate on 128-text subset |
| [Tonic/Wolof-Qwen3-4B-v2610-GGUF](https://huggingface.co/Tonic/Wolof-Qwen3-4B-v2610-GGUF) | yes | Q8_0, Q4_K_M (imatrix), Q4_0 | Pilot; recommended **Q4_K_M** |
| [Tonic/Wolof-Qwen3-8B-v2610-GGUF](https://huggingface.co/Tonic/Wolof-Qwen3-8B-v2610-GGUF) | yes | **Q4_K_M** (imatrix) only | Q8/Q6/Q5/Q4_0 failed ΔchrF on 64-text/24-prompt subset (noisy); job ERROR after publish during smoke download — repo is fine |
| [Tonic/Wolof-Qwen3-14B-v2610-GGUF](https://huggingface.co/Tonic/Wolof-Qwen3-14B-v2610-GGUF) | yes | Q8_0, Q6_K, Q4_K_M (imatrix), Q4_0 | Q5_K_M failed ΔchrF |
| [Tonic/Wolof-Qwen3-0.6B-v2610-ONNX](https://huggingface.co/Tonic/Wolof-Qwen3-0.6B-v2610-ONNX) | yes | fp16, q4, q4f16 (+ `_data`) | re-pub `6afd7ac6…cb13`; q4/q4f16 ~650MB each |
| [Tonic/Wolof-Qwen3-4B-v2610-ONNX](https://huggingface.co/Tonic/Wolof-Qwen3-4B-v2610-ONNX) | yes | fp16, q4, q4f16 (+ `_data`) | re-pub `6afd7ac6…cb11`; fp16 ~9.6GB, q4 ~3.1GB |
| [Tonic/Wolof-Qwen3-8B-v2610-ONNX](https://huggingface.co/Tonic/Wolof-Qwen3-8B-v2610-ONNX) | yes | fp16, q4, q4f16 (+ `_data`) | re-pub `6afd7ae9…cb21`; fp16 ~17.6GB, q4 ~5.5GB |
| [Tonic/Wolof-Qwen3-14B-v2610-ONNX](https://huggingface.co/Tonic/Wolof-Qwen3-14B-v2610-ONNX) | yes | fp16, q4, q4f16 (+ `_data`) | re-pub `6afd7ae9…4310`; fp16 ~31.6GB, q4 ~9.5GB |## Tier-1 HF CLM (full 4,282 texts) — gate prerequisite

| Model | BPB | PPL | BLEU | chrF | src_sha |
|---|---:|---:|---:|---:|---|
| Orange 0.6B | 1.5282 | 15.91 | 3.51 | 20.70 | a296360960 |
| Orange 4B | 1.3889 | 12.36 | 5.55 | 24.22 | 64314c62da |
| Orange 8B | 1.3656 | 11.85 | 5.54 | 23.29 | eacd4f3428 |
| Orange 14B | 1.3381 | 11.28 | 5.90 | 23.19 | cc3d985809 |

Dataset: `Tonic/Wolof-Qwen3-v2610-eval` (also has stock Qwen3 baselines).

## Flavors used (cheapest that fit)

| Work | Flavor | $/h |
|---|---|---:|
| Tier-1 0.6B | t4-small | 0.40 |
| Tier-1 8B | a10g-large | 1.50 |
| GGUF 0.6B | cpu-basic | 0.01 |
| GGUF 4B | cpu-upgrade | 0.03 |
| GGUF 8B | cpu-upgrade | 0.03 |
| GGUF 14B | cpu-xl | 1.00 |
| ONNX 0.6B | cpu-basic | 0.01 |
| ONNX 4B | cpu-xl | 1.00 |
| ONNX 8B | cpu-xl | 1.00 |
| ONNX 14B | cpu-xl | 1.00 |

## Follow-ups (not blocking private publish)

1. Full-corpus GGUF re-eval before any `--promote-public` (subset evals are noisy; 0.6B/8B lost mid quants to ΔchrF/ΔPPL).
2. Soften post-upload smoke so a smoke download failure does not mark the job ERROR after a successful private publish (8B).
3. 1.7B: wait for Orange weights, then same pipeline.
4. Push local `publish_quant.py` / `eval_onnx_dtype.py` sidecar fixes to `orange-wolof-qwen3-v2610` (jobs currently self-patch the clone).
