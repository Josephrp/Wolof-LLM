# Tiny Aya Wolof Family — eval table

Updated: `2026-10-09T16:06:51.921443+00:00`

License: **CC-BY-NC-4.0** + Cohere Labs AUP. Gated stock checkpoints.

## Protocol

- **Gold CLM** = `Evaluation/scripts/evaluate_clm.py` (same as LFM/MORENA):
  verbalized BPB + perplexity + BLEU + chrF on ortho NTP test CSV.
- Primary corpus: ns → `corpus_test_clm_ortho_ns_mixed*.csv`; std → TinyAya std export.
- SFT task evals: `results/sft/eval_*/evaluation.json` (val mean loss).
- Hub: `metrics_clm.json` + composed `metrics.json` / `model-index` via `56_publish_hub_metrics.py`.

## HF CLM (evaluate_clm.py)

| checkpoint | BPB ↓ | BLEU | chrF | PPL | sft_loss ↓ | notes |
|---|---:|---:|---:|---:|---:|---|
| stock_base | 2.4702 | 3.4600 | 20.42 | 108.55 | — | ns Base |
| stock_earth | 1.8584 | 2.6500 | 20.94 | 34.00 | — | ns Instruct |
| stock_global | 1.8787 | 2.4500 | 21.10 | 35.33 | — | ns Instruct |
| Tonic/TinyAya-3.35B-Wolof-CPT-v1 | 1.8884 | 1.6800 | 14.2600 | 190.00 | — | ns CPT |
| Tonic/TinyAya-3.35B-Wolof-CPT-std-v1 | 1.7074 | 0.4200 | 8.4400 | 118.65 | — | std CPT |
| Tonic/TinyAya-3.35B-Wolof-InstructEarth-v1 | 2.0417 | 2.2900 | 15.3900 | 49.00 | 3.3983 | ns InstructEarth |
| Tonic/TinyAya-3.35B-Wolof-InstructGlobal-v1 | 2.0698 | 2.3700 | 18.0300 | 51.69 | 3.1509 | ns InstructGlobal |
| Tonic/TinyAya-3.35B-Wolof-InstructEarth-std-v1 | 1.9331 | 1.5500 | 12.3100 | 38.77 | 3.1945 | std InstructEarth |
| Tonic/TinyAya-3.35B-Wolof-InstructGlobal-std-v1 | 1.9663 | 1.6100 | 11.6800 | 41.28 | 3.4330 | std InstructGlobal |
| Tonic/TinyAya-3.35B-Wolof-DPO-v1 | 2.0764 | 2.3000 | 18.0300 | 52.35 | 3.1731 | ns DPO |
| Tonic/TinyAya-3.35B-Wolof-DPO-std-v1 | 1.9332 | 1.5000 | 11.9600 | 38.77 | 3.1873 | std DPO |

from_cpt Instruct skipped (vocab layout).

## GGUF Q4_K_M (gguf-llama-cpp; imatrix-calibrated; not HF-BPB-equivalent)

| Job | PPL | BLEU | chrF |
|---|---:|---:|---:|
| gguf-q4:CPT:ns | 92.3399 | 2.88 | 16.99 |
| gguf-q4:CPT:std | 63.7353 | 2.89 | 18.62 |
| gguf-q4:InstructEarth:ns | 51.8923 | 1.94 | 16.36 |
| gguf-q4:InstructEarth:std | 33.1479 | 2.85 | 20.14 |
| gguf-q4:InstructGlobal:ns | 52.708 | 2.23 | 16.52 |
| gguf-q4:InstructGlobal:std | 34.8773 | 3.19 | 19.75 |
| gguf-q4:DPO:ns | 53.9919 | 2.05 | 17.2 |
| gguf-q4:DPO:std | 33.1612 | 2.41 | 19.67 |
