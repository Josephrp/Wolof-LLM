---
title: Qwen3.5-9B Wolof
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: GalsenAI Qwen3.5-9B Wolof Instruct + CPT on ZeroGPU
startup_duration_timeout: 1h
tags:
  - qwen
  - qwen3.5
  - wolof
  - african-languages
  - zerogpu
  - text-generation
  - conversational
---

# Qwen3.5-9B Wolof (ZeroGPU)

Gradio demo for the largest GalsenAI **Qwen3.5 Wolof** checkpoints:

| Role | Hub | BPB ↓ | Mode |
|---|---|---:|---|
| **Instruct-v1** *(default)* | `Tonic/Qwen3.5-9B-Wolof-Instruct-v1` | 1.00 | chat |
| CPT-v1 *(best LM)* | `Tonic/Qwen3.5-9B-Wolof-CPT-v1` | 0.85 | continuation |

Stock ext-init baseline BPB: **1.84**. Protocol: `evaluate_clm.py` on ortho-filtered Wolof CLM test (`FAMILY_EVAL.md`).

Mirrors `Tonic/LFM2.5-Wolof` / `apps/lfm25-wolof-instruct-v3-zerogpu`: Transformers load, eager `.to("cuda")`, `@spaces.GPU(size="large")`, dynamic duration.

## Space setup

1. Create a **Gradio** Space with **ZeroGPU** (`zero-a10g`) hardware.
2. Upload **this entire folder** (contents at Space root).
3. First boot downloads ~18 GB bf16 weights — `startup_duration_timeout: 1h` is set for cold start.
4. Demo injects training anchors (`<|lang_wo|>` `<|ortho_ns|>`) automatically; defaults use top-p **0.99**, top-k **99**, repetition penalty **1.5**.

## Local run

```bash
cd apps/qwen35-9b-wolof-zerogpu
pip install -r requirements.txt spaces gradio
export HF_TOKEN=hf_...   # read private Tonic weights
python app.py
```

## Notes

- Requires **`transformers>=5.19`** (`Qwen3_5ForCausalLM` / `qwen3_5_text`).
- Prefer **Instruct** for chat demos; switch to **CPT** for anchored continuation (`<|lang_wo|> <|ortho_ns|>`).
- Do not compare PPL across tokenizer families — use **BPB**.
