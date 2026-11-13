---
title: LFM2.5 Wolof Instruct-v3
emoji: 🌊
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: Five Tonic/LFM2.5-*-Wolof-Instruct-v3 models on ZeroGPU
tags:
  - lfm2
  - lfm2.5
  - wolof
  - african-languages
  - zerogpu
  - text-generation
  - conversational
---

# LFM2.5 Wolof Instruct-v3 (ZeroGPU)

Gradio demo for the five GalsenAI **Instruct-v3** checkpoints:

| Size | Hub |
|---|---|
| 230M | `Tonic/LFM2.5-230M-Wolof-Instruct-v3` |
| 350M | `Tonic/LFM2.5-350M-Wolof-Instruct-v3` |
| 1.2B | `Tonic/LFM2.5-1.2B-Wolof-Instruct-v3` *(default)* |
| 2.6B | `Tonic/LFM2.5-2.6B-Wolof-Instruct-v3` |
| 8B-A1B MoE | `Tonic/LFM2.5-8B-A1B-Wolof-Instruct-v3` *(ZeroGPU xlarge)* |

Built like `apps/morena-wolof-zerogpu`, but loads **Transformers** `Lfm2*` /
`Lfm2Moe*` weights and each repo’s official chat template.

## Space setup

1. Create a **Gradio** Space with **ZeroGPU** hardware.
2. Upload **this entire folder** (contents at Space root).
3. Add secret **`HF_TOKEN`** with **read** access to the private `Tonic/LFM2.5-*-Wolof-Instruct-v3` repos.
4. Prefer a ZeroGPU tier that can schedule **xlarge** if you want the 8B-A1B option.

## Local run

```bash
cd apps/lfm25-wolof-instruct-v3-zerogpu
pip install -r requirements.txt
export HF_TOKEN=hf_...   # read private Tonic weights
python app.py
```

## Protocol note

Eval cards use **HF BPB** (lower is better) from `Experiments/LFM2.5-Base-1B/docs/FAMILY_EVAL.md`
(`clm-instruct-v3:*`). PPL is same-tokenizer only — do not compare 65k-ext vs 128k-ext PPL.
