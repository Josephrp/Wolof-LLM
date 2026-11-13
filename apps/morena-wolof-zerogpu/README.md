---
title: MORENA Wolof Continuum
emoji: 🌍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: GalsenAI MORENA Wolof CPT/Distill/SFT family on ZeroGPU
tags:
  - morena
  - wolof
  - african-languages
  - zerogpu
  - text-generation
---

# MORENA Wolof continuum (ZeroGPU)

Gradio demo for the **GalsenAI MORENA Wolof** family published under `Tonic/MORENA-*`.

## What it shows

- Dropdown across Distill / CPT / Instruct / stock baseline checkpoints
- **Mode-aware prompting** with Auto / Chat / Completion override
- System prompts restored (chat: folded into user turn; completion: plain leading context)
- Sampler: official temp/top-p defaults plus restored top-k + mild repetition penalty
- Examples with Wolof system strings + CLM / SFT prompts
- Per-model eval cards (BPB gold, BLEU, chrF) with honest training implications

## Space setup

1. Create a **Gradio** Space with **ZeroGPU** hardware.
2. Upload **this entire folder** (contents at Space root).
3. Add secret **`HF_TOKEN`** with read access to private `Tonic/MORENA-*` repos
   (stock baseline works without it).
4. Optional: raise ZeroGPU size to `xlarge` in `@spaces.GPU` if 1.5B swaps are tight.

## Local run

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...   # for private Tonic weights
python app.py
```

## Protocol note

Eval scores use **`morena-native` BPB** (lower is better). Distill ≈ stock BPB is the
success case; CPT is mixed; Instruct SFT in this continuum collapsed on a toy SFT mix
and is shown for transparency.
