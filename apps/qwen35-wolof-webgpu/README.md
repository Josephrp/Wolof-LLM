---
title: Qwen3.5 Wolof WebGPU
emoji: 🌍
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: other
short_description: In-browser Wolof chat via Transformers.js WebGPU
models:
- Tonic/Qwen3.5-08B-Wolof-Instruct-v1-ONNX
- Tonic/Qwen3.5-2B-Wolof-Instruct-v1-ONNX
- Tonic/Qwen3.5-4B-Wolof-Instruct-v1-ONNX
tags:
- wolof
- qwen3.5
- transformers.js
- webgpu
- african-languages
---

# Qwen3.5 Wolof WebGPU

Static **browser** demo (no Space GPU). Runs GalsenAI **Instruct** checkpoints locally with
Transformers.js + WebGPU. Default model: **0.8B**.

Requires ONNX sidecars published by `Experiments/Qwen3.5-Wolof-Family` (`*-Instruct-v1-ONNX`).
Training anchors (`<|lang_wo|>` `<|ortho_ns|>`) are injected automatically.

Chrome / Edge with WebGPU recommended. First load downloads quantized ONNX weights into the browser cache.
