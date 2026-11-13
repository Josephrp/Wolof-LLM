# LFM2.5 Wolof GPU smoke — 2026-09-16

Authorized smoke artifact; not a trained model. No quality claim.

- VM: Ubuntu 24.04, Python 3.12.3; ephemeral storage 2.9T total / 1.2T free.
- GPU: NVIDIA H200, 143771 MiB; driver 570.211.01; CUDA 12.8; bf16 supported.
- Environment: uv 0.12.15, Python 3.12.8, torch 2.11.0+cu128, transformers 5.17.0.
- Validation: uv sync PASS; pytest 22 PASS; Ruff PASS; compileall PASS; CUDA env verifier PASS.
- 350M probe and 6-token forward PASS; finite logits; peak 710.88 MiB allocated.
- 1.2B config-only module probe PASS; full weights and forward deliberately not run.
- Temporary ext32 tokenizer pipeline PASS: eos=7, append-only, round trip 500/500, mean-pool exercised. Norm ratio 0.714 was above the script's expected <0.7 threshold; full tokenizer gate remains pending.
- Default-tokenizer LoRA: 2 steps PASS; losses [[1, 8.87620735168457], [2, 7.468776226043701]]; peak 976.63 MiB allocated / 1902.0 MiB reserved.
- Ext32 LoRA with modules_to_save: 1 step PASS; finite loss 8.892138481140137; peak 1746.5 MiB allocated / 2074.0 MiB reserved.
- Hugging Face: authenticated as Tonic, but token role is read. Private repo creation/upload/readback were not attempted.

Completed: G-UV and tiny GPU computation gates. Pending: full 1.2B forward, canonical ext4000/fertility gate, canonical 10-step default+ext smokes, production/quality gates, and HF publishing smoke with a write-scoped token.
