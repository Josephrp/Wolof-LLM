# MORENA Wolof continuum

Dense Llama-style African foundation models from [vamboai/morena](https://huggingface.co/collections/vamboai/morena),
adapted for Wolof with the **native Muon + FSDP2 trainer** (not TRL / HF Trainer).

## Layout

- `vendor/modeling_morena.py` — pinned production trainer + `Transformer`
- `scripts/` — env, data packing, CPT/SFT wrappers, prune/distill, publication
- `configs/` — CPT/SFT/eval/family YAML+JSON
- `docs/FAMILY_EVAL.md` — family metrics table

## Critical rules

1. Load with `Transformer` + `tokenizer.json` (HuggingFace `tokenizers`), not `AutoModelForCausalLM`.
2. Instruct chat markers are `<reserved_0>` / `<reserved_1>` only.
3. Cross-family ranking uses **BPB** (`protocol: morena-native`); never compare PPL to LFM.
4. Train with `torchrun … vendor/modeling_morena.py` (or `scripts/cpt/03_train_morena.py`).

## Quick start

```bash
# Env gate (GPU host)
python scripts/env/00_verify_env.py

# Stock eval
python ../../Evaluation/scripts/evaluate_morena.py --config configs/eval/stock_1.5b_base.yaml

# Family orchestrator (durable)
python scripts/publication/50_run_morena_family.py --resume
```
