# LFM2.5 8B-A1B MoE LoRA gate

The 8B-A1B recipe is a gated placeholder until `06_probe_modules.py` is run against `LiquidAI/LFM2.5-8B-A1B-Base`.

Required probe evidence:

1. Record all attention, router, shared-expert, and routed-expert module suffixes.
2. Confirm whether PEFT suffix matching selects every expert or only shared projections.
3. Start with attention-only LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
4. Add expert projections only after reporting trainable parameter count and verifying router parameters remain frozen intentionally.
5. Use the native 128k tokenizer with EOS 124900 unless T-128 has passed for an extended tokenizer.
6. Run the 10-step tiny-batch smoke before mini CPT. A dry-run is not a training result.

Probe command:

```bash
uv run python scripts/clm/06_probe_modules.py --base-id LiquidAI/LFM2.5-8B-A1B-Base --output docs/lfm_8b_moe_module_names.txt
```
