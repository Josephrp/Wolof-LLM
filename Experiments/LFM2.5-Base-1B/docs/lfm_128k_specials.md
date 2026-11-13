# LFM2.5 128k family — special id layout

Inspected from Hub configs (public Base cards), 2026-09-16. Re-verified from
`LiquidAI/LFM2.5-2.6B-Base/config.json` and `tokenizer_config.json`:
architecture `Lfm2ForCausalLM`, `vocab_size=128000`, the IDs below,
`tie_word_embeddings=true`, EOS `<|im_end|>`, and tokenization-space cleanup disabled.
The implementation additionally refuses to append unless runtime tokenizer IDs are
contiguous through `len(tokenizer)-1`; that runtime proof remains part of T-128.

## Contrast

| | **65k** (`LFM2.5-1.2B-Base`) | **128k** (`LFM2.5-2.6B-Base`) |
|---|---|---|
| `vocab_size` | 65 536 | 128 000 |
| `pad_token_id` | **0** | **124 893** |
| `bos_token_id` | **1** | **124 894** |
| `eos_token_id` | **7** (`<\|im_end\|>`) | **124 900** (`<\|im_end\|>`) |
| tying flag | `tie_embedding: true` | `tie_word_embeddings: true` |
| Specials location | **Low ids** (0–16+) | **High ids** near vocab end (~124893+) |

## Implications for tokenizer extension

1. **Do not** assume eos=7 on 128k. Packing / journal / asserts must read `eos_token_id` from config (or YAML `eos_token_id: 124900`).
2. Specials sit at **high** IDs (unlike 65k), but not at the final ID. The implemented path appends only after the runtime maximum ID and never shifts existing IDs.
3. Any sparse/non-contiguous runtime layout, special-ID mismatch, or `len(tokenizer)` inconsistency is a hard failure. Do not fall back to grow-and-shift automatically.
4. Chat strings still use `<|im_start|>` / `<|im_end|>`; only **ids** differ.

## Next

- P4.5.2: implement `--family 128k` path using these defaults once `len(tok)` vs specials layout is confirmed on a local download.
- P10.3.x: separate fertility/BPB interpretation from 65k runs.
