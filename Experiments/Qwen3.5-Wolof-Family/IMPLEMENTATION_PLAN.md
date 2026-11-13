# Qwen3.5 Wolof Family — Implementation Plan

See also: `.cursor/plans/qwen3.5_wolof_family_98b6ec94.plan.md`

## Locked decisions

| Topic | Choice |
|-------|--------|
| Root | `Experiments/Qwen3.5-Wolof-Family/` (distinct) |
| Sizes | 0.8B / 2B / 4B / 9B Base |
| MoE | Excluded (35B-A3B ~36B) |
| Ortho | Drop `standard` (formal); keep `non_standard`+`mixed` |
| CPT | 3 epochs on NTP 70/20/10 after ortho filter |
| SFT | 3 epochs ×2: stock Instruct + CPT-merged Base |
| Anchors | Qwen chat template + `<|lang_wo|>` `<|ortho_ns|>` at data stage |
| Vocab | **LFM-style** pretok-chunk append into `tokenizer.json` BPE vocab (not `add_tokens`), specials preserved (`eos=248044`), per-token mean-pool init, fertility gate before CPT |
| DPO | LFM-style `prompt/chosen/rejected` after Instruct-from-CPT |
| Hub | Private `Tonic/Qwen3.5-Wolof-*` datasets + models |

## Remote

`/mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/Qwen3.5-Wolof-Family`

Helper: `scripts/remote/ssh_exec.py`
