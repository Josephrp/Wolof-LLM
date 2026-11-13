"""Wolof orthography training variants for Tiny Aya.

Two parallel tracks share the extended tokenizer + Trackio Space:

- **ns** — informal / phone-social spelling (`non_standard` + `mixed`)
- **std** — school / CLAD Latin (`standard` + `mixed`)

FR/EN NTP rows are kept in both CPT corpora. `not_wolof` is always dropped
from Wolof-labeled rows.
"""

from __future__ import annotations

from typing import Any

VARIANT_KEYS = ("ns", "std")

ORTHO_VARIANTS: dict[str, dict[str, Any]] = {
    "ns": {
        "key": "ns",
        "title": "non-standard (informal)",
        "keep_labels": frozenset({"non_standard", "mixed"}),
        "data_dirname": "ortho_ns_mixed_v1",
        # Legacy paths / Hub ids (already published); ns is the primary track.
        "prepared_dirname": "prepared_anchored_v1",
        "anchor_token": "<|ortho_ns|>",
        "hub_ntp": "TinyAya-Wolof-NTP-ortho_ns_mixed_v1",
        "hub_sft": "TinyAya-Wolof-SFT-ortho_ns_mixed_stratified_v1",
        "hub_prepared": "TinyAya-Wolof-SFT-prepared_anchored_v1",
        "hub_cpt_model": "TinyAya-3.35B-Wolof-CPT-v1",
        "hub_instruct_cpt": "TinyAya-3.35B-Wolof-Instruct-v1",
        "hub_instruct_earth": "TinyAya-3.35B-Wolof-InstructEarth-v1",
        "hub_instruct_global": "TinyAya-3.35B-Wolof-InstructGlobal-v1",
        "hub_dpo": "TinyAya-3.35B-Wolof-DPO-v1",
        "cpt_run": "v1_ntp70_ortho_lora",
        "cpt_smoke_run": "tinyaya_3b35_cpt_smoke",
        "merged_cpt": "tinyaya-3.35b-cpt-v1-merged",
        "merged_earth": "tinyaya-3.35b-instruct-from-earth-merged",
        "merged_global": "tinyaya-3.35b-instruct-from-global-merged",
        "merged_instruct": "tinyaya-3.35b-instruct-from-cpt-merged",
        "merged_best": "tinyaya-3.35b-instruct-best-merged",
        "merged_dpo": "tinyaya-3.35b-dpo-v1-merged",
        "sft_run_prefix": "3b35",
        "stage_suffix": "",
    },
    "std": {
        "key": "std",
        "title": "standard (CLAD / school)",
        "keep_labels": frozenset({"standard", "mixed"}),
        "data_dirname": "ortho_std_mixed_v1",
        "prepared_dirname": "prepared_anchored_std_v1",
        "anchor_token": "<|ortho_std|>",
        "hub_ntp": "TinyAya-Wolof-NTP-ortho_std_mixed_v1",
        "hub_sft": "TinyAya-Wolof-SFT-ortho_std_mixed_stratified_v1",
        "hub_prepared": "TinyAya-Wolof-SFT-prepared_anchored_std_v1",
        "hub_cpt_model": "TinyAya-3.35B-Wolof-CPT-std-v1",
        "hub_instruct_cpt": "TinyAya-3.35B-Wolof-Instruct-std-v1",
        "hub_instruct_earth": "TinyAya-3.35B-Wolof-InstructEarth-std-v1",
        "hub_instruct_global": "TinyAya-3.35B-Wolof-InstructGlobal-std-v1",
        "hub_dpo": "TinyAya-3.35B-Wolof-DPO-std-v1",
        "cpt_run": "v1_ntp70_ortho_std_lora",
        "cpt_smoke_run": "tinyaya_3b35_cpt_std_smoke",
        "merged_cpt": "tinyaya-3.35b-cpt-std-v1-merged",
        "merged_earth": "tinyaya-3.35b-instruct-from-earth-std-merged",
        "merged_global": "tinyaya-3.35b-instruct-from-global-std-merged",
        "merged_instruct": "tinyaya-3.35b-instruct-from-cpt-std-merged",
        "merged_best": "tinyaya-3.35b-instruct-best-std-merged",
        "merged_dpo": "tinyaya-3.35b-dpo-std-v1-merged",
        "sft_run_prefix": "3b35_std",
        "stage_suffix": "_std",
    },
}


def get_variant(key: str) -> dict[str, Any]:
    if key not in ORTHO_VARIANTS:
        raise KeyError(f"unknown ortho variant {key!r}; expected one of {VARIANT_KEYS}")
    return ORTHO_VARIANTS[key]


def resolve_variants(spec: str) -> list[str]:
    """Parse CLI --ortho ns|std|both → ordered variant keys."""
    value = (spec or "both").strip().lower()
    if value in {"both", "all"}:
        return list(VARIANT_KEYS)
    if value in ORTHO_VARIANTS:
        return [value]
    raise ValueError(f"invalid --ortho {spec!r}; use ns, std, or both")
