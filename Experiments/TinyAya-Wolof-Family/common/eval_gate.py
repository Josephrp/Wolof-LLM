"""Full CLM eval gate for Tiny Aya Hub publish / resume.

Gold protocol matches LFM/MORENA: Evaluation/scripts/evaluate_clm.py must
produce BPB + perplexity + BLEU + chrF before model metrics are Hub-ready.
Lightweight 04_evaluate_bpb.py JSON (bpb.all only) is NOT sufficient.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CLM_KEYS = ("BPB", "BLEU", "chrF", "perplexity")


def is_full_clm(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in REQUIRED_CLM_KEYS:
        value = payload.get(key)
        if value is None:
            return False
    return True


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def is_full_clm_file(path: Path | None) -> bool:
    return is_full_clm(load_json(path))


def expected_clm_paths_for_ortho(ortho: str) -> list[Path]:
    """Publish-path + gold-path CLM JSONs that should exist after eval for an ortho."""
    bpb = ROOT / "results" / "bpb"
    roles = [
        ("cpt", f"cpt_{ortho}_3b35.json"),
        ("from_earth", f"instruct_from_earth_{ortho}_3b35.json"),
        ("from_global", f"instruct_from_global_{ortho}_3b35.json"),
        ("dpo", f"dpo_{ortho}_3b35.json"),
    ]
    paths: list[Path] = []
    for role, legacy in roles:
        paths.append(bpb / legacy)
        paths.append(bpb / f"clm_{role}_{ortho}_3b35.json")
        if role.startswith("from_"):
            paths.append(bpb / f"clm_{role}_{ortho}_3b35.json")
    # Prefer unique
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def candidate_clm_files(ortho: str, name: str) -> list[Path]:
    """Publish + gold filenames accepted as full CLM for an artifact."""
    bpb = ROOT / "results" / "bpb"
    gold_role = {
        "cpt": "cpt",
        "instruct_from_earth": "from_earth",
        "instruct_from_global": "from_global",
        "dpo": "dpo",
    }[name]
    legacy = {
        "cpt": f"cpt_{ortho}_3b35.json",
        "instruct_from_earth": f"instruct_from_earth_{ortho}_3b35.json",
        "instruct_from_global": f"instruct_from_global_{ortho}_3b35.json",
        "dpo": f"dpo_{ortho}_3b35.json",
    }[name]
    return [
        bpb / legacy,
        bpb / f"clm_{gold_role}_{ortho}_3b35.json",
        bpb / f"clm_{name}_{ortho}_3b35.json",
    ]


def ortho_eval_complete(ortho: str, *, models_root: Path | None = None) -> tuple[bool, list[str]]:
    """Return (ok, missing_labels). Skips artifacts whose model dir is absent."""
    models_root = models_root or (ROOT / "models")
    variant_models = {
        "ns": {
            "cpt": "tinyaya-3.35b-cpt-v1-merged",
            "instruct_from_earth": "tinyaya-3.35b-instruct-from-earth-merged",
            "instruct_from_global": "tinyaya-3.35b-instruct-from-global-merged",
            "dpo": "tinyaya-3.35b-dpo-v1-merged",
        },
        "std": {
            "cpt": "tinyaya-3.35b-cpt-std-v1-merged",
            "instruct_from_earth": "tinyaya-3.35b-instruct-from-earth-std-merged",
            "instruct_from_global": "tinyaya-3.35b-instruct-from-global-std-merged",
            "dpo": "tinyaya-3.35b-dpo-std-v1-merged",
        },
    }[ortho]
    missing: list[str] = []
    for name, model_name in variant_models.items():
        model_dir = models_root / model_name
        if not model_dir.is_dir():
            continue
        if any(is_full_clm_file(path) for path in candidate_clm_files(ortho, name)):
            continue
        missing.append(f"{ortho}:{name}")
    return (not missing, missing)


def family_eval_complete() -> tuple[bool, list[str]]:
    missing: list[str] = []
    for ortho in ("ns", "std"):
        ok, miss = ortho_eval_complete(ortho)
        if not ok:
            missing.extend(miss)
    return (not missing, missing)
