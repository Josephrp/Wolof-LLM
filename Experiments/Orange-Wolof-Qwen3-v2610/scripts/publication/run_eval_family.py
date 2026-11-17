#!/usr/bin/env python3
"""Local/H200 Tier-1 CLM eval matrix for Orange Wolof-Qwen3 v2610 + stock baselines.

Pattern: Qwen3.5-Wolof-Family/scripts/publication/55_run_eval_family.py.
Runs Evaluation/scripts/evaluate_clm.py with the YAML configs in
configs/eval/clm/, writing Evaluation/metrics/clm/*.json, resumable via --resume.
The 1.7B Orange config is skipped with a BLOCKED note (no weights on Hub, Gate 0).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
EVAL_ROOT = REPO / "Evaluation"
CFG_DIR = ROOT / "configs" / "eval" / "clm"

STEMS = [
    "orange_wolof_qwen3_06b", "qwen3_06b",
    "orange_wolof_qwen3_4b", "qwen3_4b",
    "orange_wolof_qwen3_8b", "qwen3_8b",
    "orange_wolof_qwen3_14b", "qwen3_14b",
]
BLOCKED = {"orange_wolof_qwen3_17b": "no weights on Hub (tokenizer/config only) — Gate 0"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier-1 CLM matrix (local GPU host)")
    parser.add_argument("--only", default=None, help="substring filter on config stem")
    parser.add_argument("--resume", action="store_true", help="skip stems with existing metrics JSON")
    parser.add_argument("--smoke", action="store_true", help="note only: shrink via job script SMOKE=1; local smoke = first stem")
    args = parser.parse_args()

    state_path = ROOT / "results" / "clm_eval_state.json"
    state = {"updated_at": utc_now(), "jobs": {}}
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    for stem in STEMS:
        if args.only and args.only.lower() not in stem:
            continue
        out_json = EVAL_ROOT / "metrics" / "clm" / f"{stem}.json"
        if args.resume and out_json.is_file():
            print(f"skip {stem} (exists)", flush=True)
            continue
        print(f"+ evaluate_clm {stem}", flush=True)
        try:
            # Materialize a temp config with absolute paths: evaluate_clm.py
            # resolves relative paths against Evaluation/configs/, not our tree.
            cfg = yaml.safe_load((CFG_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
            cfg["corpus_csv"] = str(EVAL_ROOT / "data" / "corpus_test_clm.csv")
            cfg["output_json"] = str(out_json)
            if args.smoke:
                cfg["n_texts"] = 32
                cfg["bleu_n_texts"] = 16
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
                tmp_cfg = handle.name
            subprocess.run(
                [sys.executable, str(EVAL_ROOT / "scripts" / "evaluate_clm.py"),
                 "--config", tmp_cfg],
                cwd=str(EVAL_ROOT), check=True)
            state["jobs"][stem] = {"status": "complete", "at": utc_now()}
        except Exception as exc:  # noqa: BLE001
            state["jobs"][stem] = {"status": "failed", "error": str(exc)[:300], "at": utc_now()}
            print(f"EVAL_FAIL {stem}: {exc}", flush=True)
        state["updated_at"] = utc_now()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    for stem, reason in BLOCKED.items():
        if not args.only or args.only.lower() in stem:
            print(f"BLOCKED {stem}: {reason}", flush=True)
            state["jobs"][stem] = {"status": "blocked", "reason": reason, "at": utc_now()}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print("CLM_EVAL_FAMILY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
