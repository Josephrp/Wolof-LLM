#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.57",
#     "accelerate>=1.0.0",
#     "pandas",
#     "pyyaml",
#     "sacrebleu",
#     "tqdm",
#     "huggingface_hub>=0.26.0",
# ]
# ///
"""HF Jobs (GPU): Tier-1 CLM eval (BPB/PPL/BLEU/chrF) for Orange Wolof-Qwen3 v2610 + stock Qwen3 baselines.

Reuses the repo protocol verbatim: clones Wolof-LLM, then runs
Evaluation/scripts/evaluate_clm.py per model with the configs in
Experiments/Orange-Wolof-Qwen3-v2610/configs/eval/clm/. Metrics are uploaded to
the dataset repo Tonic/Wolof-Qwen3-v2610-eval.

Launch (from repo root, token as secret — never baked in):
  hf jobs uv run --flavor h200 --timeout 4h --secrets HF_TOKEN \
    --env ONLY=14b Experiments/Orange-Wolof-Qwen3-v2610/scripts/jobs/job_eval_clm_family.py
Flavors by size: 14b -> h200/a100-large, 8b/4b -> l40sx1, 06b -> l4x1.
Env: ONLY (substring filter, e.g. "14b" or "orange"), SMOKE=1 (32 texts/16 BLEU pairs).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

REPO_URL = os.environ.get("WOLOF_REPO_URL", "https://github.com/Josephrp/Wolof-LLM.git")
REPO_REF = os.environ.get("WOLOF_REPO_REF", "main")
WORK = Path("/tmp/wolof-llm")
OUT = Path("/tmp/clm_metrics")
EVAL_DATASET = os.environ.get("EVAL_DATASET", "Tonic/Wolof-Qwen3-v2610-eval")

# (config stem, needs_gpu_note) — 1.7B Orange is BLOCKED (no weights on Hub, Gate 0).
CONFIGS = [
    "orange_wolof_qwen3_06b", "qwen3_06b",
    "orange_wolof_qwen3_4b", "qwen3_4b",
    "orange_wolof_qwen3_8b", "qwen3_8b",
    "orange_wolof_qwen3_14b", "qwen3_14b",
]
# Pinned Orange revisions (configs/family.yaml) stamped into Tier-1 JSONs so the
# publish gate in publish_quant.py can SHA-match eval <-> source checkpoint.
SRC_SHA = {
    "orange_wolof_qwen3_06b": "a296360960",
    "orange_wolof_qwen3_4b": "64314c62da",
    "orange_wolof_qwen3_8b": "eacd4f3428",
    "orange_wolof_qwen3_14b": "cc3d985809",
}


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required (pass with --secrets HF_TOKEN)")
    smoke = os.environ.get("SMOKE", "") not in ("", "0")
    only = os.environ.get("ONLY", "").strip().lower()

    if not WORK.is_dir():
        run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(WORK)])
    cfg_dir = WORK / "Experiments/Orange-Wolof-Qwen3-v2610/configs/eval/clm"
    eval_dir = WORK / "Evaluation"
    OUT.mkdir(parents=True, exist_ok=True)

    done: list[str] = []
    for stem in CONFIGS:
        if only and only not in stem:
            continue
        # Always materialize a temp config with absolute paths: evaluate_clm.py
        # resolves relative paths against Evaluation/configs/, but our configs
        # live in Experiments/Orange-.../configs/eval/clm/ (path mismatch bug
        # found by the first pilot job, 2026-11-17).
        cfg = yaml.safe_load((cfg_dir / f"{stem}.yaml").read_text(encoding="utf-8"))
        cfg["corpus_csv"] = str(eval_dir / "data/corpus_test_clm.csv")
        cfg["output_json"] = str(OUT / f"{stem}.json")
        if smoke:
            cfg["n_texts"] = 32
            cfg["bleu_n_texts"] = 16
        cfg_path = Path(f"/tmp/{stem}_eval.yaml")
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"EVAL {stem} smoke={smoke}", flush=True)
        run([sys.executable, str(eval_dir / "scripts/evaluate_clm.py"), "--config", str(cfg_path)], cwd=eval_dir)
        # Non-smoke runs write into Evaluation/metrics/clm/ per config; collect either way.
        for candidate in (OUT / f"{stem}.json", eval_dir / "metrics/clm" / f"{stem}.json"):
            if candidate.is_file():
                dest = OUT / f"{stem}.json"
                if candidate != dest:
                    dest.write_bytes(candidate.read_bytes())
                if stem in SRC_SHA:
                    data = json.loads(dest.read_text(encoding="utf-8"))
                    data["src_sha"] = SRC_SHA[stem]
                    dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                done.append(stem)
                print(f"OK {stem}: {json.loads(dest.read_text())}", flush=True)
                break
        else:
            print(f"WARN no metrics json for {stem}", flush=True)

    api = HfApi(token=token)
    api.create_repo(EVAL_DATASET, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=EVAL_DATASET, folder_path=str(OUT), repo_type="dataset",
                      commit_message=f"Tier-1 CLM eval ({', '.join(done) or 'none'}, smoke={smoke})")
    print("EVAL_CLM_FAMILY_JOB_OK", done, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
