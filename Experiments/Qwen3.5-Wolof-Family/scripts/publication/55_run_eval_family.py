#!/usr/bin/env python3
"""Run full HF CLM evals (BPB, PPL, BLEU, chrF) for Qwen3.5 Wolof family artifacts.

Writes metrics under results/clm_eval/{stock,cpt,instruct}_{size}.json and
refreshes docs/FAMILY_EVAL.md via 56_publish_hub_metrics.py when --publish.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
EVAL_ROOT = REPO / "Evaluation"
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now  # noqa: E402

SIZE_HUB = {"08b": "08B", "2b": "2B", "4b": "4B", "9b": "9B"}
STOCK_HUB = {
    "08b": "Qwen/Qwen3.5-0.8B-Base",
    "2b": "Qwen/Qwen3.5-2B-Base",
    "4b": "Qwen/Qwen3.5-4B-Base",
    "9b": "Qwen/Qwen3.5-9B-Base",
}
# Prefer local ext-init (vocab-extended) over Hub stock — same CLM protocol as CPT/Instruct.
STOCK_LOCAL = {
    "08b": ("qwen35-0.8b-ext-init", "qwen35-08b-ext-init"),
    "2b": ("qwen35-2b-ext-init",),
    "4b": ("qwen35-4b-ext-init",),
    "9b": ("qwen35-9b-ext-init",),
}


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def default_corpus() -> Path:
    for candidate in (
        EVAL_ROOT / "data" / "corpus_test_clm_ortho_ns_mixed.csv",
        ROOT / "data" / "ortho_ns_mixed_v1" / "cpt" / "test.csv",
        EVAL_ROOT / "data" / "corpus_test_clm.csv",
    ):
        if candidate.is_file():
            return candidate
    # Build CSV from parquet if needed.
    parquet = ROOT / "data" / "ortho_ns_mixed_v1" / "cpt" / "test.parquet"
    if parquet.is_file():
        import pandas as pd

        out = ROOT / "data" / "ortho_ns_mixed_v1" / "cpt" / "test_clm.csv"
        df = pd.read_parquet(parquet)
        if "text" not in df.columns:
            raise SystemExit(f"no text column in {parquet}")
        out.parent.mkdir(parents=True, exist_ok=True)
        df[["text"]].to_csv(out, index=False)
        return out
    raise FileNotFoundError("No CLM corpus CSV/parquet found for Qwen eval")


def resolve_model(size: str, kind: str) -> Path | str | None:
    if kind == "stock":
        for name in STOCK_LOCAL.get(size, ()):
            path = ROOT / "models" / name
            if path.is_dir():
                return path
        return STOCK_HUB[size]
    if kind == "cpt":
        path = ROOT / f"models/qwen35-{size}-cpt-v1-merged"
        return path if path.is_dir() else None
    if kind == "instruct":
        path = ROOT / f"models/qwen35-{size}-instruct-from-cpt-merged"
        return path if path.is_dir() else None
    raise ValueError(kind)


def run_one(
    *,
    size: str,
    kind: str,
    model: Path | str,
    corpus: Path,
    out_json: Path,
    max_texts: int | None,
    bleu_n: int,
    smoke: bool,
) -> dict:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path(sys.executable)
    n_texts = 32 if smoke else max_texts
    bleu_n_texts = min(16, bleu_n) if smoke else bleu_n
    payload = {
        "corpus_csv": str(corpus),
        "text_column": "text",
        "n_texts": n_texts,
        "apply_verbalization": True,
        "bpb_max_len": 1024,
        "bleu_n_texts": bleu_n_texts,
        "bleu_seed": 0,
        "bleu_prompt_ratio": 0.5,
        "bleu_length_ratio": 1.3,
        "bleu_min_new_tokens": 16,
        "bleu_max_new_tokens": 256,
        "base_model_id": str(model),
        "subfolder": None,
        "tokenizer_subfolder": None,
        "output_json": str(out_json),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        cfg = Path(handle.name)
    try:
        print(f"+ evaluate_clm {kind}:{size} -> {out_json}", flush=True)
        subprocess.run(
            [str(py), str(EVAL_ROOT / "scripts" / "evaluate_clm.py"), "--config", str(cfg)],
            cwd=str(EVAL_ROOT),
            check=True,
        )
    finally:
        cfg.unlink(missing_ok=True)
    metrics = json.loads(out_json.read_text(encoding="utf-8"))
    metrics["family_size"] = size
    metrics["family_kind"] = kind
    metrics["hub_repo"] = f"Tonic/Qwen3.5-{SIZE_HUB[size]}-Wolof-" + {
        "stock": "ext-init-v1",
        "cpt": "CPT-v1",
        "instruct": "Instruct-v1",
    }[kind]
    metrics["evaluated_at"] = utc_now()
    atomic_json_write(out_json, metrics)
    print(
        f"EVAL_OK {kind}:{size} BPB={metrics.get('BPB')} PPL={metrics.get('perplexity')} "
        f"BLEU={metrics.get('BLEU')} chrF={metrics.get('chrF')}",
        flush=True,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full CLM eval matrix for Qwen Wolof family")
    parser.add_argument("--sizes", nargs="+", default=["08b", "2b", "4b", "9b"])
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["stock", "cpt", "instruct"],
        choices=["stock", "cpt", "instruct"],
    )
    parser.add_argument("--max-texts", type=int, default=None, help="None = full corpus")
    parser.add_argument("--bleu-n", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with existing metrics JSON")
    parser.add_argument("--publish", action="store_true", help="Refresh Hub cards after evals")
    parser.add_argument("--corpus", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_trackio_env()
    corpus = args.corpus or default_corpus()
    print(f"corpus={corpus}", flush=True)
    out_dir = ROOT / "results" / "clm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "results" / "full_pipeline" / "clm_eval_state.json"
    state = {"updated_at": utc_now(), "jobs": {}}
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("jobs", {})

    for size in args.sizes:
        for kind in args.kinds:
            key = f"{kind}:{size}"
            out_json = out_dir / f"{kind}_{size}.json"
            if args.resume and out_json.is_file() and state.get("jobs", {}).get(key) == "complete":
                print(f"skip {key}", flush=True)
                continue
            model = resolve_model(size, kind)
            if model is None:
                print(f"SKIP missing model {key}", flush=True)
                state["jobs"][key] = "missing"
                continue
            try:
                metrics = run_one(
                    size=size,
                    kind=kind,
                    model=model,
                    corpus=corpus,
                    out_json=out_json,
                    max_texts=args.max_texts,
                    bleu_n=args.bleu_n,
                    smoke=args.smoke,
                )
                state["jobs"][key] = {
                    "status": "complete",
                    "BPB": metrics.get("BPB"),
                    "perplexity": metrics.get("perplexity"),
                    "BLEU": metrics.get("BLEU"),
                    "chrF": metrics.get("chrF"),
                    "at": utc_now(),
                }
            except Exception as exc:  # noqa: BLE001
                print(f"EVAL_FAIL {key}: {exc}", flush=True)
                state["jobs"][key] = {"status": "failed", "error": str(exc), "at": utc_now()}
            state["updated_at"] = utc_now()
            atomic_json_write(state_path, state)

    if args.publish:
        subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python")
                if (ROOT / ".venv" / "bin" / "python").is_file()
                else sys.executable,
                str(ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"),
                "--sizes",
                *args.sizes,
            ],
            cwd=str(ROOT),
            check=False,
        )

    print("CLM_EVAL_FAMILY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
