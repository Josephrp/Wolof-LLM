#!/usr/bin/env python3
"""Run golden CLM eval (BPB) on a few elite checkpoints/models and keep top-N.

Protocol matches LFM/TinyAya family gold:
  Evaluation/scripts/evaluate_clm.py → BPB (primary).

Only evaluates ``--max-candidates`` (best by prior score first). Use this on
merged models, or on ``best_pool`` adapters after a quick merge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
EVAL_ROOT = REPO / "Evaluation"
sys.path.insert(0, str(ROOT))
from common.checkpoint_pool import load_pool_index, pool_dir, prune_pool, save_pool_index
from common.pipeline import atomic_json_write, utc_now


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def default_corpus() -> Path:
    for candidate in (
        EVAL_ROOT / "data" / "corpus_test_clm_ortho_ns_mixed.csv",
        EVAL_ROOT / "data" / "corpus_test_clm.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No golden CLM corpus CSV under Evaluation/data/")


def evaluate_model(model_path: Path, *, corpus: Path, max_texts: int | None, out_json: Path) -> float:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path(sys.executable)
    payload = {
        "corpus_csv": str(corpus),
        "text_column": "text",
        "n_texts": max_texts,
        "apply_verbalization": True,
        "bpb_max_len": 1024,
        "bleu_n_texts": min(50, max_texts or 50),
        "bleu_seed": 0,
        "base_model_id": str(model_path),
        "subfolder": None,
        "tokenizer_subfolder": None,
        "output_json": str(out_json),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        cfg_path = Path(handle.name)
    try:
        run_cmd(
            [str(py), str(EVAL_ROOT / "scripts" / "evaluate_clm.py"), "--config", str(cfg_path)],
            cwd=EVAL_ROOT,
        )
    finally:
        cfg_path.unlink(missing_ok=True)
    metrics = json.loads(out_json.read_text(encoding="utf-8"))
    if "BPB" in metrics:
        return float(metrics["BPB"])
    if "bpb" in metrics:
        return float(metrics["bpb"])
    raise KeyError(f"No BPB in {out_json}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden BPB select over best_pool / model list")
    parser.add_argument("--run-dir", default="runs/clm/08b/v1_ntp70_ortho_lora")
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit model dirs")
    parser.add_argument("--keep", type=int, default=15)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--max-texts", type=int, default=200)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    if args.models:
        candidates = [
            {"name": Path(m).name, "step": i, "score": float(i), "path": str(Path(m).resolve())}
            for i, m in enumerate(args.models)
        ]
    else:
        index = load_pool_index(run_dir)
        candidates = sorted(list(index.get("entries") or []), key=lambda e: float(e["score"]))
        if not candidates:
            live = sorted(
                [p for p in run_dir.glob("checkpoint-*") if p.is_dir()],
                key=lambda p: int(p.name.split("-")[-1]),
                reverse=True,
            )
            candidates = [
                {
                    "name": p.name,
                    "step": int(p.name.split("-")[-1]),
                    "score": float("inf"),
                    "path": str(p),
                }
                for p in live
            ]
        candidates = candidates[: max(1, int(args.max_candidates))]

    corpus = args.corpus or default_corpus()
    results_dir = run_dir / "best_pool" / "golden_eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    scored: list[dict] = []
    for entry in candidates:
        model_path = Path(entry["path"])
        if not model_path.exists():
            print(f"SKIP missing {model_path}", flush=True)
            continue
        # LoRA adapter dirs are not golden-evalable without merge.
        if (model_path / "adapter_config.json").is_file() and not (
            model_path / "config.json"
        ).is_file():
            print(f"SKIP adapter-only {model_path} (merge before golden eval)", flush=True)
            continue
        out_json = results_dir / f"{model_path.name}_golden.json"
        print(f"GOLDEN_EVAL {model_path} n={args.max_texts}", flush=True)
        if args.dry_run:
            scored.append({**entry, "bpb": None})
            continue
        bpb = evaluate_model(
            model_path, corpus=corpus, max_texts=args.max_texts, out_json=out_json
        )
        scored.append({**entry, "bpb": bpb, "golden_json": str(out_json)})
        print(f"GOLDEN_OK {model_path.name} BPB={bpb:.4f}", flush=True)

    if args.dry_run:
        print(json.dumps({"candidates": scored}, indent=2), flush=True)
        return 0
    if not scored:
        print("GOLDEN_EVAL_EMPTY", flush=True)
        return 1

    scored.sort(key=lambda e: float(e["bpb"]))
    index = load_pool_index(run_dir)
    index["metric"] = "bpb"
    index["greater_is_better"] = False
    index["golden_selected_at"] = utc_now()
    index["entries"] = [
        {
            "name": e["name"],
            "step": int(e["step"]),
            "score": float(e["bpb"]),
            "eval_loss": e.get("score"),
            "path": e["path"],
            "golden_json": e.get("golden_json"),
            "promoted_at": utc_now(),
        }
        for e in scored
    ]
    save_pool_index(run_dir, index)
    if pool_dir(run_dir).is_dir():
        prune_pool(run_dir, keep=args.keep)
    summary = {
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "keep": args.keep,
        "max_candidates": args.max_candidates,
        "ranking": [
            {"name": e["name"], "bpb": e["bpb"], "prior_score": e.get("score")} for e in scored
        ],
    }
    atomic_json_write(results_dir / "selection.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("GOLDEN_EVAL_SELECT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
