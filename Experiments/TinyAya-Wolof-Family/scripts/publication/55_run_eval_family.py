#!/usr/bin/env python3
"""Run full Tiny Aya family CLM evals (BPB + PPL + BLEU + chrF) via Evaluation/evaluate_clm.py.

Aligns Tiny Aya Hub metrics with the LFM/MORENA gold protocol:
  Evaluation/scripts/evaluate_clm.py → BPB, perplexity, BLEU, chrF

Also refreshes FAMILY_EVAL.md and republishes Hub metrics via 06/56.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]  # Wolof-LLM-main
EVAL_ROOT = REPO / "Evaluation"
sys.path.insert(0, str(ROOT))
from common.eval_gate import is_full_clm, is_full_clm_file
from common.ortho_variants import ORTHO_VARIANTS
from common.pipeline import atomic_json_write, utc_now


def py() -> str:
    return str(ROOT / ".venv" / "bin" / "python")


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, env=merged)


def export_corpus_csv(ortho: str) -> Path:
    """Export Tiny Aya ortho test parquet → Evaluation/data CSV (text column)."""
    variant = ORTHO_VARIANTS[ortho]
    kept = ROOT / "data" / variant["data_dirname"] / "cpt" / "test_ortho_kept.parquet"
    src = kept if kept.is_file() else ROOT / "data" / variant["data_dirname"] / "cpt" / "test.parquet"
    out = EVAL_ROOT / "data" / f"corpus_test_clm_ortho_{ortho}_mixed_tinyaya.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Prefer shared LFM ns corpus when identical protocol is desired for ns.
    shared_ns = EVAL_ROOT / "data" / "corpus_test_clm_ortho_ns_mixed.csv"
    if ortho == "ns" and shared_ns.is_file() and not out.is_file():
        # Still write a TinyAya-tagged copy for provenance; reuse shared bytes.
        out.write_bytes(shared_ns.read_bytes())
        print(f"CORPUS_COPY {shared_ns.name} -> {out.name}", flush=True)
        return out
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"CORPUS_SKIP {out}", flush=True)
        return out
    frame = pd.read_parquet(src)
    frame[["text"]].dropna().to_csv(out, index=False)
    print(f"CORPUS_OK {out} n={len(frame)}", flush=True)
    return out


def targets() -> list[dict]:
    """Models that exist locally and should get full CLM metrics."""
    rows: list[dict] = []
    # Stock baselines on ns primary corpus.
    for role, model_id in (
        ("stock_base", "CohereLabs/tiny-aya-base"),
        ("stock_earth", "CohereLabs/tiny-aya-earth"),
        ("stock_global", "CohereLabs/tiny-aya-global"),
    ):
        rows.append(
            {
                "key": role,
                "ortho": "ns",
                "model": model_id,
                "hub_repo": None,
                "out": ROOT / "results" / "bpb" / f"clm_{role}_ns_3b35.json",
                "stage": "Base" if role == "stock_base" else "Instruct",
            }
        )
    cfg = yaml.safe_load(
        (ROOT / "configs" / "publication" / "eval_family.yaml").read_text(encoding="utf-8")
    )
    for group, ortho in (
        ("cpt", "ns"),
        ("cpt_std", "std"),
        ("instruct", "ns"),
        ("instruct_std", "std"),
        ("dpo", "ns"),
        ("dpo_std", "std"),
    ):
        for entry in cfg.get(group) or []:
            local = entry.get("local_artifact")
            if not local:
                continue
            path = ROOT / local
            if not path.is_dir():
                print(f"TARGET_SKIP missing {local}", flush=True)
                continue
            role = entry.get("role") or "model"
            rows.append(
                {
                    "key": f"{group}:{role}",
                    "ortho": ortho,
                    "model": str(path),
                    "hub_repo": entry.get("hub_repo"),
                    "out": ROOT
                    / "results"
                    / "bpb"
                    / f"clm_{role}_{ortho}_3b35.json",
                    "stage": entry.get("stage") or "Instruct",
                    "legacy_probe": ROOT
                    / "results"
                    / "bpb"
                    / (
                        f"cpt_{ortho}_3b35.json"
                        if role == "cpt"
                        else f"instruct_{role}_{ortho}_3b35.json"
                        if role.startswith("from_")
                        else f"dpo_{ortho}_3b35.json"
                    ),
                    "sft_eval": entry.get("sft_eval"),
                }
            )
    return rows


def run_evaluate_clm(
    *,
    model: str,
    corpus_csv: Path,
    output_json: Path,
    bleu_n_texts: int,
    n_texts,
    force: bool = False,
) -> dict:
    if output_json.is_file() and not force and is_full_clm_file(output_json):
        print(f"CLM_SKIP full {output_json.relative_to(ROOT)}", flush=True)
        return json.loads(output_json.read_text(encoding="utf-8"))
    if output_json.is_file() and not force and not is_full_clm_file(output_json):
        print(
            f"CLM_RESUME incomplete (need BPB+BLEU+chrF+PPL) -> {output_json.name}",
            flush=True,
        )

    cfg = {
        "corpus_csv": str(corpus_csv.resolve()),
        "text_column": "text",
        "n_texts": n_texts,
        "apply_verbalization": True,
        "bpb_max_len": 1024,
        "compute_bleu": True,
        "bleu_n_texts": bleu_n_texts,
        "bleu_seed": 0,
        "bleu_prompt_ratio": 0.5,
        "bleu_length_ratio": 1.3,
        "bleu_min_new_tokens": 16,
        "bleu_max_new_tokens": 256,
        "base_model_id": model,
        "subfolder": None,
        "tokenizer_subfolder": None,
        "allow_cpu": False,
        "output_json": str(output_json.resolve()),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8", dir=str(ROOT / "results" / "full_pipeline")
    ) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temp = handle.name
    try:
        run_cmd(
            [py(), str(EVAL_ROOT / "scripts" / "evaluate_clm.py"), "--config", temp],
            cwd=EVAL_ROOT,
            env={
                "PYTHONPATH": f"{EVAL_ROOT}:{EVAL_ROOT / 'scripts'}",
                "TRACKIO_DISABLED": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            },
        )
    finally:
        Path(temp).unlink(missing_ok=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def sync_primary_alias(clm_path: Path, legacy: Path | None) -> None:
    """Also write/overwrite the publish-path JSON used by 06/56 as full CLM metrics."""
    if not clm_path.is_file():
        return
    data = json.loads(clm_path.read_text(encoding="utf-8"))
    # Prefer canonical publish names from eval_family.yaml (cpt_ns, instruct_*, dpo_*).
    if legacy is not None:
        legacy.parent.mkdir(parents=True, exist_ok=True)
        # Merge probe bpb_by_lang if present on old lightweight file.
        if legacy.is_file():
            try:
                old = json.loads(legacy.read_text(encoding="utf-8"))
                if isinstance(old.get("bpb"), dict) and "bpb_by_lang" not in data:
                    data["bpb_by_lang"] = old["bpb"]
                    data["bpb"] = old["bpb"]
            except Exception:  # noqa: BLE001
                pass
        atomic_json_write(legacy, data)


def write_family_eval(rows: list[dict], cfg: dict) -> Path:
    out = ROOT / (cfg.get("metrics_out") or "docs/FAMILY_EVAL.md")
    lines = [
        "# Tiny Aya Wolof Family — eval table",
        "",
        f"Updated: `{utc_now()}`",
        "",
        "License: **CC-BY-NC-4.0** + Cohere Labs AUP. Gated stock checkpoints.",
        "",
        "## Protocol",
        "",
        "- **Gold CLM** = `Evaluation/scripts/evaluate_clm.py` (same as LFM/MORENA):",
        "  verbalized BPB + perplexity + BLEU + chrF on ortho NTP test CSV.",
        "- Primary corpus: ns → `corpus_test_clm_ortho_ns_mixed*.csv`; std → TinyAya std export.",
        "- SFT task evals: `results/sft/eval_*/evaluation.json` (val mean loss).",
        "- Hub: `metrics_clm.json` + composed `metrics.json` / `model-index` via `56_publish_hub_metrics.py`.",
        "",
        "## HF CLM (evaluate_clm.py)",
        "",
        "| checkpoint | BPB ↓ | BLEU | chrF | PPL | sft_loss ↓ | notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if v < 20 else f"{v:.2f}"
        return str(v)

    for row in rows:
        clm = row.get("metrics") or {}
        sft_loss = row.get("sft_loss")
        name = row.get("hub_repo") or row.get("key")
        lines.append(
            f"| {name} | {fmt(clm.get('BPB'))} | {fmt(clm.get('BLEU'))} | "
            f"{fmt(clm.get('chrF'))} | {fmt(clm.get('perplexity'))} | "
            f"{fmt(sft_loss)} | {row.get('notes', '')} |"
        )
    lines.extend(
        [
            "",
            "from_cpt Instruct skipped (vocab layout).",
            "GGUF not in scope for Tiny Aya 3.35B (HF weights only).",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    atomic_json_write(
        ROOT / "results" / "full_pipeline" / "family_eval_summary.json",
        {"updated_at": utc_now(), "rows": rows},
    )
    print(f"FAMILY_EVAL_OK {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny Aya full CLM family eval + Hub publish")
    parser.add_argument("--bleu-n-texts", type=int, default=100)
    parser.add_argument("--n-texts", default="all")
    parser.add_argument("--only-key", action="append", default=[])
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    n_texts = None if str(args.n_texts).lower() in {"all", "none", ""} else int(args.n_texts)
    cfg = yaml.safe_load(
        (ROOT / "configs" / "publication" / "eval_family.yaml").read_text(encoding="utf-8")
    )

    corpora = {ortho: export_corpus_csv(ortho) for ortho in ("ns", "std")}
    jobs = targets()
    if args.only_key:
        wanted = set(args.only_key)
        jobs = [j for j in jobs if j["key"] in wanted or (j.get("hub_repo") in wanted)]

    table_rows: list[dict] = []
    for job in jobs:
        out = job["out"]
        if args.force and out.is_file():
            out.unlink()
        metrics = None
        if not args.skip_eval:
            try:
                metrics = run_evaluate_clm(
                    model=job["model"],
                    corpus_csv=corpora[job["ortho"]],
                    output_json=out,
                    bleu_n_texts=args.bleu_n_texts,
                    n_texts=n_texts,
                    force=args.force,
                )
                if not is_full_clm(metrics):
                    raise RuntimeError(
                        f"evaluate_clm returned incomplete metrics for {job['key']}: "
                        f"keys={sorted(metrics)}"
                    )
                sync_primary_alias(out, job.get("legacy_probe"))
                print(
                    f"CLM_OK {job['key']} BPB={metrics.get('BPB')} BLEU={metrics.get('BLEU')} "
                    f"chrF={metrics.get('chrF')} PPL={metrics.get('perplexity')}",
                    flush=True,
                )
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                print(f"CLM_FAIL {job['key']}: {exc}", flush=True)
                continue
        elif out.is_file():
            metrics = json.loads(out.read_text(encoding="utf-8"))

        sft_loss = None
        sft_rel = job.get("sft_eval")
        if sft_rel:
            sft_path = ROOT / sft_rel
            if sft_path.is_file():
                sft = json.loads(sft_path.read_text(encoding="utf-8"))
                sft_loss = sft.get("mean_loss") or sft.get("eval_loss")

        # Copy full CLM JSON onto the publish path expected by eval_family.yaml.
        if metrics and job.get("hub_repo"):
            # Map to yaml clm_metrics by matching hub_repo.
            for group in ("cpt", "cpt_std", "instruct", "instruct_std", "dpo", "dpo_std"):
                for entry in cfg.get(group) or []:
                    if entry.get("hub_repo") != job["hub_repo"]:
                        continue
                    dest = ROOT / entry["clm_metrics"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json_write(dest, metrics)
                    print(f"ALIAS {dest.relative_to(ROOT)}", flush=True)

        table_rows.append(
            {
                "key": job["key"],
                "hub_repo": job.get("hub_repo") or job["key"],
                "metrics": metrics or {},
                "sft_loss": sft_loss,
                "notes": f"{job['ortho']} {job.get('stage')}",
            }
        )

    write_family_eval(table_rows, cfg)

    if args.skip_publish:
        return 0

    which = [
        "cpt_merged",
        "cpt_merged_std",
        "instruct_earth",
        "instruct_earth_std",
        "instruct_global",
        "instruct_global_std",
        "dpo",
        "dpo_std",
        "bpb",
        "sft_eval",
        "family_eval",
    ]
    run_cmd([py(), str(ROOT / "scripts" / "publication" / "06_publish_artifacts.py"), "--which", *which])
    print("EVAL_FAMILY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
