#!/usr/bin/env python3
"""Backfill Tiny Aya BPB/SFT evals, rewrite FAMILY_EVAL.md, publish Hub metrics.

Fixes the production gap where publish looked for legacy metric paths
(cpt_3b35.json / eval_from_earth) while the runner wrote ortho-aware names
(cpt_ns_3b35.json / eval_3b35_from_earth), so Hub cards shipped with METRICS_SKIP.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.ortho_variants import ORTHO_VARIANTS, resolve_variants
from common.pipeline import atomic_json_write, utc_now


def uv(*args: str) -> list[str]:
    return [str(ROOT / ".venv" / "bin" / "python"), *args]


def run_cmd(cmd: list[str], *, check: bool = True) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=check)


def bpb_test(ortho: str) -> Path:
    variant = ORTHO_VARIANTS[ortho]
    kept = ROOT / "data" / variant["data_dirname"] / "cpt" / "test_ortho_kept.parquet"
    if kept.is_file():
        return kept
    return ROOT / "data" / variant["data_dirname"] / "cpt" / "test.parquet"


def ensure_bpb(model: Path | str, out: Path, data: Path, *, max_texts: int) -> bool:
    if out.is_file():
        print(f"BPB_SKIP exists {out.relative_to(ROOT)}", flush=True)
        return True
    model_s = str(model)
    if not str(model).startswith(("CohereLabs/", "Tonic/", "galsenai/")):
        model_path = ROOT / model if not Path(model).is_absolute() else Path(model)
        if not model_path.is_dir():
            print(f"BPB_SKIP missing model {model}", flush=True)
            return False
        model_s = str(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_cmd(
            uv(
                str(ROOT / "scripts" / "clm" / "04_evaluate_bpb.py"),
                "--model",
                model_s,
                "--data",
                str(data),
                "--output",
                str(out),
                "--max-texts",
                str(max_texts),
            )
        )
        return out.is_file()
    except subprocess.CalledProcessError as exc:
        print(f"BPB_FAIL {out.name}: {exc}", flush=True)
        return False


def ensure_sft(
    model: Path,
    out_dir: Path,
    prepared: Path,
    *,
    max_rows: int,
    force: bool = False,
) -> bool:
    eval_json = out_dir / "evaluation.json"
    if eval_json.is_file() and not force:
        print(f"SFT_SKIP exists {eval_json.relative_to(ROOT)}", flush=True)
        return True
    if not model.is_dir():
        print(f"SFT_SKIP missing model {model}", flush=True)
        return False
    if not (prepared / "val.parquet").is_file():
        print(f"SFT_SKIP missing val {prepared}", flush=True)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_cmd(
            uv(
                str(ROOT / "scripts" / "sft" / "12_evaluate_sft.py"),
                "--model",
                str(model),
                "--data",
                str(prepared),
                "--output",
                str(out_dir),
                "--max-rows",
                str(max_rows),
            )
        )
        return eval_json.is_file()
    except subprocess.CalledProcessError as exc:
        print(f"SFT_FAIL {out_dir.name}: {exc}", flush=True)
        return False


def merge_dpo(ortho: str) -> Path | None:
    variant = ORTHO_VARIANTS[ortho]
    out = ROOT / "models" / variant["merged_dpo"]
    if (out / "config.json").is_file():
        print(f"DPO_MERGE_SKIP exists {out.name}", flush=True)
        return out
    adapter = ROOT / "runs" / "sft" / (
        "3b35_std_dpo_v1" if ortho == "std" else "3b35_dpo_v1"
    ) / "adapter"
    if not adapter.is_dir():
        print(f"DPO_MERGE_SKIP missing adapter {adapter}", flush=True)
        return None
    best_ptr = ROOT / "models" / variant["merged_best"] / "best_pointer.json"
    base = None
    if best_ptr.is_file():
        best = json.loads(best_ptr.read_text(encoding="utf-8")).get("best") or {}
        cand = Path(best.get("path") or "")
        if cand.is_dir() and (cand / "config.json").is_file():
            base = cand
    if base is None:
        # Prefer the pick_best roles used in production.
        for key in ("merged_global", "merged_earth", "merged_instruct"):
            cand = ROOT / "models" / variant[key]
            if cand.is_dir() and (cand / "config.json").is_file():
                base = cand
                break
    if base is None:
        print(f"DPO_MERGE_SKIP no base for {ortho}", flush=True)
        return None
    try:
        run_cmd(
            uv(
                str(ROOT / "scripts" / "clm" / "05_merge_adapter.py"),
                "--base",
                str(base),
                "--adapter",
                str(adapter),
                "--output",
                str(out),
            )
        )
        return out if (out / "config.json").is_file() else None
    except subprocess.CalledProcessError as exc:
        print(f"DPO_MERGE_FAIL {ortho}: {exc}", flush=True)
        return None


def load_bpb(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_sft_loss(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("mean_loss", "eval_loss", "loss"):
        if data.get(key) is not None:
            return float(data[key])
    return None


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_family_eval(cfg: dict) -> Path:
    out = ROOT / (cfg.get("metrics_out") or "docs/FAMILY_EVAL.md")
    rows: list[dict] = []

    def add_row(name: str, clm: Path | None, sft: Path | None, notes: str) -> None:
        bpb = load_bpb(clm) if clm else None
        b = (bpb or {}).get("bpb") or {}
        rows.append(
            {
                "name": name,
                "bpb_all": b.get("all"),
                "bpb_wo": b.get("wo"),
                "sft_loss": load_sft_loss(sft) if sft else None,
                "notes": notes,
            }
        )

    for entry in cfg.get("stock") or []:
        add_row(
            entry.get("model_id") or entry.get("role"),
            ROOT / entry["clm_metrics"] if entry.get("clm_metrics") else None,
            None,
            "stock baseline (ns test)",
        )
    for group, note in (
        ("cpt", "ns CPT"),
        ("cpt_std", "std CPT"),
        ("instruct", "ns Instruct"),
        ("instruct_std", "std Instruct"),
        ("dpo", "ns DPO"),
        ("dpo_std", "std DPO"),
    ):
        for entry in cfg.get(group) or []:
            add_row(
                entry.get("hub_repo") or entry.get("role"),
                ROOT / entry["clm_metrics"] if entry.get("clm_metrics") else None,
                ROOT / entry["sft_eval"] if entry.get("sft_eval") else None,
                note,
            )

    lines = [
        "# Tiny Aya Wolof Family — eval table",
        "",
        f"Updated: `{utc_now()}`",
        "",
        "License: **CC-BY-NC-4.0** + Cohere Labs AUP. Gated stock checkpoints.",
        "",
        "## Protocol",
        "",
        "- **Primary BPB** = matching ortho NTP test (`results/bpb/{role}_{ns|std}_3b35.json` → Hub `metrics_clm.json`).",
        "- **SFT eval** = val mean loss (`results/sft/eval_*/evaluation.json` → Hub `metrics_sft_evaluation.json`).",
        "- Hub cards: composed `metrics.json` + HF `model-index` via `56_publish_hub_metrics.py`.",
        "",
        "| checkpoint | bpb_all ↓ | bpb_wo ↓ | sft_eval_loss ↓ | notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {fmt(row['bpb_all'])} | {fmt(row['bpb_wo'])} | "
            f"{fmt(row['sft_loss'])} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "Ortho filter: formal/`standard` or informal/`non_standard` + `mixed`; `not_wolof` dropped.",
            "Anchors: Cohere developer preamble + `<|lang_wo|>` + `<|ortho_ns|>` or `<|ortho_std|>`.",
            "Epochs: CPT=3, SFT=3. from_cpt Instruct skipped (vocab layout).",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "updated_at": utc_now(),
        "rows": rows,
    }
    atomic_json_write(ROOT / "results" / "full_pipeline" / "family_eval_summary.json", summary)
    print(f"FAMILY_EVAL_OK {out}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Tiny Aya evals + Hub metrics")
    parser.add_argument("--ortho", default="both", choices=["ns", "std", "both"])
    parser.add_argument("--max-texts", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-dpo-merge", action="store_true")
    parser.add_argument("--force-sft-std", action="store_true", help="Re-run std SFT evals on std prepared val")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--dry-run-publish", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(
        (ROOT / "configs" / "publication" / "eval_family.yaml").read_text(encoding="utf-8")
    )
    orthos = resolve_variants(args.ortho)

    if not args.skip_eval:
        for ortho in orthos:
            variant = ORTHO_VARIANTS[ortho]
            test = bpb_test(ortho)
            prepared = ROOT / "data" / "sft" / variant["prepared_dirname"]
            prefix = variant["sft_run_prefix"]

            # CPT BPB (already present for prod; ensure).
            ensure_bpb(
                ROOT / "models" / variant["merged_cpt"],
                ROOT / "results" / "bpb" / f"cpt_{ortho}_3b35.json",
                test,
                max_texts=args.max_texts,
            )
            # Stock baselines once on ns (and optionally std test).
            if ortho == "ns":
                for label, model_id in (
                    ("stock_base", "CohereLabs/tiny-aya-base"),
                    ("stock_earth", "CohereLabs/tiny-aya-earth"),
                    ("stock_global", "CohereLabs/tiny-aya-global"),
                ):
                    ensure_bpb(
                        model_id,
                        ROOT / "results" / "bpb" / f"{label}_{ortho}_3b35.json",
                        test,
                        max_texts=args.max_texts,
                    )

            for role, model_key in (
                ("from_earth", "merged_earth"),
                ("from_global", "merged_global"),
                ("from_cpt", "merged_instruct"),
            ):
                model = ROOT / "models" / variant[model_key]
                ensure_bpb(
                    model,
                    ROOT / "results" / "bpb" / f"instruct_{role}_{ortho}_3b35.json",
                    test,
                    max_texts=args.max_texts,
                )
                ensure_sft(
                    model,
                    ROOT / "results" / "sft" / f"eval_{prefix}_{role}",
                    prepared,
                    max_rows=args.max_rows,
                    force=bool(args.force_sft_std and ortho == "std"),
                )

            if not args.skip_dpo_merge:
                dpo_merged = merge_dpo(ortho)
                if dpo_merged is not None:
                    ensure_bpb(
                        dpo_merged,
                        ROOT / "results" / "bpb" / f"dpo_{ortho}_3b35.json",
                        test,
                        max_texts=args.max_texts,
                    )
                    ensure_sft(
                        dpo_merged,
                        ROOT / "results" / "sft" / f"eval_{prefix}_dpo",
                        prepared,
                        max_rows=args.max_rows,
                        force=False,
                    )

    write_family_eval(cfg)

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
    # Restrict by ortho if requested.
    if orthos == ["ns"]:
        which = [w for w in which if not w.endswith("_std")] + ["bpb", "sft_eval", "family_eval"]
    elif orthos == ["std"]:
        which = [w for w in which if w.endswith("_std") or w in {"bpb", "sft_eval", "family_eval"}]

    publish_cmd = uv(
        str(ROOT / "scripts" / "publication" / "06_publish_artifacts.py"),
        "--which",
        *sorted(set(which)),
    )
    if args.dry_run_publish:
        print("DRY_RUN would run:", " ".join(publish_cmd), flush=True)
        run_cmd(
            uv(
                str(ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"),
                "--dry-run",
            ),
            check=False,
        )
        return 0

    run_cmd(publish_cmd)
    print("BACKFILL_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
