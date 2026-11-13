#!/usr/bin/env python3
"""Free disk under Experiments/* by removing intermediate family run artifacts.

Keeps merged models and an active Qwen CPT run. Targets ~70% volume usage.
Also prunes Qwen best_pool to top-N by eval_loss.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
VOLUME = Path("/mnt/volume_w2l8nkp")
sys.path.insert(0, str(ROOT))


def du_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    proc = subprocess.run(
        ["du", "-sb", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0.0
    return int(proc.stdout.split()[0]) / (1024**3)


def disk_pct(mount: Path) -> tuple[float, float, float]:
    st = shutil.disk_usage(mount)
    used_pct = 100.0 * st.used / st.total
    return used_pct, st.used / (1024**3), st.free / (1024**3)


def rm_tree(path: Path, *, dry_run: bool) -> float:
    size = du_gb(path)
    if not path.exists():
        return 0.0
    print(f"{'DRY ' if dry_run else ''}RM {path} ({size:.1f} GiB)", flush=True)
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)
    return size


def keep_newest_checkpoints(root: Path, keep: int, *, dry_run: bool) -> float:
    if not root.is_dir():
        return 0.0
    ckpts = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    freed = 0.0
    for path in ckpts[:-keep] if keep > 0 else ckpts:
        freed += rm_tree(path, dry_run=dry_run)
    return freed


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune family intermediate artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-pct", type=float, default=70.0)
    parser.add_argument("--pool-keep", type=int, default=15)
    parser.add_argument("--skip-galsenllm", action="store_true")
    args = parser.parse_args()

    disk_root = VOLUME if VOLUME.exists() else REPO
    pct, used, free = disk_pct(disk_root)
    print(f"DISK_BEFORE used={used:.0f}GiB free={free:.0f}GiB pct={pct:.1f}%", flush=True)

    freed = 0.0
    # Intermediate training runs (merged models already exist under models/).
    delete_paths = [
        REPO / "Experiments/LFM2.5-Base-1B/runs/clm",
        REPO / "Experiments/LFM2.5-Base-1B/runs/sft",
        REPO / "Experiments/LFM2.5-Base-1B/runs/smoke",
        REPO / "Experiments/LFM2.5-Base-1B/runs/smoke_8b_moe_lora",
        REPO / "Experiments/LFM2.5-Base-1B/runs/smoke_2.6b_ext_lora",
        REPO / "Experiments/LFM2.5-Base-1B/runs/mini",
        REPO / "Experiments/LFM2.5-Base-1B/runs/lfm_8b_a1b_v2_moe_lora",
        REPO / "Experiments/LFM2.5-Base-1B/runs/lfm_2.6b_v2_ext4000_lora",
        REPO / "Experiments/MORENA/runs/cpt",
        REPO / "Experiments/MORENA/runs/sft",
        REPO / "Experiments/TinyAya-Wolof-Family/runs/clm",
        REPO / "Experiments/TinyAya-Wolof-Family/runs/sft",
        # Smoke / superseded Qwen merges (active CPT uses runs/ + ext-init).
        ROOT / "models/qwen35-08b-cpt-v1-merged",
        ROOT / "models/qwen35-08b-instruct-from-cpt-merged",
        ROOT / "results/smoke",
    ]
    for path in delete_paths:
        if path.exists():
            freed += rm_tree(path, dry_run=args.dry_run)

    # Old flat CLM dump: keep newest 15 only.
    clm_v1 = VOLUME / "clm_v1_checkpoints"
    if clm_v1.is_dir():
        freed += keep_newest_checkpoints(clm_v1, 15, dry_run=args.dry_run)

    if not args.skip_galsenllm:
        for base in (
            VOLUME / "galsenllm/OutputOfSmolLM_Model/runs",
            VOLUME / "galsenllm/OutputOfSmolLM_Model/runs_sft",
            VOLUME / "galsenllm/OutputOfTinyAya_Model/runs",
            VOLUME / "galsenllm/OutputOfTinyAya_Model/runs_sft",
        ):
            if not base.is_dir():
                continue
            # Prefer deleting whole run trees when models/ already holds finals.
            models_sib = base.parent / "models"
            if models_sib.is_dir():
                freed += rm_tree(base, dry_run=args.dry_run)
            else:
                for run in base.iterdir():
                    if run.is_dir():
                        freed += keep_newest_checkpoints(run, args.pool_keep, dry_run=args.dry_run)

    # Qwen active CPT: prune best_pool only (do not touch live checkpoint-*).
    qwen_run = ROOT / "runs/clm/08b/v1_ntp70_ortho_lora"
    if qwen_run.is_dir():
        from common.checkpoint_pool import prune_pool

        prune_pool(qwen_run, keep=args.pool_keep)
        print(f"BEST_POOL pruned keep={args.pool_keep} under {qwen_run}", flush=True)

    pct2, used2, free2 = disk_pct(disk_root)
    print(
        f"DISK_AFTER used={used2:.0f}GiB free={free2:.0f}GiB pct={pct2:.1f}% "
        f"freed_est={freed:.0f}GiB target={args.target_pct}%",
        flush=True,
    )
    if pct2 > args.target_pct + 5:
        print(
            "DISK_WARN still above target; consider pruning duplicate merged models "
            "via 61_golden_eval_select.py after eval.",
            flush=True,
        )
    print("PRUNE_FAMILY_DISK_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
