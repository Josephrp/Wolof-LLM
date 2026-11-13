#!/usr/bin/env python3
"""Launch vendored MORENA trainer (CPT or SFT) via torchrun.

Downloads Hub init weights when --hub-init is set. Prefer single-GPU --no-fsdp
on H200 for modest CPT budgets; multi-GPU when world>1.

Enables Hugging Face Trackio / Gradio logging by default when TRACKIO_SPACE_ID
is configured (see Experiments/MORENA/.env.trackio).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "modeling_morena.py"
sys.path.insert(0, str(ROOT))
from common.tracking import load_trackio_env  # noqa: E402


def infer_size_stage(config: Path, loss_mask: bool) -> tuple[str, str]:
    name = config.stem.lower()
    size = "morena"
    for token in ("1.5b", "0.5b", "0.2b", "mini", "nano"):
        if token in name:
            size = token
            break
    if "sft" in name or loss_mask:
        return size, "sft"
    if "cpt" in name or "clm" in name:
        return size, "cpt"
    return size, "sft" if loss_mask else "cpt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MORENA with vendor Muon/FSDP2")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hub-init", default="", help="Hub repo id with model.safetensors")
    parser.add_argument("--init-weights", default="", help="Local safetensors path")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--loss-mask", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--anneal", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Force max-steps=10")
    parser.add_argument("--no-fsdp", action="store_true", default=True)
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP (disables --no-fsdp)")
    parser.add_argument("--trackio", action="store_true", default=None, help="Force Trackio on")
    parser.add_argument("--no-trackio", action="store_true", help="Disable Trackio logging")
    parser.add_argument("--trackio-size", default="", help="Override Trackio size label")
    parser.add_argument("--trackio-stage", default="", help="Override Trackio stage label")
    parser.add_argument("--trackio-run-id", default="", help="Override Trackio run id")
    args = parser.parse_args()

    load_trackio_env(ROOT)

    init_weights = args.init_weights
    if args.hub_init and not init_weights:
        dest = ROOT / "models" / "hub_cache" / args.hub_init.replace("/", "__")
        snapshot_download(
            args.hub_init,
            local_dir=str(dest),
            allow_patterns=["model.safetensors", "config.json", "tokenizer.json"],
        )
        init_weights = str(dest / "model.safetensors")

    out = args.out
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    config = args.config if args.config.is_absolute() else ROOT / args.config
    mix = args.mix if args.mix.is_absolute() else ROOT / args.mix
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root

    size, stage = infer_size_stage(config, args.loss_mask)
    if args.trackio_size:
        size = args.trackio_size
    if args.trackio_stage:
        stage = args.trackio_stage

    trackio_on = not args.no_trackio and os.environ.get("TRACKIO_DISABLED") != "1"
    if args.trackio is True:
        trackio_on = True
    elif args.trackio is None and not os.environ.get("TRACKIO_SPACE_ID"):
        # Still pass --trackio so local SQLite logs; Space sync needs TRACKIO_SPACE_ID.
        trackio_on = True

    max_steps = 10 if args.smoke else args.max_steps
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        str(VENDOR),
        "--config",
        str(config),
        "--mix",
        str(mix),
        "--data-root",
        str(data_root),
        "--out",
        str(out),
        "--resume",
        args.resume,
    ]
    if init_weights:
        cmd.extend(["--init-weights", init_weights])
    if args.loss_mask:
        cmd.append("--loss-mask")
    if max_steps:
        cmd.extend(["--max-steps", str(max_steps)])
    if args.anneal:
        cmd.extend(["--anneal", str(args.anneal)])
    if args.fsdp:
        pass
    elif args.no_fsdp or args.nproc == 1:
        cmd.append("--no-fsdp")
    if trackio_on:
        cmd.append("--trackio")
        cmd.extend(["--trackio-size", size, "--trackio-stage", stage])
        run_id = args.trackio_run_id or os.environ.get("MORENA_PIPELINE_RUN_ID") or ""
        if run_id:
            # Keep run ids filesystem-safe for Trackio naming.
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-")
            cmd.extend(["--trackio-run-id", safe])

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    # Vendor imports modeling_morena as script; common.* needs ROOT on PYTHONPATH.
    existing = env.get("PYTHONPATH", "")
    pieces = [str(ROOT), str(ROOT / "vendor")]
    for piece in pieces:
        if piece not in existing.split(os.pathsep):
            existing = piece + (os.pathsep + existing if existing else "")
    env["PYTHONPATH"] = existing
    print("+", " ".join(cmd), flush=True)
    if trackio_on:
        print(
            f"[trackio] enabled size={size} stage={stage} "
            f"space={env.get('TRACKIO_SPACE_ID', '<unset>')}",
            flush=True,
        )
    return subprocess.call(cmd, env=env, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
