#!/usr/bin/env python3
"""Offline distill: CE on student labels + KL(teacher||student) on packed Wolof shards.

Single-GPU reference implementation for H200. For production-scale runs, prefer
extending vendor modeling_morena with a distill loss; this script covers the plan
gate and smoke budgets.

Logs to Hugging Face Trackio / Gradio when TRACKIO_SPACE_ID is configured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
import modeling_morena as M  # noqa: E402
from common.tracking import TrackingRun, load_trackio_env  # noqa: E402


def load_model(cfg_path: Path, weights: Path, device: torch.device) -> M.Transformer:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = M.Transformer(M.ModelConfig(**cfg["model"]), "sdpa")
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(device=device, dtype=torch.bfloat16)


def iter_windows(shard_dir: Path, seq_len: int):
    index = json.loads((shard_dir / "index.json").read_text(encoding="utf-8"))
    dtype = np.dtype(index["dtype"])
    for shard in index["shards"]:
        mm = np.memmap(shard_dir / shard["file"], dtype=dtype, mode="r")
        n = int(shard["n_tokens"])
        for off in range(0, n - seq_len - 1, seq_len):
            yield np.asarray(mm[off : off + seq_len + 1], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--student-config", required=True, type=Path)
    parser.add_argument("--student-weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="wolof_clm shard dir")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trackio-size", default="distill")
    parser.add_argument("--no-trackio", action="store_true")
    args = parser.parse_args()

    load_trackio_env(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 10 if args.smoke else args.steps
    teacher = load_model(args.teacher_config, args.teacher_weights, device).eval()
    student = load_model(args.student_config, args.student_weights, device).train()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    size = args.trackio_size
    for token in ("0.5b", "0.2b", "mini", "nano"):
        if token in str(args.out).lower() or token in str(args.student_config).lower():
            size = token
            break

    tracking = None
    if not args.no_trackio and os.environ.get("TRACKIO_DISABLED") != "1":
        run_id = os.environ.get("MORENA_PIPELINE_RUN_ID") or args.out.name
        tracking = TrackingRun(
            size=size,
            stage="distill",
            tokenizer_family="65k",
            method="kl-ce",
            seed=1234,
            run_id=run_id,
            config={
                "steps": steps,
                "seq_len": args.seq_len,
                "lr": args.lr,
                "kl_weight": args.kl_weight,
                "temperature": args.temperature,
                "out": str(args.out),
            },
            production=bool(os.environ.get("TRACKIO_SPACE_ID")),
            state_path=args.out / "trackio_run.json",
        ).start()

    windows = iter_windows(args.data, args.seq_len)
    pbar = tqdm(range(steps), desc="distill")
    try:
        for step in pbar:
            try:
                toks = next(windows)
            except StopIteration:
                windows = iter_windows(args.data, args.seq_len)
                toks = next(windows)
            x = torch.tensor(toks[:-1], device=device).unsqueeze(0)
            y = torch.tensor(toks[1:], device=device).unsqueeze(0)
            pos = torch.arange(x.shape[1], device=device).unsqueeze(0)
            with torch.no_grad():
                t_logits = teacher(x, pos, None, x.shape[1], None).float()
            s_logits = student(x, pos, None, x.shape[1], None).float()
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), y.view(-1))
            t = args.temperature
            kl = F.kl_div(
                F.log_softmax(s_logits / t, dim=-1),
                F.softmax(t_logits / t, dim=-1),
                reduction="batchmean",
            ) * (t * t)
            loss = ce + args.kl_weight * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            metrics = {"loss": float(loss), "ce": float(ce), "kl": float(kl)}
            pbar.set_postfix(**metrics)
            if tracking is not None and (step % 5 == 0 or step == steps - 1):
                tracking.log(metrics, step=step)
    except Exception as exc:
        if tracking is not None:
            tracking.finish(success=False, error=exc)
        raise

    args.out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}
    save_file(state, str(args.out / "model.safetensors"))
    cfg = json.loads(args.student_config.read_text(encoding="utf-8"))
    cfg["distill"] = {
        "steps": steps,
        "kl_weight": args.kl_weight,
        "temperature": args.temperature,
        "lr": args.lr,
        "seq_len": args.seq_len,
    }
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    if tracking is not None:
        tracking.log({"distill_done": 1, "steps": steps}, step=steps)
        tracking.finish(success=True)
    print(f"DISTILL_OK {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
