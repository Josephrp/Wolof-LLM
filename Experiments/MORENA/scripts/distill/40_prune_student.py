#!/usr/bin/env python3
"""One-shot structural prune from MORENA 1.5B -> mini (16x1536) or nano (12x1024).

Heuristic: keep first `n_layer` blocks; truncate channel dims with leading
principal axes (slice) for d_model / heads / d_ff. This is a lineage-compatible
*init* for distillation, not a bit-exact replica of Vambo prune_mini.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
import modeling_morena as M  # noqa: E402

TARGETS = {
    "0.5b-mini": {
        "vocab_size": 65536,
        "n_layer": 16,
        "d_model": 1536,
        "n_head": 12,
        "n_kv_head": 4,
        "d_ff": 4096,
        "rope_theta": 500000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "init_std": 0.02,
    },
    "0.2b-nano": {
        "vocab_size": 65536,
        "n_layer": 12,
        "d_model": 1024,
        "n_head": 8,
        "n_kv_head": 4,
        "d_ff": 2816,
        "rope_theta": 500000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "init_std": 0.02,
    },
}


def slice_2d(weight: torch.Tensor, out_dim: int | None, in_dim: int | None) -> torch.Tensor:
    w = weight
    if out_dim is not None:
        w = w[:out_dim]
    if in_dim is not None:
        w = w[:, :in_dim]
    return w.contiguous()


def prune_state(teacher: dict, src_cfg: M.ModelConfig, dst_cfg: M.ModelConfig) -> dict:
    out = {}
    # embeddings
    out["embed.weight"] = teacher["embed.weight"][: dst_cfg.vocab_size, : dst_cfg.d_model].contiguous()
    if not dst_cfg.tie_embeddings and "lm_head.weight" in teacher:
        out["lm_head.weight"] = teacher["lm_head.weight"][: dst_cfg.vocab_size, : dst_cfg.d_model].contiguous()

    src_hd = src_cfg.d_model // src_cfg.n_head
    dst_hd = dst_cfg.d_model // dst_cfg.n_head
    assert src_hd == dst_hd == 128, "head_dim must stay 128"

    out["norm.weight"] = teacher["norm.weight"][: dst_cfg.d_model].contiguous()

    for layer in range(dst_cfg.n_layer):
        prefix = f"layers.{layer}."
        # norms
        out[prefix + "attn_norm.weight"] = teacher[prefix + "attn_norm.weight"][: dst_cfg.d_model]
        out[prefix + "mlp_norm.weight"] = teacher[prefix + "mlp_norm.weight"][: dst_cfg.d_model]
        # attention projections
        out[prefix + "attn.wq.weight"] = slice_2d(
            teacher[prefix + "attn.wq.weight"],
            dst_cfg.n_head * dst_hd,
            dst_cfg.d_model,
        )
        out[prefix + "attn.wk.weight"] = slice_2d(
            teacher[prefix + "attn.wk.weight"],
            dst_cfg.n_kv_head * dst_hd,
            dst_cfg.d_model,
        )
        out[prefix + "attn.wv.weight"] = slice_2d(
            teacher[prefix + "attn.wv.weight"],
            dst_cfg.n_kv_head * dst_hd,
            dst_cfg.d_model,
        )
        out[prefix + "attn.wo.weight"] = slice_2d(
            teacher[prefix + "attn.wo.weight"],
            dst_cfg.d_model,
            dst_cfg.n_head * dst_hd,
        )
        # MLP
        out[prefix + "mlp.w1.weight"] = slice_2d(
            teacher[prefix + "mlp.w1.weight"], dst_cfg.d_ff, dst_cfg.d_model
        )
        out[prefix + "mlp.w3.weight"] = slice_2d(
            teacher[prefix + "mlp.w3.weight"], dst_cfg.d_ff, dst_cfg.d_model
        )
        out[prefix + "mlp.w2.weight"] = slice_2d(
            teacher[prefix + "mlp.w2.weight"], dst_cfg.d_model, dst_cfg.d_ff
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tcfg = json.loads(args.teacher_config.read_text(encoding="utf-8"))
    src = M.ModelConfig(**tcfg["model"])
    dst_dict = TARGETS[args.target]
    dst = M.ModelConfig(**dst_dict)
    teacher = load_file(str(args.teacher_weights))
    student = prune_state(teacher, src, dst)

    # validate load
    model = M.Transformer(dst, "sdpa")
    missing, unexpected = model.load_state_dict(student, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"prune load failed missing={missing[:5]} unexpected={unexpected[:5]}")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(student, str(args.out / "model.safetensors"))
    payload = {"model": dst_dict, "train": tcfg.get("train", {}), "pruned_from": str(args.teacher_weights)}
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "target": args.target, "params_m": model.n_params() / 1e6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
