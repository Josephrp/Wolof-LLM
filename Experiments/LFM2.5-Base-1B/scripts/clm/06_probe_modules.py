#!/usr/bin/env python3
"""P4.1 — Probe LFM2.5 Base for LoRA-relevant Linear/Conv1d module suffixes.

Writes docs/lfm_module_names.txt and prints CPT (attn+mlp) vs SFT (attn-only)
proposed target_modules.

Example:
  uv run --group train python scripts/clm/06_probe_modules.py
  uv run --group train python scripts/clm/06_probe_modules.py \\
      --base-id LiquidAI/LFM2.5-350M-Base --allow-cpu
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

try:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM
except ImportError:
    torch = None
    nn = None
    AutoModelForCausalLM = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "lfm_module_names.txt"
DEFAULT_BASE = "LiquidAI/LFM2.5-350M-Base"

from common.tracking import TrackingRun

# Heuristic name fragments for proposing LoRA targets (placeholders until probe runs).
ATTN_HINTS = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "in_proj")
MLP_HINTS = ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2")
CONV_HINTS = ("conv",)


def _suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _propose(suffixes: set[str]) -> tuple[list[str], list[str]]:
    attn = sorted(s for s in suffixes if any(h in s for h in ATTN_HINTS))
    mlp = sorted(s for s in suffixes if any(h in s for h in MLP_HINTS))
    # Prefer exact known names when present
    preferred_attn = [s for s in ("q_proj", "k_proj", "v_proj", "o_proj") if s in suffixes]
    preferred_mlp = [s for s in ("gate_proj", "up_proj", "down_proj") if s in suffixes]
    cpt = preferred_attn + preferred_mlp
    if not cpt:
        cpt = attn + mlp
    sft = preferred_attn or attn
    return cpt, sft


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe LFM module names for LoRA")
    ap.add_argument("--base-id", default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Load on CPU (slow; for hosts without CUDA)",
    )
    args = ap.parse_args()
    state_path = (
        (args.json_out if args.json_out.is_absolute() else ROOT / args.json_out).with_suffix(
            ".trackio.json"
        )
        if args.json_out is not None
        else ROOT / "results" / "full_pipeline" / "tracking" / "probe.json"
    )
    tracking = TrackingRun(
        size=Path(args.base_id).name,
        stage="module-probe",
        tokenizer_family="base",
        method="inspection",
        seed=42,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", Path(args.base_id).name),
        config={"base_id": args.base_id},
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=state_path,
    ).start()
    atexit.register(tracking.finish, success=False)

    if torch is None or nn is None or AutoModelForCausalLM is None:
        print(
            "torch/transformers not available.\n"
            "Install train deps first:\n"
            "  uv sync --group train\n"
            "Or skip probe until a GPU train host is ready.",
            file=sys.stderr,
        )
        return 1

    device_map = "cpu" if args.allow_cpu or not torch.cuda.is_available() else "auto"
    if device_map == "cpu" and not args.allow_cpu and not torch.cuda.is_available():
        print(
            "CUDA not available — pass --allow-cpu to load on CPU, "
            "or run on a GPU host.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {args.base_id} (device_map={device_map}) …")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_id,
            dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as e:  # noqa: BLE001 - Hub/model loaders raise varied errors
        print(f"Failed to load model: {e}", file=sys.stderr)
        return 1

    linear_suffixes: set[str] = set()
    conv_suffixes: set[str] = set()
    expert_module_suffixes: set[str] = set()
    expert_param_suffixes: set[str] = set()
    all_module_types: list[str] = []

    for name, module in model.named_modules():
        cls = type(module).__name__
        if isinstance(module, nn.Linear):
            linear_suffixes.add(_suffix(name))
            if "expert" in name.lower():
                expert_module_suffixes.add(_suffix(name))
            all_module_types.append(f"Linear\t{name}")
        elif isinstance(module, nn.Conv1d) or "Conv1d" in cls:
            conv_suffixes.add(_suffix(name))
            all_module_types.append(f"Conv1d\t{name}")
    for name, _param in model.named_parameters():
        if "expert" in name.lower():
            expert_param_suffixes.add(_suffix(name))

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Module probe: {args.base_id}",
        f"# device_map={device_map}",
        "",
        "## Unique Linear suffixes",
        *[f"- {s}" for s in sorted(linear_suffixes)],
        "",
        "## Unique Conv1d suffixes",
        *[f"- {s}" for s in sorted(conv_suffixes)],
        "",
        "## All Linear/Conv1d modules (name)",
        *all_module_types,
        "",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    cpt, sft = _propose(linear_suffixes | conv_suffixes)
    print("Proposed CPT target_modules (attn + mlp):")
    print(f"  {cpt}")
    print("Proposed SFT target_modules (attn-only):")
    print(f"  {sft}")
    model_class = type(model).__name__
    expert_suffixes = sorted(expert_module_suffixes)
    payload = {
        "base_id": args.base_id,
        "model_class": model_class,
        "architecture": "moe" if "Moe" in model_class else "dense",
        "linear_suffixes": sorted(linear_suffixes),
        "conv_suffixes": sorted(conv_suffixes),
        "expert_suffixes": expert_suffixes,
        "expert_param_suffixes": sorted(expert_param_suffixes),
        "proposed_cpt_targets": cpt,
        "proposed_sft_targets": sft,
    }
    json_out = args.json_out
    if json_out is not None:
        json_out = json_out if json_out.is_absolute() else ROOT / json_out
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {json_out}")
    print("Target validation must pass before training.")
    tracking.log(
        {
            "linear_suffix_count": len(linear_suffixes),
            "conv_suffix_count": len(conv_suffixes),
            "expert_suffix_count": len(expert_suffixes),
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
