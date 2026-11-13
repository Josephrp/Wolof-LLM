#!/usr/bin/env python3
"""Export a MORENA DCP/run directory (or Hub dir) to standalone model.safetensors + tokenizer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
import modeling_morena as M  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="model.safetensors or .pt")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer-repo", default="vamboai/morena-1.5b-base")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mc = M.ModelConfig(**cfg["model"])
    model = M.Transformer(mc, "sdpa")
    if str(args.weights).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(str(args.weights))
    else:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(cpu_state, str(args.out / "model.safetensors"))
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tok = Path(hf_hub_download(args.tokenizer_repo, "tokenizer.json"))
    shutil.copy(tok, args.out / "tokenizer.json")
    shutil.copy(ROOT / "vendor" / "modeling_morena.py", args.out / "modeling_morena.py")
    print(json.dumps({"out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
