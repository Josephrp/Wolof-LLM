#!/usr/bin/env python3
"""Export a MORENA DCP checkpoint directory to a Hub-ready package.

Writes model.safetensors, config.json, tokenizer.json, modeling_morena.py,
SHA256SUMS.json, and a minimal README with YAML front matter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
import modeling_morena as M  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_ckpt(run_dir: Path, step: int | None) -> Path:
    if step is not None:
        d = run_dir / "ckpt" / f"step_{step:08d}"
        if not (d / "meta.json").is_file():
            raise FileNotFoundError(d)
        return d
    latest = run_dir / "ckpt" / "latest.txt"
    if not latest.is_file():
        raise FileNotFoundError(f"no latest.txt under {run_dir}/ckpt")
    step_n = int(latest.read_text(encoding="utf-8").strip())
    d = run_dir / "ckpt" / f"step_{step_n:08d}"
    if not (d / "meta.json").is_file():
        raise FileNotFoundError(d)
    return d


def build_readme(*, title: str, size: str, stage: str, base_hub: str, step: int) -> str:
    meta = {
        "language": ["wo"],
        "license": "other",
        "library_name": "pytorch",
        "pipeline_tag": "text-generation",
        "tags": ["morena", "wolof", "african-languages", "galsenai", stage.lower()],
        "base_model": base_hub,
    }
    import yaml

    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    body = f"""# {title}

Private Wolof continuum artifact for MORENA `{size}` (`{stage}`).

- Protocol: `morena-native` (BPB gold)
- Tokenizer: frozen MORENA 65k (`tokenizer.json`)
- Checkpoint step: `{step}`
- Base: `{base_hub}`
- Runtime: vendored `modeling_morena.py` (plain PyTorch; not Transformers)

Load with `Experiments/MORENA/common/load.py` or the included `modeling_morena.py`.
"""
    return f"---\n{front}\n---\n\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MORENA DCP → Hub package")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="training out dir with ckpt/")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer-repo", default="vamboai/morena-1.5b-base")
    parser.add_argument("--title", default="")
    parser.add_argument("--size", default="morena")
    parser.add_argument("--stage", default="CPT", choices=("CPT", "Instruct", "Distill"))
    parser.add_argument("--base-hub", default="vamboai/morena-1.5b-base")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional direct model.safetensors (skips DCP load)",
    )
    args = parser.parse_args()

    config = args.config if args.config.is_absolute() else ROOT / args.config
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config.read_text(encoding="utf-8"))
    mc = M.ModelConfig(**cfg["model"])
    model = M.Transformer(mc, "sdpa")

    step = args.step or 0
    if args.weights is not None:
        weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
        from safetensors.torch import load_file

        state = load_file(str(weights))
        model.load_state_dict(state, strict=True)
        meta = {"step": step, "source": str(weights)}
    else:
        ckpt = resolve_ckpt(run_dir, args.step)
        meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
        step = int(meta.get("step", args.step or 0))
        # model-only resume (no optimizer / process group required for no-FSDP ckpts)
        import os

        os.environ["MORENA_RESUME_PARTS"] = "model"
        sd = {"model": get_model_state_dict(model)}
        dcp.load(sd, checkpoint_id=str(ckpt))
        set_model_state_dict(model, sd["model"])
        meta["source"] = str(ckpt)

    cpu_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(cpu_state, str(out / "model.safetensors"))

    # Strip train-only keys from published config; keep model + light provenance.
    pub_cfg = {
        "model": cfg["model"],
        "morena_wolof": {
            "size": args.size,
            "stage": args.stage,
            "step": step,
            "base_hub": args.base_hub,
            "protocol": "morena-native",
            "tokenizer_repo": args.tokenizer_repo,
        },
    }
    if "train" in cfg:
        pub_cfg["train_summary"] = {
            k: cfg["train"].get(k)
            for k in ("total_steps", "seq_len", "optimizer", "lr", "seed")
            if k in cfg["train"]
        }
    (out / "config.json").write_text(json.dumps(pub_cfg, indent=2) + "\n", encoding="utf-8")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    tok = Path(hf_hub_download(args.tokenizer_repo, "tokenizer.json"))
    shutil.copy(tok, out / "tokenizer.json")
    shutil.copy(ROOT / "vendor" / "modeling_morena.py", out / "modeling_morena.py")
    # Drop training CLI from published vendor copy is OK; keep full file for load parity.

    title = args.title or f"MORENA {args.size} Wolof {args.stage}"
    (out / "README.md").write_text(
        build_readme(
            title=title,
            size=args.size,
            stage=args.stage,
            base_hub=args.base_hub,
            step=step,
        ),
        encoding="utf-8",
    )

    checksums = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.json":
            checksums[path.name] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
    (out / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(out), "step": step, "n_files": len(checksums)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
