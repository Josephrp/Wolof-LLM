"""Load MORENA Transformer + HuggingFace tokenizers.Tokenizer from Hub or local dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import modeling_morena as M  # noqa: E402


SIZE_REPOS = {
    "1.5b-base": "vamboai/morena-1.5b-base",
    "1.5b-instruct": "vamboai/morena-1.5b-instruct",
    "0.5b-mini": "vamboai/morena-0.5b-mini",
    "0.5b-mini-instruct": "vamboai/morena-0.5b-mini-instruct",
    "0.2b-nano": "vamboai/morena-0.2b-nano",
}

GGUF_REPOS = {
    "1.5b-instruct": "vamboai/morena-1.5b-instruct-gguf",
    "0.5b-mini": "vamboai/morena-0.5b-mini-gguf",
    "0.2b-nano": "vamboai/morena-0.2b-nano-gguf",
}


def resolve_model_dir(model_id: str, cache_dir: Path | None = None) -> Path:
    path = Path(model_id)
    if path.is_dir() and (path / "model.safetensors").is_file():
        return path.resolve()
    dest = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "modeling_morena.py",
            "SHA256SUMS",
            "load_example.py",
            "README.md",
        ],
        local_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(dest)


def load_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def load_morena(
    model_id: str,
    *,
    device: str | torch.device | None = None,
    attn_mode: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Path | None = None,
):
    model_dir = resolve_model_dir(model_id, cache_dir=cache_dir)
    cfg = load_config(model_dir)
    mc = M.ModelConfig(**cfg["model"])
    model = M.Transformer(mc, attn_mode)
    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model = model.to(device=device, dtype=dtype).eval()
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    meta = {
        "run": str(model_id),
        "model_dir": str(model_dir),
        "vocab_size": mc.vocab_size,
        "n_layer": mc.n_layer,
        "d_model": mc.d_model,
        "device": str(device),
        "protocol": "morena-native",
    }
    return model, tok, meta, cfg


def download_tokenizer_only(repo_id: str = "vamboai/morena-1.5b-base") -> Path:
    return Path(hf_hub_download(repo_id, "tokenizer.json"))
