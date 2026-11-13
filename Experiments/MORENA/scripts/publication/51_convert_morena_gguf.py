#!/usr/bin/env python3
"""Convert MORENA safetensors to GGUF via llama.cpp convert script when available.

MORENA ships upstream GGUF with logit-verified conversion. This wrapper:
1) Prefer copying/publishing an existing Hub GGUF for stock models.
2) For adapted checkpoints, invoke llama.cpp convert if the arch is registered;
   otherwise write a conversion_needed.json gate for manual follow-up.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, help="local MORENA dir with model.safetensors")
    parser.add_argument("--hub-gguf-repo", default="", help="optional stock GGUF repo to fetch Q4_K_M")
    parser.add_argument("--gguf-filename", default="")
    parser.add_argument("--llamacpp-root", type=Path, default=Path("/mnt/volume_w2l8nkp/tools/llama.cpp"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.hub_gguf_repo:
        files = list_repo_files(args.hub_gguf_repo)
        name = args.gguf_filename
        if not name:
            cands = [f for f in files if f.endswith(f"-{args.quant}.gguf") or f.endswith(f"_{args.quant}.gguf")]
            if not cands:
                cands = [f for f in files if args.quant.replace("_", "") in f.replace("_", "") and f.endswith(".gguf")]
            if not cands:
                raise FileNotFoundError(f"no {args.quant} gguf in {args.hub_gguf_repo}: {files}")
            name = cands[0]
        local = hf_hub_download(args.hub_gguf_repo, name)
        dest = args.out / Path(name).name
        shutil.copy(local, dest)
        meta = {"source": "hub", "repo": args.hub_gguf_repo, "file": name, "path": str(dest)}
        (args.out / "gguf_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return 0

    if not args.model_dir or not (args.model_dir / "model.safetensors").is_file():
        raise SystemExit("--model-dir with model.safetensors required when not using --hub-gguf-repo")

    convert = args.llamacpp_root / "convert_hf_to_gguf.py"
    if not convert.is_file():
        convert = args.llamacpp_root / "convert-hf-to-gguf.py"
    if not convert.is_file():
        note = {
            "status": "conversion_needed",
            "reason": "llama.cpp convert script not found; MORENA is not Transformers-native",
            "model_dir": str(args.model_dir),
            "hint": "Use upstream Vambo conversion path or register MORENA arch in llama.cpp",
        }
        (args.out / "conversion_needed.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(note, indent=2))
        return 0

    f16 = args.out / "morena-f16.gguf"
    cmd = [sys.executable, str(convert), str(args.model_dir), "--outfile", str(f16), "--outtype", "f16"]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code != 0:
        note = {
            "status": "conversion_failed",
            "code": code,
            "hint": "MORENA may need a custom llama.cpp converter (upstream GGUFs are logit-verified)",
        }
        (args.out / "conversion_needed.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
        return 0

    quantize = args.llamacpp_root / "build" / "bin" / "llama-quantize"
    qout = args.out / f"morena-{args.quant}.gguf"
    if quantize.is_file():
        subprocess.check_call([str(quantize), str(f16), str(qout), args.quant])
    meta = {"source": "local_convert", "f16": str(f16), "quant": str(qout) if qout.is_file() else None}
    (args.out / "gguf_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
