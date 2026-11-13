#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=5.19.0",
#     "accelerate>=1.0.0",
#     "safetensors>=0.4.0",
#     "huggingface_hub>=0.26.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=4.0.0",
#     "numpy",
#     "gguf",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""HF Jobs (cpu-xl): GGUF matrix for all Qwen Wolof CPT+Instruct sizes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# Lean quant set to cut job time/cost (still “complete enough” for deploy).
QUANTS = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]
MODELS = [
    ("Tonic/Qwen3.5-08B-Wolof-CPT-v1", "Tonic/Qwen3.5-08B-Wolof-CPT-v1-GGUF"),
    ("Tonic/Qwen3.5-08B-Wolof-Instruct-v1", "Tonic/Qwen3.5-08B-Wolof-Instruct-v1-GGUF"),
    ("Tonic/Qwen3.5-2B-Wolof-CPT-v1", "Tonic/Qwen3.5-2B-Wolof-CPT-v1-GGUF"),
    ("Tonic/Qwen3.5-2B-Wolof-Instruct-v1", "Tonic/Qwen3.5-2B-Wolof-Instruct-v1-GGUF"),
    ("Tonic/Qwen3.5-4B-Wolof-CPT-v1", "Tonic/Qwen3.5-4B-Wolof-CPT-v1-GGUF"),
    ("Tonic/Qwen3.5-4B-Wolof-Instruct-v1", "Tonic/Qwen3.5-4B-Wolof-Instruct-v1-GGUF"),
    ("Tonic/Qwen3.5-9B-Wolof-CPT-v1", "Tonic/Qwen3.5-9B-Wolof-CPT-v1-GGUF"),
    ("Tonic/Qwen3.5-9B-Wolof-Instruct-v1", "Tonic/Qwen3.5-9B-Wolof-Instruct-v1-GGUF"),
]
LLAMA = Path("/tmp/llama.cpp")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_build_tools() -> None:
    if shutil.which("cmake") and shutil.which("git"):
        return
    print("installing build tools…", flush=True)
    run(["bash", "-lc", "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential"])


def ensure_llamacpp() -> tuple[Path, Path]:
    ensure_build_tools()
    if not LLAMA.is_dir():
        run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA)])
    convert = LLAMA / "convert_hf_to_gguf.py"
    quantize = LLAMA / "build" / "bin" / "llama-quantize"
    if not quantize.is_file():
        run(["cmake", "-B", "build", "-S", ".", "-DGGML_NATIVE=OFF"], cwd=LLAMA)
        run(["cmake", "--build", "build", "--target", "llama-quantize", "-j", "4"], cwd=LLAMA)
    if not quantize.is_file():
        raise FileNotFoundError("llama-quantize build failed")
    # Do NOT install llama.cpp/requirements.txt via uv here: the pytorch-cpu index
    # shadows PyPI and breaks requests resolution. Script deps already cover convert.
    return convert, quantize


def convert_one(src: str, dst: str, token: str, convert: Path, quantize: Path) -> None:
    stem = dst.split("/")[-1].removesuffix("-GGUF")
    work = Path(f"/tmp/gguf-{stem}")
    if work.exists():
        shutil.rmtree(work)
    src_dir, out_dir = work / "src", work / "out"
    src_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    print(f"download {src}", flush=True)
    snapshot_download(src, local_dir=str(src_dir), token=token)

    f16 = out_dir / f"{stem}-F16.gguf"
    run([sys.executable, str(convert), str(src_dir), "--outfile", str(f16), "--outtype", "f16"])
    for qtype in QUANTS:
        if qtype == "F16":
            continue
        path = out_dir / f"{stem}-{qtype}.gguf"
        run([str(quantize), str(f16), str(path), qtype])

    files = sorted(out_dir.glob(f"{stem}-*.gguf"))
    checksums = {p.name: {"sha256": sha256_file(p), "size": p.stat().st_size} for p in files}
    (out_dir / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (out_dir / "gguf_meta.json").write_text(
        json.dumps({"src": src, "stem": stem, "quants": QUANTS}, indent=2) + "\n"
    )
    (out_dir / "README.md").write_text(
        f"""---
library_name: gguf
pipeline_tag: text-generation
tags: [gguf, llama.cpp, wolof, qwen3.5, galsenai]
base_model: {src}
---
# {dst.split('/')[-1]}
GGUF export of [`{src}`](https://huggingface.co/{src}). Quants: {", ".join(QUANTS)}.
```bash
llama-cli -hf {dst}:Q4_K_M -p "Salaam aleekum, naka nga def?"
```
""",
        encoding="utf-8",
    )
    api = HfApi(token=token)
    api.create_repo(dst, repo_type="model", private=False, exist_ok=True)
    try:
        api.update_repo_settings(repo_id=dst, private=False, repo_type="model")
    except Exception as exc:  # noqa: BLE001
        print("visibility warn", exc, flush=True)
    commit = api.upload_folder(
        repo_id=dst,
        folder_path=str(out_dir),
        repo_type="model",
        commit_message=f"GGUF matrix from {src}",
    )
    print(f"DONE {dst}@{commit.oid}", flush=True)
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    convert, quantize = ensure_llamacpp()
    only = os.environ.get("ONLY", "").strip().lower()
    for src, dst in MODELS:
        if only and only not in src.lower() and only not in dst.lower():
            continue
        convert_one(src, dst, token, convert, quantize)
    print("GGUF_FAMILY_JOB_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
