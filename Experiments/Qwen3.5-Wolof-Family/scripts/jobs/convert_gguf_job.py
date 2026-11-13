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
# ///
"""Cheap HF Jobs GGUF matrix export for one Qwen3.5 Wolof checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SRC = os.environ["SRC_REPO"]
DST = os.environ["DST_REPO"]
STEM = os.environ.get("FILE_STEM") or DST.split("/")[-1].removesuffix("-GGUF")
PRIVATE = os.environ.get("PRIVATE", "0") == "1"
# Prefer fewer quants on cheap jobs; override via QUANTS.
QUANTS = [
    q.strip().upper()
    for q in os.environ.get("QUANTS", "F16,Q8_0,Q5_K_M,Q4_K_M,Q4_0").split(",")
    if q.strip()
]
WORK = Path(os.environ.get("WORK_DIR", "/tmp/qwen-gguf"))
LLAMA = Path(os.environ.get("LLAMA_DIR", "/tmp/llama.cpp"))


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_llamacpp() -> tuple[Path, Path]:
    if not LLAMA.is_dir():
        run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA)])
    convert = LLAMA / "convert_hf_to_gguf.py"
    if not convert.is_file():
        raise FileNotFoundError(convert)
    quantize = LLAMA / "build" / "bin" / "llama-quantize"
    if not quantize.is_file():
        run(["cmake", "-B", "build", "-S", ".", "-DGGML_NATIVE=OFF"], cwd=LLAMA)
        run(["cmake", "--build", "build", "--target", "llama-quantize", "-j", "4"], cwd=LLAMA)
    if not quantize.is_file():
        # older layout
        alt = LLAMA / "llama-quantize"
        if alt.is_file():
            quantize = alt
        else:
            raise FileNotFoundError("llama-quantize missing after build")
    # converter deps
    req = LLAMA / "requirements.txt"
    if req.is_file():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])
    return convert, quantize


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    if WORK.exists():
        shutil.rmtree(WORK)
    src_dir = WORK / "src"
    out_dir = WORK / "out"
    src_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    print(f"download {SRC}", flush=True)
    snapshot_download(SRC, local_dir=str(src_dir), token=token)
    convert, quantize = ensure_llamacpp()

    masters = [q for q in QUANTS if q in {"BF16", "F16"}]
    if not masters and any(q not in {"BF16", "F16"} for q in QUANTS):
        masters = ["F16"]
    produced: dict[str, Path] = {}
    for master in masters:
        path = out_dir / f"{STEM}-{master}.gguf"
        if not path.exists():
            run(
                [
                    sys.executable,
                    str(convert),
                    str(src_dir),
                    "--outfile",
                    str(path),
                    "--outtype",
                    master.lower(),
                ]
            )
        produced[master] = path

    source_key = "BF16" if "BF16" in produced else "F16"
    if source_key not in produced:
        path = out_dir / f"{STEM}-{source_key}.gguf"
        run(
            [
                sys.executable,
                str(convert),
                str(src_dir),
                "--outfile",
                str(path),
                "--outtype",
                source_key.lower(),
            ]
        )
        produced[source_key] = path
    source = produced[source_key]

    for qtype in QUANTS:
        if qtype in {"BF16", "F16"}:
            continue
        path = out_dir / f"{STEM}-{qtype}.gguf"
        if not path.exists():
            run([str(quantize), str(source), str(path), qtype])
        produced[qtype] = path

    files = sorted(out_dir.glob(f"{STEM}-*.gguf"))
    checksums = {
        p.name: {"sha256": sha256_file(p), "size": p.stat().st_size} for p in files
    }
    (out_dir / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (out_dir / "gguf_meta.json").write_text(
        json.dumps({"src": SRC, "stem": STEM, "produced": list(produced)}, indent=2) + "\n"
    )
    (out_dir / "README.md").write_text(
        f"""---
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- wolof
- qwen3.5
- galsenai
base_model: {SRC}
---

# {DST.split('/')[-1]}

GGUF export of [`{SRC}`](https://huggingface.co/{SRC}).

Quants: {", ".join(QUANTS)}

```bash
llama-cli -hf {DST}:Q4_K_M -p "Salaam aleekum, naka nga def?"
```
""",
        encoding="utf-8",
    )

    api = HfApi(token=token)
    api.create_repo(DST, repo_type="model", private=PRIVATE, exist_ok=True)
    if not PRIVATE:
        try:
            api.update_repo_settings(repo_id=DST, private=False, repo_type="model")
        except Exception as exc:  # noqa: BLE001
            print("visibility warn", exc, flush=True)
    commit = api.upload_folder(
        repo_id=DST,
        folder_path=str(out_dir),
        repo_type="model",
        commit_message=f"GGUF matrix from {SRC}",
    )
    print(f"DONE {DST}@{commit.oid} files={[p.name for p in files]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
