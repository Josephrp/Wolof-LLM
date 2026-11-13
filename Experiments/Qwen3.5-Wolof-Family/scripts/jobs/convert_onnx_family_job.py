#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=5.19.0",
#     "accelerate>=1.0.0",
#     "safetensors>=0.4.0",
#     "huggingface_hub>=0.26.0",
#     "optimum[onnx]>=1.24.0",
#     "onnxruntime>=1.20.0",
#     "onnx>=1.17.0",
#     "onnxconverter-common>=1.14.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=4.0.0",
#     "numpy",
# ]
# ///
"""One cheap HF Job: ONNX export for Qwen Wolof Instruct 0.8B/2B/4B."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MODELS = [
    ("Tonic/Qwen3.5-08B-Wolof-Instruct-v1", "Tonic/Qwen3.5-08B-Wolof-Instruct-v1-ONNX"),
    ("Tonic/Qwen3.5-2B-Wolof-Instruct-v1", "Tonic/Qwen3.5-2B-Wolof-Instruct-v1-ONNX"),
    ("Tonic/Qwen3.5-4B-Wolof-Instruct-v1", "Tonic/Qwen3.5-4B-Wolof-Instruct-v1-ONNX"),
]

SINGLE = Path(__file__).with_name("convert_onnx_job.py")


def main() -> int:
    # When uploaded as a single file job, __file__ may be the family script only.
    # Inline-run by importing sibling content via subprocess env loop instead.
    script = Path("/tmp/convert_onnx_job.py")
    if not script.is_file():
        # family script embeds by downloading from Hub? Prefer env SCRIPT_ONNX_SINGLE.
        bundled = os.environ.get("ONNX_SINGLE_SCRIPT")
        if bundled and Path(bundled).is_file():
            script = Path(bundled)
        elif SINGLE.is_file():
            script = SINGLE
        else:
            raise SystemExit("convert_onnx_job.py not found beside family runner")

    only = os.environ.get("ONLY_SRC", "").strip()
    for src, dst in MODELS:
        if only and only not in {src, dst, src.split("/")[-1]}:
            continue
        print(f"=== ONNX {src} -> {dst}", flush=True)
        env = os.environ.copy()
        env["SRC_REPO"] = src
        env["DST_REPO"] = dst
        env["PRIVATE"] = "0"
        env["WORK_DIR"] = f"/tmp/onnx-{src.split('/')[-1]}"
        # Re-exec the single-model UV script body via python (deps already installed).
        code = script.read_text(encoding="utf-8")
        # Strip PEP 723 header for exec
        if code.startswith("# ///"):
            end = code.find("# ///", 5)
            code = code[end + 5 :] if end != -1 else code
        path = Path(f"/tmp/run_{dst.split('/')[-1]}.py")
        path.write_text(code, encoding="utf-8")
        subprocess.run([sys.executable, str(path)], check=True, env=env)
    print("ONNX_FAMILY_JOB_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
