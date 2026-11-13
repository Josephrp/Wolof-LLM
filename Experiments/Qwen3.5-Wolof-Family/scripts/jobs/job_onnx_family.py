#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=5.19.0",
#     "accelerate>=1.0.0",
#     "safetensors>=0.4.0",
#     "huggingface_hub>=0.26.0",
#     "requests>=2.31.0",
#     "onnxruntime>=1.20.0",
#     "onnx>=1.17.0",
#     "onnxconverter-common>=1.14.0",
#     "onnxscript>=0.1.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=4.0.0",
#     "numpy",
#     "packaging",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""HF Jobs (cpu-xl): ONNX export for Instruct 0.8B/2B/4B (qwen3_5_text).

Installs optimum + optimum-onnx from git with --no-deps so transformers>=5.19
is not blocked by their published upper bounds.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = [
    ("Tonic/Qwen3.5-08B-Wolof-Instruct-v1", "Tonic/Qwen3.5-08B-Wolof-Instruct-v1-ONNX"),
    ("Tonic/Qwen3.5-2B-Wolof-Instruct-v1", "Tonic/Qwen3.5-2B-Wolof-Instruct-v1-ONNX"),
    ("Tonic/Qwen3.5-4B-Wolof-Instruct-v1", "Tonic/Qwen3.5-4B-Wolof-Instruct-v1-ONNX"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_optimum_onnx() -> None:
    """Install optimum + Qwen3.5-capable optimum-onnx (unmerged PR fork)."""
    print("installing optimum + optimum-onnx (qwen3_5-pr-clean) --no-deps…", flush=True)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "git+https://github.com/huggingface/optimum.git",
            # Closed-but-working PR: Qwen3_5OnnxConfig for qwen3_5_text hybrid cache.
            "git+https://github.com/eduardoworrel/optimum-onnx.git@qwen3_5-pr-clean",
        ]
    )
    importlib.invalidate_caches()
    importlib.import_module("optimum.exporters.onnx")
    importlib.import_module("optimum.exporters.onnx.model_configs")
    print("optimum.exporters.onnx ready (qwen3_5_text)", flush=True)


def quantize(onnx_dir: Path) -> list[str]:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
    except ImportError:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as MatMul4BitsQuantizer
    try:
        from onnxconverter_common import float16 as onnx_float16
    except ImportError:
        onnx_float16 = None

    bases = list(onnx_dir.glob("*.onnx"))
    if not bases:
        raise FileNotFoundError(onnx_dir)
    base = next((p for p in bases if p.name == "model.onnx"), bases[0])
    if base.name != "model.onnx":
        shutil.copy2(base, onnx_dir / "model.onnx")
        data = Path(str(base) + "_data")
        if data.is_file():
            shutil.copy2(data, onnx_dir / "model.onnx_data")
        base = onnx_dir / "model.onnx"
    logs: list[str] = []
    model = onnx.load(str(base))
    if onnx_float16 is not None:
        try:
            onnx.save(
                onnx_float16.convert_float_to_float16(model, keep_io_types=True),
                str(onnx_dir / "model_fp16.onnx"),
            )
            logs.append("fp16:ok")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"fp16:{exc}")
    try:
        quantize_dynamic(str(base), str(onnx_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)
        logs.append("quantized:ok")
    except Exception as exc:  # noqa: BLE001
        logs.append(f"quantized:{exc}")
    try:
        quantizer = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True)
        quantizer.process()
        q4_model = quantizer.model.model
        onnx.save(q4_model, str(onnx_dir / "model_q4.onnx"))
        logs.append("q4:ok")
        if onnx_float16 is not None:
            try:
                onnx.save(
                    onnx_float16.convert_float_to_float16(q4_model, keep_io_types=True),
                    str(onnx_dir / "model_q4f16.onnx"),
                )
                logs.append("q4f16:ok")
            except Exception as exc:  # noqa: BLE001
                logs.append(f"q4f16:{exc}")
    except Exception as exc:  # noqa: BLE001
        logs.append(f"q4:{exc}")
    return logs


def convert_one(src: str, dst: str, token: str) -> None:
    work = Path(f"/tmp/onnx-{src.split('/')[-1]}")
    if work.exists():
        shutil.rmtree(work)
    src_dir, out_dir, export_tmp = work / "src", work / "out", work / "export"
    for p in (src_dir, out_dir, export_tmp):
        p.mkdir(parents=True)
    print(f"download {src}", flush=True)
    snapshot_download(src, local_dir=str(src_dir), token=token)

    from optimum.exporters.onnx import main_export

    # PR patcher: dynamo path is the validated one for hybrid Gated-DeltaNet.
    os.environ["QWEN35_DYNAMO"] = "1"
    print("exporting via main_export (dynamo=True)…", flush=True)
    main_export(
        model_name_or_path=str(src_dir),
        output=str(export_tmp),
        task="text-generation-with-past",
        trust_remote_code=True,
        do_validation=False,
        device="cpu",
        dynamo=True,
    )

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    for path in export_tmp.rglob("*.onnx*"):
        shutil.copy2(path, onnx_dir / path.name)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        if (src_dir / name).is_file():
            shutil.copy2(src_dir / name, out_dir / name)
    for name in ("config.json", "generation_config.json"):
        if (export_tmp / name).is_file():
            shutil.copy2(export_tmp / name, out_dir / name)
    logs = quantize(onnx_dir)
    cfg_path = out_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        external = {
            p.name: len(list(onnx_dir.glob(p.name + "_data*")))
            for p in onnx_dir.glob("*.onnx")
            if list(onnx_dir.glob(p.name + "_data*"))
        }
        cfg["transformers.js_config"] = {
            "use_external_data_format": external or True,
            "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
        }
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / "onnx_meta.json").write_text(json.dumps({"src": src, "logs": logs}, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        f"""---
library_name: transformers.js
pipeline_tag: text-generation
tags: [onnx, transformers.js, webgpu, wolof, qwen3.5, galsenai]
base_model: {src}
---
# {dst.split('/')[-1]}
ONNX export of [`{src}`](https://huggingface.co/{src}) for Transformers.js WebGPU (`dtype: \"q4f16\"`).
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
        commit_message=f"ONNX WebGPU export from {src}",
    )
    print(f"DONE {dst}@{commit.oid} {logs}", flush=True)
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    ensure_optimum_onnx()
    only = os.environ.get("ONLY", "").strip().lower()
    for src, dst in MODELS:
        if only and only not in src.lower() and only not in dst.lower():
            continue
        convert_one(src, dst, token)
    print("ONNX_FAMILY_JOB_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
