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
"""Cheap HF Jobs ONNX export for one Qwen3.5 Wolof Instruct checkpoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SRC = os.environ["SRC_REPO"]
DST = os.environ["DST_REPO"]
PRIVATE = os.environ.get("PRIVATE", "0") == "1"
WORK = Path(os.environ.get("WORK_DIR", "/tmp/qwen-onnx"))
DTYPES = [d.strip() for d in os.environ.get("DTYPES", "fp16,q4,q4f16,quantized").split(",") if d.strip()]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def copy_meta(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        path = src / name
        if path.is_file():
            shutil.copy2(path, dest / name)


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
        raise FileNotFoundError(f"no onnx in {onnx_dir}")
    base = next((p for p in bases if p.name == "model.onnx"), bases[0])
    if base.name != "model.onnx":
        shutil.copy2(base, onnx_dir / "model.onnx")
        data = Path(str(base) + "_data")
        if data.is_file():
            shutil.copy2(data, onnx_dir / "model.onnx_data")
        base = onnx_dir / "model.onnx"

    logs: list[str] = []
    model = onnx.load(str(base))
    if "fp16" in DTYPES and onnx_float16 is not None:
        try:
            onnx.save(
                onnx_float16.convert_float_to_float16(model, keep_io_types=True),
                str(onnx_dir / "model_fp16.onnx"),
            )
            logs.append("fp16: ok")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"fp16: failed ({exc})")
    if "quantized" in DTYPES:
        try:
            quantize_dynamic(str(base), str(onnx_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)
            logs.append("quantized: ok")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"quantized: failed ({exc})")
    if "q4" in DTYPES or "q4f16" in DTYPES:
        try:
            quantizer = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True)
            quantizer.process()
            q4_model = quantizer.model.model
            onnx.save(q4_model, str(onnx_dir / "model_q4.onnx"))
            logs.append("q4: ok")
            if "q4f16" in DTYPES and onnx_float16 is not None:
                try:
                    onnx.save(
                        onnx_float16.convert_float_to_float16(q4_model, keep_io_types=True),
                        str(onnx_dir / "model_q4f16.onnx"),
                    )
                    logs.append("q4f16: ok")
                except Exception as exc:  # noqa: BLE001
                    logs.append(f"q4f16: failed ({exc})")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"q4: failed ({exc})")
    return logs


def patch_js_config(dest: Path) -> None:
    cfg_path = dest / "config.json"
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text())
    onnx_dir = dest / "onnx"
    external = {}
    for path in onnx_dir.glob("*.onnx"):
        n = len(list(onnx_dir.glob(path.name + "_data*")))
        if n:
            external[path.name] = n
    cfg["transformers.js_config"] = {
        "use_external_data_format": external or True,
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    work = WORK
    if work.exists():
        shutil.rmtree(work)
    src_dir = work / "src"
    out_dir = work / "out"
    export_tmp = work / "export"
    src_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    export_tmp.mkdir(parents=True)

    print(f"download {SRC}", flush=True)
    snapshot_download(SRC, local_dir=str(src_dir), token=token)

    run(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            str(src_dir),
            "--task",
            "text-generation-with-past",
            "--trust-remote-code",
            str(export_tmp),
        ]
    )

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    for path in export_tmp.rglob("*.onnx*"):
        shutil.copy2(path, onnx_dir / path.name)
    copy_meta(src_dir, out_dir)
    for name in ("config.json", "generation_config.json"):
        if (export_tmp / name).is_file():
            shutil.copy2(export_tmp / name, out_dir / name)

    logs = quantize(onnx_dir)
    patch_js_config(out_dir)
    (out_dir / "onnx_meta.json").write_text(json.dumps({"src": SRC, "logs": logs}, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        f"""---
library_name: transformers.js
pipeline_tag: text-generation
tags:
- onnx
- transformers.js
- webgpu
- wolof
- qwen3.5
- galsenai
base_model: {SRC}
---

# {DST.split('/')[-1]}

ONNX export of [`{SRC}`](https://huggingface.co/{SRC}) for Transformers.js / WebGPU.

```js
import {{ pipeline }} from "@huggingface/transformers";
const generator = await pipeline("text-generation", "{DST}", {{
  device: "webgpu",
  dtype: "q4f16",
}});
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
        commit_message=f"ONNX WebGPU export from {SRC}",
    )
    print(f"DONE {DST}@{commit.oid} logs={logs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
