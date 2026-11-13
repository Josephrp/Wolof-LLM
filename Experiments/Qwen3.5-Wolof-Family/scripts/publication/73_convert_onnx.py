#!/usr/bin/env python3
"""Export one Qwen3.5 text checkpoint to Transformers.js-ready ONNX (+ quant variants)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtypes",
        default="fp16,q4,q4f16,quantized",
        help="Comma-separated post-export quant names",
    )
    return parser


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env or os.environ.copy())


def copy_tokenizer_files(src: Path, dest: Path) -> None:
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


def apply_quantizations(onnx_dir: Path, dtypes: list[str]) -> list[str]:
    """Produce Transformers.js-style model_*.onnx siblings via onnxruntime APIs."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

    try:
        from onnxconverter_common import float16 as onnx_float16
    except ImportError:
        onnx_float16 = None

    base = onnx_dir / "model.onnx"
    if not base.is_file():
        # optimum may emit model.onnx or decoder-only naming
        candidates = list(onnx_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"no onnx graph in {onnx_dir}")
        base = candidates[0]
        if base.name != "model.onnx":
            shutil.copy2(base, onnx_dir / "model.onnx")
            data = Path(str(base) + "_data")
            if data.is_file():
                shutil.copy2(data, onnx_dir / "model.onnx_data")
            base = onnx_dir / "model.onnx"

    logs: list[str] = []
    model = onnx.load(str(base))

    if "fp16" in dtypes and onnx_float16 is not None:
        try:
            fp16 = onnx_float16.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(fp16, str(onnx_dir / "model_fp16.onnx"))
            logs.append("fp16: ok")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"fp16: failed ({exc})")

    if "quantized" in dtypes or "q8" in dtypes:
        try:
            quantize_dynamic(
                model_input=str(base),
                model_output=str(onnx_dir / "model_quantized.onnx"),
                weight_type=QuantType.QInt8,
            )
            logs.append("quantized: ok")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"quantized: failed ({exc})")

    if "q4" in dtypes or "q4f16" in dtypes:
        try:
            quantizer = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True)
            quantizer.process()
            q4_model = quantizer.model.model
            q4_path = onnx_dir / "model_q4.onnx"
            onnx.save(q4_model, str(q4_path))
            logs.append("q4: ok")
            if "q4f16" in dtypes and onnx_float16 is not None:
                try:
                    q4f16 = onnx_float16.convert_float_to_float16(q4_model, keep_io_types=True)
                    onnx.save(q4f16, str(onnx_dir / "model_q4f16.onnx"))
                    logs.append("q4f16: ok")
                except Exception as exc:  # noqa: BLE001
                    logs.append(f"q4f16: failed ({exc})")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"q4: failed ({exc})")

    return logs


def patch_transformers_js_config(dest: Path) -> None:
    cfg_path = dest / "config.json"
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    onnx_dir = dest / "onnx"
    external = {}
    for path in onnx_dir.glob("*.onnx"):
        data_files = list(onnx_dir.glob(path.name + "_data*"))
        if data_files:
            external[path.name] = len(data_files)
    cfg["transformers.js_config"] = {
        "use_external_data_format": external or True,
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
    }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not artifact.is_dir():
        raise SystemExit(f"artifact missing: {artifact}")

    export_tmp = output_dir / "_export_tmp"
    if export_tmp.exists():
        shutil.rmtree(export_tmp)
    export_tmp.mkdir(parents=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Text-only causal LM export for Transformers.js pipeline("text-generation").
    run(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            str(artifact),
            "--task",
            "text-generation-with-past",
            "--trust-remote-code",
            str(export_tmp),
        ],
        env=env,
    )

    onnx_dir = output_dir / "onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(parents=True)
    for path in export_tmp.rglob("*.onnx*"):
        rel = path.relative_to(export_tmp)
        target = onnx_dir / rel.name
        shutil.copy2(path, target)

    copy_tokenizer_files(artifact, output_dir)
    # Prefer tokenizer/config from export if present
    for name in ("config.json", "generation_config.json"):
        exported = export_tmp / name
        if exported.is_file():
            shutil.copy2(exported, output_dir / name)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    logs = apply_quantizations(onnx_dir, dtypes)
    patch_transformers_js_config(output_dir)
    shutil.rmtree(export_tmp, ignore_errors=True)

    meta = {
        "updated_at": datetime.now(UTC).isoformat(),
        "artifact": str(artifact),
        "quant_logs": logs,
        "dtypes": dtypes,
        "onnx_files": sorted(p.name for p in onnx_dir.glob("*.onnx")),
    }
    (output_dir / "onnx_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        f"""---
library_name: transformers.js
pipeline_tag: text-generation
tags:
- onnx
- transformers.js
- webgpu
- wolof
- qwen3.5
---

# ONNX export for Transformers.js / WebGPU

Source: `{artifact.name}`

```js
import {{ pipeline }} from "@huggingface/transformers";
const generator = await pipeline("text-generation", "<this-repo>", {{
  device: "webgpu",
  dtype: "q4f16",
}});
```
""",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
