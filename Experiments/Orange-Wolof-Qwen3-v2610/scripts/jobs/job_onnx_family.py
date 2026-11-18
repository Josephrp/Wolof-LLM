#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.57",
#     "accelerate>=1.0.0",
#     "safetensors>=0.4.0",
#     "huggingface_hub>=0.26.0",
#     "onnxruntime>=1.20.0",
#     "onnx>=1.17.0",
#     "onnxconverter-common>=1.14.0",
#     "onnxscript>=0.1.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=4.0.0",
#     "numpy",
#     "packaging",
#     "optimum[onnxruntime]>=1.24.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""HF Jobs (cpu-xl): ONNX/Transformers.js export for Wolof-Qwen3 0.6B + 4B (browser sizes).

Adapted from Qwen3.5-Wolof-Family/scripts/jobs/job_onnx_family.py with one major
simplification: stock `qwen3` is supported by upstream optimum, so the Qwen3.5
optimum-onnx PR fork and dynamo patcher are NOT installed here.
Dtypes: fp16 (4B export), quantized (QInt8), q4, q4f16 (MatMul4Bits block 32).

Each dtype is evaluated (ORT CPU, chrF + parity) immediately after production;
publishing is delegated to publish_quant.py, which FAILS CLOSED if any
per-dtype eval JSON is missing/stale/failing (Eval Plan §2b).

Cheapest feasible flavor (artifacts retained until the gate passes):
  0.6B: --flavor cpu-basic ($0.01/h)   4B: --flavor cpu-xl ($1.00/h)
Always: --timeout 4h --secrets HF_TOKEN --env ONLY=<size>
Prerequisite: Tier-1 HF eval for that size in Tonic/Wolof-Qwen3-v2610-eval.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def _name_tokens(*names: str) -> set[str]:
    """Size tokens of repo ids, so ONLY=4b matches '4B' but not '14B'."""
    tokens: set[str] = set()
    for name in names:
        tokens.update(re.split(r"[^a-z0-9.]+", name.lower()))
    return tokens

MODELS = [
    ("Orange/Wolof-Qwen3-0.6B-v2610", "a296360960", "Tonic/Wolof-Qwen3-0.6B-v2610-ONNX"),
    ("Orange/Wolof-Qwen3-4B-v2610", "64314c62da", "Tonic/Wolof-Qwen3-4B-v2610-ONNX"),
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def save_onnx_external(model, path: Path) -> None:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + "_data"
    for stale in path.parent.glob(data_name + "*"):
        stale.unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_name, size_threshold=1024)


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
    model = onnx.load(str(base), load_external_data=True)
    if onnx_float16 is not None:
        try:
            save_onnx_external(onnx_float16.convert_float_to_float16(model, keep_io_types=True),
                               onnx_dir / "model_fp16.onnx")
            logs.append("fp16:ok")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"fp16:{exc}")
    try:
        quantize_dynamic(str(base), str(onnx_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)
        qpath = onnx_dir / "model_quantized.onnx"
        if qpath.is_file() and not (onnx_dir / "model_quantized.onnx_data").is_file():
            save_onnx_external(onnx.load(str(qpath), load_external_data=True), qpath)
        logs.append("quantized:ok")
    except Exception as exc:  # noqa: BLE001
        logs.append(f"quantized:{exc}")
    try:
        quantizer = MatMul4BitsQuantizer(model, block_size=32, is_symmetric=True)
        quantizer.process()
        q4_model = quantizer.model.model
        save_onnx_external(q4_model, onnx_dir / "model_q4.onnx")
        logs.append("q4:ok")
        if onnx_float16 is not None:
            try:
                save_onnx_external(onnx_float16.convert_float_to_float16(q4_model, keep_io_types=True),
                                   onnx_dir / "model_q4f16.onnx")
                logs.append("q4f16:ok")
            except Exception as exc:  # noqa: BLE001
                logs.append(f"q4f16:{exc}")
    except Exception as exc:  # noqa: BLE001
        logs.append(f"q4:{exc}")
    return logs


def convert_one(src: str, sha: str, dst: str, token: str) -> None:
    work = Path(f"/tmp/onnx-{src.split('/')[-1]}")
    if work.exists():
        shutil.rmtree(work)
    src_dir, out_dir, export_tmp = work / "src", work / "out", work / "export"
    for p in (src_dir, out_dir, export_tmp):
        p.mkdir(parents=True)
    print(f"download {src}@{sha}", flush=True)
    snapshot_download(src, revision=sha, local_dir=str(src_dir), token=token)

    from optimum.exporters.onnx import main_export

    export_dtype = "fp16" if "-4B-" in src else None  # ORT/WebGPU reject bfloat16
    print(f"exporting via main_export (dtype={export_dtype})...", flush=True)
    kwargs: dict = {"model_name_or_path": str(src_dir), "output": str(export_tmp),
                    "task": "text-generation-with-past", "trust_remote_code": True,
                    "do_validation": False, "device": "cpu"}
    if export_dtype:
        kwargs["dtype"] = export_dtype
    main_export(**kwargs)

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    for path in export_tmp.rglob("*.onnx*"):
        shutil.copy2(path, onnx_dir / path.name)
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        if (src_dir / name).is_file():
            shutil.copy2(src_dir / name, out_dir / name)
    for name in ("config.json", "generation_config.json"):
        if (export_tmp / name).is_file():
            shutil.copy2(export_tmp / name, out_dir / name)
    logs = quantize(onnx_dir)

    # Per-dtype eval (CPU) + eval-gated private publish via publish_quant.py.
    from importlib.metadata import version as pkg_version

    tool_versions = {"onnxruntime": pkg_version("onnxruntime"), "optimum": pkg_version("optimum")}
    repo = Path("/tmp/wolof-llm")
    if not repo.is_dir():
        ref = os.environ.get("WOLOF_REPO_REF", "")
        clone_cmd = ["git", "clone", "--depth", "1"]
        if ref:
            clone_cmd += ["--branch", ref]
        run(clone_cmd + [os.environ.get("WOLOF_REPO_URL", "https://github.com/Josephrp/Wolof-LLM.git"), str(repo)])
    size_key = "06b" if "0.6B" in src else "4b"
    tier1_path = hf_hub_download("Tonic/Wolof-Qwen3-v2610-eval",
                                 f"orange_wolof_qwen3_{size_key}.json",
                                 repo_type="dataset", token=token)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    evaluator = repo / "Experiments/Orange-Wolof-Qwen3-v2610/scripts/publication/eval_onnx_dtype.py"
    for dtype, fname in (("fp16", "model_fp16.onnx"), ("quantized", "model_quantized.onnx"),
                         ("q4", "model_q4.onnx"), ("q4f16", "model_q4f16.onnx")):
        if not (onnx_dir / fname).is_file():
            print(f"EVAL_SKIP {dtype}: {fname} not produced ({logs})", flush=True)
            continue
        run([sys.executable, str(evaluator), "--onnx-dir", str(out_dir), "--file", fname,
             "--dtype", dtype, "--out", str(eval_dir / f"{dtype}_onnx_eval.json")])
    cfg_path = out_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        external = {p.name: len(list(onnx_dir.glob(p.name + "_data*")))
                    for p in onnx_dir.glob("*.onnx") if list(onnx_dir.glob(p.name + "_data*"))}
        cfg["transformers.js_config"] = {"use_external_data_format": external or True,
                                         "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"}}
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (out_dir / "onnx_meta.json").write_text(json.dumps(
        {"src": src, "src_sha": sha, "logs": logs, "tool_versions": tool_versions}, indent=2) + "\n")
    publisher = repo / "Experiments/Orange-Wolof-Qwen3-v2610/scripts/publication/publish_quant.py"
    run([sys.executable, str(publisher), "--staging-dir", str(out_dir), "--kind", "onnx",
         "--src", src, "--src-sha", sha, "--dst-repo", dst,
         "--tier1-json", str(tier1_path), "--eval-dir", str(eval_dir),
         "--meta-json", str(out_dir / "onnx_meta.json")])
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    only = os.environ.get("ONLY", "").strip().lower()
    for src, sha, dst in MODELS:
        if only and only not in _name_tokens(src, dst):
            continue
        convert_one(src, sha, dst, token)
    print("ONNX_FAMILY_JOB_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
