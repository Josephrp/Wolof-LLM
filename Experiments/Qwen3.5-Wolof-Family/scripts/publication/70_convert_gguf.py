#!/usr/bin/env python3
"""Convert one HF checkpoint dir into a GGUF quant matrix (llama.cpp)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hub_utils import build_checksums, sha256_file

MASTER_TYPES = ("BF16", "F16")
QUANT_FROM_MASTER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HF model dir to GGUF matrix")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-stem", required=True)
    parser.add_argument(
        "--llama-root",
        type=Path,
        default=Path("/mnt/volume_w2l8nkp/tools/llama.cpp"),
    )
    parser.add_argument(
        "--quants",
        default="BF16,F16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0",
    )
    return parser


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, check=True, env=env)


def resolve_tools(llama_root: Path) -> tuple[Path, Path]:
    convert = llama_root / "convert_hf_to_gguf.py"
    quantize = llama_root / "build" / "bin" / "llama-quantize"
    if not convert.is_file():
        raise FileNotFoundError(f"missing {convert}")
    if not quantize.is_file():
        raise FileNotFoundError(f"missing {quantize} — rebuild llama.cpp")
    return convert, quantize


def convert_master(convert_script: Path, artifact: Path, outfile: Path, outtype: str) -> None:
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        print(f"skip existing master {outfile}", flush=True)
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(convert_script),
            str(artifact),
            "--outfile",
            str(outfile),
            "--outtype",
            outtype.lower(),
        ]
    )


def quantize(llama_quantize: Path, source: Path, outfile: Path, qtype: str) -> None:
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        print(f"skip existing quant {outfile}", flush=True)
        return
    run([str(llama_quantize), str(source), str(outfile), qtype])


def main() -> int:
    args = build_parser().parse_args()
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not artifact.is_dir():
        raise SystemExit(f"artifact missing: {artifact}")
    convert_script, llama_quantize = resolve_tools(args.llama_root)
    wanted = [item.strip().upper() for item in args.quants.split(",") if item.strip()]
    stem = args.file_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    produced: dict[str, str] = {}
    for master in MASTER_TYPES:
        if master not in wanted:
            continue
        path = output_dir / f"{stem}-{master}.gguf"
        convert_master(convert_script, artifact, path, master)
        produced[master] = str(path)

    quant_source_key = "BF16" if "BF16" in produced else "F16"
    if any(q in wanted for q in QUANT_FROM_MASTER) and quant_source_key not in produced:
        path = output_dir / f"{stem}-{quant_source_key}.gguf"
        convert_master(convert_script, artifact, path, quant_source_key)
        produced[quant_source_key] = str(path)

    source = Path(produced[quant_source_key])
    for qtype in QUANT_FROM_MASTER:
        if qtype not in wanted:
            continue
        path = output_dir / f"{stem}-{qtype}.gguf"
        quantize(llama_quantize, source, path, qtype)
        produced[qtype] = str(path)

    files = sorted(output_dir.glob(f"{stem}-*.gguf"))
    checksums = build_checksums(output_dir, files)
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    meta = {
        "updated_at": datetime.now(UTC).isoformat(),
        "artifact": str(artifact),
        "file_stem": stem,
        "produced": {key: Path(value).name for key, value in produced.items()},
        "checksums": checksums,
        "quant_source": quant_source_key,
        "config_sha256": sha256_file(artifact / "config.json")
        if (artifact / "config.json").is_file()
        else None,
    }
    (output_dir / "gguf_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "produced": list(produced)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
