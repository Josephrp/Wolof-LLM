#!/usr/bin/env python3
"""Convert one Tiny Aya HF merge to a GGUF matrix with imatrix-calibrated K-quants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.publication import build_checksums, sha256_file

MASTER_TYPES = ("BF16", "F16")
QUANT_FROM_MASTER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0")
DEFAULT_IMATRIX_QUANTS = ("Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Tiny Aya HF model to calibrated GGUF matrix")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-stem", required=True)
    parser.add_argument("--llama-root", type=Path, default=Path("/mnt/volume_w2l8nkp/tools/llama.cpp"))
    parser.add_argument(
        "--quants",
        default="BF16,Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_0",
        help="Comma-separated GGUF types",
    )
    parser.add_argument(
        "--imatrix-quants",
        default="Q6_K,Q5_K_M,Q4_K_M,Q4_0",
        help="Comma-separated quant types that use the importance matrix",
    )
    parser.add_argument("--calib-csv", type=Path, required=True)
    parser.add_argument("--calib-text-column", default="text")
    parser.add_argument("--calib-n-texts", type=int, default=512)
    parser.add_argument("--calib-chunks", type=int, default=128)
    parser.add_argument("--calib-ctx", type=int, default=1024)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--apply-verbalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/mnt/volume_w2l8nkp/Wolof-LLM-main/Evaluation"),
    )
    parser.add_argument("--skip-imatrix", action="store_true")
    return parser


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, check=True, env=env)


def convert_master(
    *,
    convert_script: Path,
    artifact: Path,
    outfile: Path,
    outtype: str,
) -> None:
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        print(f"skip existing master {outfile}", flush=True)
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).resolve().parent / "51_convert_hf_to_gguf_cohere2.py"
    run(
        [
            sys.executable,
            str(wrapper),
            str(convert_script),
            str(artifact),
            "--outfile",
            str(outfile),
            "--outtype",
            outtype.lower(),
        ]
    )


def build_calib_text(
    *,
    calib_csv: Path,
    out_txt: Path,
    text_column: str,
    n_texts: int,
    apply_verbalization: bool,
    eval_root: Path,
) -> dict:
    if not calib_csv.is_file():
        raise FileNotFoundError(f"calib csv missing: {calib_csv}")
    verbalize = None
    if apply_verbalization:
        sys.path.insert(0, str(eval_root / "scripts"))
        try:
            from verbalizer import verbalize_text  # type: ignore

            verbalize = verbalize_text
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: verbalizer unavailable ({exc}); writing raw calib text", flush=True)

    texts: list[str] = []
    with calib_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"column {text_column!r} missing in {calib_csv}")
        for row in reader:
            raw = (row.get(text_column) or "").strip()
            if not raw:
                continue
            texts.append(verbalize(raw) if verbalize else raw)
            if n_texts > 0 and len(texts) >= n_texts:
                break
    if not texts:
        raise RuntimeError(f"no calib texts from {calib_csv}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    return {
        "calib_csv": str(calib_csv),
        "n_texts": len(texts),
        "apply_verbalization": bool(verbalize),
        "path": str(out_txt),
    }


def run_imatrix(
    *,
    llama_imatrix: Path,
    master: Path,
    calib_txt: Path,
    outfile: Path,
    ctx: int,
    chunks: int,
    n_gpu_layers: int,
) -> None:
    if outfile.exists() and outfile.stat().st_size > 1000:
        print(f"skip existing imatrix {outfile}", flush=True)
        return
    cmd = [
        str(llama_imatrix),
        "-m",
        str(master),
        "-f",
        str(calib_txt),
        "-o",
        str(outfile),
        "-c",
        str(ctx),
        "-ngl",
        str(n_gpu_layers),
        "--chunks",
        str(chunks),
        "--process-output",
    ]
    run(cmd)


def quantize(
    *,
    llama_quantize: Path,
    source: Path,
    outfile: Path,
    qtype: str,
    imatrix: Path | None,
) -> None:
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        print(f"skip existing quant {outfile}", flush=True)
        return
    cmd = [str(llama_quantize)]
    if imatrix is not None:
        cmd.extend(["--imatrix", str(imatrix)])
    cmd.extend([str(source), str(outfile), qtype])
    run(cmd)


def run_convert(args: argparse.Namespace) -> int:
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not artifact.is_dir():
        raise FileNotFoundError(f"artifact missing: {artifact}")

    llama_root = args.llama_root
    convert_script = llama_root / "convert_hf_to_gguf.py"
    llama_quantize = llama_root / "build" / "bin" / "llama-quantize"
    llama_imatrix = llama_root / "build" / "bin" / "llama-imatrix"
    if not convert_script.is_file():
        raise FileNotFoundError(convert_script)
    if not llama_quantize.is_file():
        raise FileNotFoundError(llama_quantize)

    wanted = [item.strip().upper() for item in args.quants.split(",") if item.strip()]
    imatrix_quants = {
        item.strip().upper()
        for item in args.imatrix_quants.split(",")
        if item.strip()
    } or set(DEFAULT_IMATRIX_QUANTS)
    stem = args.file_stem
    produced: dict[str, str] = {}

    for master in MASTER_TYPES:
        if master not in wanted:
            continue
        path = output_dir / f"{stem}-{master}.gguf"
        convert_master(
            convert_script=convert_script,
            artifact=artifact,
            outfile=path,
            outtype=master,
        )
        produced[master] = str(path)

    quant_source_key = "BF16" if "BF16" in produced else "F16"
    if any(q in wanted for q in QUANT_FROM_MASTER) and quant_source_key not in produced:
        path = output_dir / f"{stem}-{quant_source_key}.gguf"
        convert_master(
            convert_script=convert_script,
            artifact=artifact,
            outfile=path,
            outtype=quant_source_key,
        )
        produced[quant_source_key] = str(path)

    source = Path(produced[quant_source_key])
    imatrix_path: Path | None = None
    calib_meta: dict | None = None
    need_imatrix = (not args.skip_imatrix) and any(
        q in wanted and q in imatrix_quants for q in QUANT_FROM_MASTER
    )
    if need_imatrix:
        if not llama_imatrix.is_file():
            raise FileNotFoundError(llama_imatrix)
        calib_csv = args.calib_csv if args.calib_csv.is_absolute() else ROOT / args.calib_csv
        calib_txt = output_dir / f"{stem}-calib.txt"
        calib_meta = build_calib_text(
            calib_csv=calib_csv,
            out_txt=calib_txt,
            text_column=args.calib_text_column,
            n_texts=args.calib_n_texts,
            apply_verbalization=args.apply_verbalization,
            eval_root=args.eval_root,
        )
        imatrix_path = output_dir / f"{stem}-imatrix.gguf"
        run_imatrix(
            llama_imatrix=llama_imatrix,
            master=source,
            calib_txt=calib_txt,
            outfile=imatrix_path,
            ctx=args.calib_ctx,
            chunks=args.calib_chunks,
            n_gpu_layers=args.n_gpu_layers,
        )

    for qtype in QUANT_FROM_MASTER:
        if qtype not in wanted:
            continue
        path = output_dir / f"{stem}-{qtype}.gguf"
        use_imatrix = imatrix_path if (qtype in imatrix_quants and imatrix_path is not None) else None
        quantize(
            llama_quantize=llama_quantize,
            source=source,
            outfile=path,
            qtype=qtype,
            imatrix=use_imatrix,
        )
        produced[qtype] = str(path)

    files = sorted(output_dir.glob(f"{stem}-*.gguf"))
    checksums = build_checksums(output_dir, files)
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    meta = {
        "updated_at": datetime.now(UTC).isoformat(),
        "artifact": str(artifact.relative_to(ROOT)) if artifact.is_relative_to(ROOT) else artifact.name,
        "artifact_config_sha256": sha256_file(artifact / "config.json")
        if (artifact / "config.json").is_file()
        else None,
        "file_stem": stem,
        "llama_root": str(llama_root),
        "produced": {key: Path(value).name for key, value in produced.items()},
        "checksums": checksums,
        "quant_source": quant_source_key,
        "wanted": wanted,
        "imatrix": None
        if imatrix_path is None
        else {
            "path": imatrix_path.name,
            "quants": sorted(imatrix_quants & set(wanted)),
            "calib": calib_meta,
        },
        "cohere2_note": (
            "llama.cpp Cohere2 aliases output to token_embd; untied HF lm_head is omitted at convert."
        ),
    }
    (output_dir / "gguf_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "produced": list(produced), "imatrix": meta["imatrix"]}, indent=2))
    return 0


def main() -> int:
    try:
        return run_convert(build_parser().parse_args())
    except (OSError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
