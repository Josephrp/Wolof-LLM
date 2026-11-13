#!/usr/bin/env python3
"""Smoke-verify a GGUF with llama-cli (gate G2/G3/G4/G6)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify GGUF load + short generation")
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument(
        "--hf",
        default=None,
        help="Hub model ref for llama-cli -hf, e.g. Tonic/LFM2.5-230M-Wolof-Instruct-GGUF:Q4_K_M",
    )
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument(
        "--setup-report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "llamacpp_setup.json",
    )
    parser.add_argument("--prompt", default="Salaam aleekum, naka nga def?")
    parser.add_argument("--n-predict", type=int, default=48)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def resolve_cli(args: argparse.Namespace) -> Path:
    if args.llama_cli:
        path = args.llama_cli if args.llama_cli.is_absolute() else ROOT / args.llama_cli
        if path.is_file():
            return path
    setup_path = (
        args.setup_report if args.setup_report.is_absolute() else ROOT / args.setup_report
    )
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    # Prefer llama-completion for non-interactive generation; fall back to llama-cli.
    for key in ("llama_completion", "llama_cli"):
        candidate = setup.get(key)
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("no llama-completion/llama-cli in setup report")


def resolve_hf_quant(hf_ref: str) -> Path:
    """Resolve a Hub GGUF quant locally (prefer existing export, else Hub download)."""
    if ":" not in hf_ref:
        raise ValueError(f"expected repo:quant in --hf, got {hf_ref}")
    repo_id, quant = hf_ref.rsplit(":", 1)
    # LiquidAI-style filename stem from repo name without -GGUF suffix
    stem = repo_id.split("/")[-1]
    if stem.endswith("-GGUF"):
        stem = stem[: -len("-GGUF")]
    filename = f"{stem}-{quant}.gguf"
    local_matches = sorted((ROOT / "results" / "gguf").rglob(filename))
    if local_matches:
        return local_matches[0]
    local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    return Path(local)


def run_verify(args: argparse.Namespace) -> int:
    if not args.gguf and not args.hf:
        raise ValueError("provide --gguf and/or --hf")
    llama_cli = resolve_cli(args)
    cmd = [str(llama_cli)]
    gguf_label = None
    hf_download_fallback = False
    if args.hf and args.gguf:
        raise ValueError("provide only one of --gguf or --hf")
    if args.hf:
        gguf_label = args.hf
        cmd.extend(["-hf", args.hf])
    if args.gguf:
        gguf = args.gguf if args.gguf.is_absolute() else ROOT / args.gguf
        if not gguf.is_file():
            raise FileNotFoundError(f"missing gguf: {gguf}")
        cmd.extend(["-m", str(gguf)])
        gguf_label = str(gguf)
    # -no-cnv: disable chat conversation mode (required for one-shot completion).
    # --single-turn: exit after the predefined prompt turn.
    cmd.extend(
        [
            "-p",
            args.prompt,
            "-n",
            str(args.n_predict),
            "--temp",
            "0.1",
            "-no-cnv",
            "--single-turn",
        ]
    )
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if args.hf and ("https is not supported" in combined.lower() or "--model is required" in combined.lower()):
        print("WARN: llama.cpp -hf HTTPS unavailable; falling back to hf_hub_download", flush=True)
        local_gguf = resolve_hf_quant(args.hf)
        hf_download_fallback = True
        cmd = [
            str(llama_cli),
            "-m",
            str(local_gguf),
            "-p",
            args.prompt,
            "-n",
            str(args.n_predict),
            "--temp",
            "0.1",
            "-no-cnv",
            "--single-turn",
        ]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        gguf_label = f"{args.hf} -> {local_gguf.name}"
    lower = combined.lower()
    failures = []
    if proc.returncode != 0:
        failures.append(f"exit_code={proc.returncode}")
    for needle in (
        "unknown model architecture",
        "failed to load",
        "error loading model",
        "failed to initialize",
        "invalid argument",
    ):
        if needle in lower:
            failures.append(needle)
    # require some generated text beyond the prompt echo
    generated_ok = len((proc.stdout or "").strip()) >= max(8, len(args.prompt) // 2)
    if not generated_ok:
        failures.append("empty_or_short_generation")

    result = {
        "updated_at": datetime.now(UTC).isoformat(),
        "gguf": Path(gguf_label).name
        if gguf_label and not str(gguf_label).startswith(("Tonic/", "http")) and " -> " not in str(gguf_label)
        else gguf_label,
        "hf": args.hf,
        "hf_download_fallback": hf_download_fallback,
        "llama_cli": Path(str(llama_cli)).name,
        "prompt": args.prompt,
        "n_predict": args.n_predict,
        "exit_code": proc.returncode,
        "ok": not failures,
        "failures": failures,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
    out = args.out
    if out is None:
        if args.gguf:
            gguf_path = args.gguf if args.gguf.is_absolute() else ROOT / args.gguf
            out = gguf_path.with_suffix(".verify.json")
        else:
            out = ROOT / "results" / "full_pipeline" / "hub_verify_tmp.json"
    else:
        out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("ok", "failures", "gguf", "hf", "hf_download_fallback")}, indent=2))
    if failures:
        raise RuntimeError(f"gguf verify failed: {failures}")
    return 0


def main() -> int:
    try:
        return run_verify(build_parser().parse_args())
    except (OSError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
