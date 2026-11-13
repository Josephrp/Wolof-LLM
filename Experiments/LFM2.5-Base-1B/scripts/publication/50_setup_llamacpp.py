#!/usr/bin/env python3
"""Clone and build llama.cpp with LFM2 / LFM2MOE support (gate G0)."""

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
DEFAULT_LLAMA = Path("/mnt/volume_w2l8nkp/tools/llama.cpp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup llama.cpp for LFM2 GGUF conversion")
    parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "llamacpp_setup.json",
    )
    return parser


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def which(name: str) -> str | None:
    return shutil.which(name)


def ensure_build_deps() -> None:
    if which("cmake") and which("g++"):
        return
    run(["sudo", "apt-get", "update", "-y"])
    run(
        [
            "sudo",
            "DEBIAN_FRONTEND=noninteractive",
            "apt-get",
            "install",
            "-y",
            "cmake",
            "ninja-build",
            "build-essential",
            "git",
        ]
    )


def resolve_bin(build_dir: Path, name: str) -> Path:
    candidates = [
        build_dir / "bin" / name,
        build_dir / name,
        build_dir / "bin" / f"{name}.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = list(build_dir.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"built binary not found: {name}")


def assert_lfm2_support(llama_root: Path) -> dict[str, bool]:
    needles = {
        "Lfm2ForCausalLM": False,
        "Lfm2MoeForCausalLM": False,
        "LFM2MOE": False,
    }
    search_roots = [
        llama_root / "convert_hf_to_gguf.py",
        llama_root / "conversion",
        llama_root / "src" / "models",
        llama_root / "gguf-py",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".cpp", ".h", ".hpp"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for needle in needles:
                if needle in text:
                    needles[needle] = True
    missing = [name for name, found in needles.items() if not found]
    if missing:
        raise RuntimeError(f"llama.cpp checkout missing LFM2 markers: {missing}")
    return needles


def run_setup(args: argparse.Namespace) -> int:
    ensure_build_deps()
    llama_root = args.llama_root if args.llama_root.is_absolute() else ROOT / args.llama_root
    llama_root.parent.mkdir(parents=True, exist_ok=True)
    if not (llama_root / ".git").exists():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggml-org/llama.cpp.git",
                str(llama_root),
            ]
        )
    else:
        run(["git", "fetch", "--depth", "1", "origin", "master"], cwd=llama_root)
        subprocess.run(
            ["git", "checkout", "origin/master"],
            cwd=llama_root,
            check=False,
        )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=llama_root, text=True
    ).strip()
    commit_date = subprocess.check_output(
        ["git", "log", "-1", "--format=%ci"], cwd=llama_root, text=True
    ).strip()
    markers = assert_lfm2_support(llama_root)

    build_dir = llama_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
    if which("ninja"):
        cmake_cmd[1:1] = ["-G", "Ninja"]
    if which("nvcc"):
        cmake_cmd.append("-DGGML_CUDA=ON")
    run(cmake_cmd, cwd=build_dir)
    jobs = str(os.cpu_count() or 8)
    run(["cmake", "--build", ".", "-j", jobs], cwd=build_dir)

    llama_cli = resolve_bin(build_dir, "llama-cli")
    llama_quantize = resolve_bin(build_dir, "llama-quantize")
    try:
        llama_completion = resolve_bin(build_dir, "llama-completion")
    except FileNotFoundError:
        llama_completion = llama_cli
    convert_script = llama_root / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        # newer layouts may move the converter
        matches = list(llama_root.rglob("convert_hf_to_gguf.py"))
        if not matches:
            raise FileNotFoundError("convert_hf_to_gguf.py not found")
        convert_script = matches[0]

    version_out = subprocess.check_output(
        [str(llama_cli), "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    report = {
        "updated_at": datetime.now(UTC).isoformat(),
        "llama_root": str(llama_root),
        "commit": commit,
        "commit_date": commit_date,
        "markers": markers,
        "llama_cli": str(llama_cli),
        "llama_completion": str(llama_completion),
        "llama_quantize": str(llama_quantize),
        "convert_hf_to_gguf": str(convert_script),
        "version_banner": version_out.strip().splitlines()[:5],
        "cuda": bool(which("nvcc")),
        "gate": "G0",
        "status": "pass",
    }
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    try:
        return run_setup(build_parser().parse_args())
    except (OSError, RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
