#!/usr/bin/env python3
"""Publish a private Tiny Aya *-GGUF Hub repo with checksum readback."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write
from common.publication import (
    build_checksums,
    scan_for_secrets,
    select_namespace,
    sha256_file,
)
from common.tracking import TrackingRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish private Tiny Aya GGUF sidecar repo")
    parser.add_argument("--gguf-dir", type=Path, required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--base-repo", required=True)
    parser.add_argument("--file-stem", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "publication_manifest_gguf.json",
    )
    parser.add_argument("--preferred-namespace", default="Tonic")
    return parser


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "models": {}, "updated_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_readme(gguf_dir: Path, *, repo_name: str, base_repo: str, stage: str) -> Path:
    path = gguf_dir / "README.md"
    body = f"""---
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- wolof
- tiny-aya
- cohere2
- private-production-artifact
base_model: {base_repo}
---

# {repo_name}

Private production artifact (ongoing/experimental).

- Exact base: `{base_repo}`
- Training stage: `{stage}` (+ GGUF export)
- Tokenizer family: Cohere2 / Tiny Aya (see base model card)
- Data caveat: Wolof CPT/SFT mixture; no public leakage claims
- Limitations: experimental; verify generation quality before downstream use
- Private production artifact for gated llama.cpp deployment
- K-quants calibrated with llama-imatrix on the matching ortho NTP test set
- Note: llama.cpp Cohere2 ties output to token embeddings; HF untied `lm_head` is omitted at convert

## Files

GGUF quant matrix: BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0.
Importance-matrix calibration applied to Q6_K / Q5_K_M / Q4_K_M / Q4_0.

## Run

```bash
llama-cli -hf <namespace>/{repo_name}:Q4_K_M -p "Salaam aleekum"
```
"""
    path.write_text(body, encoding="utf-8")
    return path


def sanitize_publish_json(path: Path) -> None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return
    text = path.read_text(encoding="utf-8")
    if "/mnt/" not in text:
        return
    sanitized = re.sub(r"/mnt/(?:volume|data)[^\"'\s]*/", "", text)
    path.write_text(sanitized, encoding="utf-8")


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def run(args: argparse.Namespace) -> int:
    load_trackio_env()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    gguf_dir = args.gguf_dir if args.gguf_dir.is_absolute() else ROOT / args.gguf_dir
    if not gguf_dir.is_dir():
        raise FileNotFoundError(f"gguf dir missing: {gguf_dir}")

    ensure_readme(
        gguf_dir,
        repo_name=args.repo_name,
        base_repo=args.base_repo,
        stage=args.stage,
    )
    for name in ("gguf_meta.json", "verify.json", "hub_verify.json", "metrics.json"):
        sanitize_publish_json(gguf_dir / name)

    gguf_files = sorted(
        path
        for path in gguf_dir.glob(f"{args.file_stem}-*.gguf")
        if "imatrix" not in path.name.lower()
    )
    if not gguf_files:
        raise RuntimeError(f"no GGUF files matching {args.file_stem}-*.gguf")
    meta_files = [
        path
        for name in (
            "README.md",
            "SHA256SUMS.json",
            "gguf_meta.json",
            "verify.json",
            "metrics.json",
        )
        if (path := gguf_dir / name).is_file()
    ]
    # Include imatrix artifact when present (small; useful for requant).
    imatrix = gguf_dir / f"{args.file_stem}-imatrix.gguf"
    if imatrix.is_file():
        meta_files.append(imatrix)
    files = gguf_files + meta_files
    scan_for_secrets(gguf_dir, [path for path in files if path.suffix.lower() in {".md", ".json"}])
    checksums = build_checksums(gguf_dir, files)
    (gguf_dir / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = sorted(set(files + [gguf_dir / "SHA256SUMS.json"]))
    checksums = build_checksums(gguf_dir, files)

    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.preferred_namespace)
    repo_id = f"{namespace}/{args.repo_name}"
    tracking_disabled = not (
        os.environ.get("TRACKIO_SPACE_ID") and os.environ.get("TRACKIO_BUCKET_ID")
    )
    tracking = TrackingRun(
        size=args.size,
        stage=f"publish-gguf-{args.stage.lower()}",
        tokenizer_family="gguf",
        method="private-hub",
        seed=42,
        run_id=os.environ.get("TINYAYA_PIPELINE_RUN_ID", args.repo_name),
        config={"repo_id": repo_id, "base_repo": args.base_repo, "n_files": len(files)},
        production=not tracking_disabled,
        disabled=tracking_disabled,
        state_path=gguf_dir / "trackio_publish.json",
    ).start()

    existing = None
    try:
        existing = api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        existing = None
    if existing is None:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
    else:
        if existing.private is not True:
            raise RuntimeError("refusing to update a non-private repository")

    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(gguf_dir),
        repo_type="model",
        commit_message=f"Publish verified private {args.size} {args.stage} TinyAya GGUF matrix",
        allow_patterns=[path.name for path in files],
    )
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=commit.oid)
    if info.private is not True:
        raise RuntimeError("published repository is not private")

    readback_candidates = [
        name for name in ("README.md", "SHA256SUMS.json", "gguf_meta.json") if name in checksums
    ]
    with tempfile.TemporaryDirectory(prefix="tinyaya-gguf-readback-") as temporary:
        for filename in readback_candidates:
            local = gguf_dir / filename
            remote = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                revision=commit.oid,
                local_dir=temporary,
            )
            if sha256_file(local) != sha256_file(Path(remote)):
                raise RuntimeError(f"Hub readback checksum mismatch: {filename}")

    remote_names = {sibling.rfilename for sibling in (info.siblings or [])}
    expected_names = {path.name for path in gguf_files}
    missing = sorted(expected_names - remote_names)
    if missing:
        raise RuntimeError(f"Hub missing GGUF files: {missing}")

    entry = {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": commit.oid,
        "private": True,
        "base_repo": args.base_repo,
        "gguf_dir": str(gguf_dir.relative_to(ROOT)),
        "checksums": checksums,
        "verified_files": readback_candidates,
        "gguf_files": sorted(expected_names),
        "verified_at": datetime.now(UTC).isoformat(),
    }
    tracking.log(
        {
            "gguf_file_count": len(gguf_files),
            "artifact_total_bytes": sum(int(v["size"]) for v in checksums.values()),
            "verified_readback_files": len(readback_candidates),
            "hub_revision_recorded": 1,
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    entry["trackio"] = tracking.metadata()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("models", {}).setdefault(args.size, {}).setdefault(args.stage, {})
    # Distinguish ns/std by repo name leaf under stage.
    manifest["models"][args.size][args.stage][args.repo_name] = entry
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json_write(manifest_path, manifest)
    print(json.dumps(entry, indent=2)[:2000])
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
