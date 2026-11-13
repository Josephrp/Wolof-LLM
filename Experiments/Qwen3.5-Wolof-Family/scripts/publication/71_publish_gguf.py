#!/usr/bin/env python3
"""Publish a *-GGUF Hub sidecar (public by default for Qwen Wolof)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.hub_utils import build_checksums, scan_for_secrets, select_namespace, sha256_file
from common.pipeline import atomic_json_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--private", action="store_true")
    return parser


def ensure_readme(gguf_dir: Path, *, repo_name: str, base_repo: str, stage: str) -> None:
    path = gguf_dir / "README.md"
    path.write_text(
        f"""---
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- wolof
- qwen3.5
- galsenai
base_model: {base_repo}
---

# {repo_name}

GGUF export of [`{base_repo}`](https://huggingface.co/{base_repo}) ({stage}).

Quant matrix: BF16, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0.

```bash
llama-cli -hf Tonic/{repo_name}:Q4_K_M -p "Salaam aleekum, naka nga def?"
```
""",
        encoding="utf-8",
    )


def sanitize_json(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "/mnt/" not in text:
        return
    path.write_text(re.sub(r"/mnt/(?:volume|data)[^\"'\\s]*/", "", text), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    gguf_dir = args.gguf_dir if args.gguf_dir.is_absolute() else ROOT / args.gguf_dir
    ensure_readme(gguf_dir, repo_name=args.repo_name, base_repo=args.base_repo, stage=args.stage)
    for name in ("gguf_meta.json", "SHA256SUMS.json"):
        sanitize_json(gguf_dir / name)
    gguf_files = sorted(gguf_dir.glob(f"{args.file_stem}-*.gguf"))
    if not gguf_files:
        raise SystemExit(f"no GGUF files for stem {args.file_stem}")
    meta = [gguf_dir / n for n in ("README.md", "SHA256SUMS.json", "gguf_meta.json") if (gguf_dir / n).is_file()]
    files = gguf_files + meta
    scan_for_secrets(gguf_dir, [p for p in files if p.suffix.lower() in {".md", ".json"}])
    checksums = build_checksums(gguf_dir, files)
    (gguf_dir / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(set(files + [gguf_dir / "SHA256SUMS.json"]))

    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.preferred_namespace)
    repo_id = f"{namespace}/{args.repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    if not args.private:
        try:
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model")
        except Exception as exc:  # noqa: BLE001
            print(f"visibility warn: {exc}", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(gguf_dir),
        repo_type="model",
        commit_message=f"Publish {args.size} {args.stage} GGUF matrix",
        allow_patterns=[p.name for p in files],
    )
    entry = {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": commit.oid,
        "private": bool(args.private),
        "base_repo": args.base_repo,
        "gguf_files": sorted(p.name for p in gguf_files),
        "verified_at": datetime.now(UTC).isoformat(),
        "sha256_readme": sha256_file(gguf_dir / "README.md"),
    }
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = {"schema_version": 1, "models": {}}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("models", {}).setdefault(args.size, {}).setdefault(args.stage, {})["gguf"] = entry
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json_write(manifest_path, manifest)
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
