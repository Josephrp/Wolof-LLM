#!/usr/bin/env python3
"""Publish intermediate artifacts + checkpoints to private Hub (datasets + models)."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import utc_now

LOCAL_PATH = re.compile(
    r"(?:/mnt/[^\s\"']+|/home/[^\s\"']+|[A-Za-z]:\\[^\s\"']+|<REDACTED_PRIVATE_ROOT>/[^\s\"']+)"
)
TEXT_SUFFIXES = {".md", ".json", ".txt", ".yaml", ".yml", ".jinja"}
STOCK_BY_SIZE = {
    "08b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish Qwen Wolof intermediate artifacts")
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--size", default="08b")
    parser.add_argument(
        "--which",
        nargs="+",
        default=["datasets", "prefs", "tokenizer", "ext_init", "state"],
        choices=[
            "datasets",
            "prefs",
            "tokenizer",
            "ext_init",
            "cpt_adapter",
            "cpt_merged",
            "sft_cpt_adapter",
            "sft_instruct_adapter",
            "instruct_merged",
            "bpb",
            "state",
            "all",
        ],
    )
    return parser


def hub_base_for_local_path(path: str, *, namespace: str, size: str) -> str:
    """Map local base_model paths to Hub ids Hub metadata validation accepts."""
    lower = path.replace("\\", "/").lower()
    size_u = size.upper()
    if "ext-init" in lower or "ext_init" in lower:
        return f"{namespace}/Qwen3.5-{size_u}-Wolof-ext-init-v1"
    if "cpt-v1-merged" in lower or "cpt_merged" in lower:
        return f"{namespace}/Qwen3.5-{size_u}-Wolof-CPT-v1"
    if "instruct-from-cpt-merged" in lower:
        return f"{namespace}/Qwen3.5-{size_u}-Wolof-Instruct-v1"
    return STOCK_BY_SIZE.get(size, "Qwen/Qwen3.5-0.8B")


def rewrite_local_base_models(text: str, *, namespace: str, size: str) -> str:
    local = LOCAL_PATH.pattern
    text = re.sub(
        rf"^base_model:\s*({local})$",
        lambda m: f"base_model: {hub_base_for_local_path(m.group(1), namespace=namespace, size=size)}",
        text,
        flags=re.M,
    )
    text = re.sub(
        rf"base_model:adapter:({local})",
        lambda m: (
            "base_model:adapter:"
            + hub_base_for_local_path(m.group(1), namespace=namespace, size=size)
        ),
        text,
    )
    text = re.sub(
        rf'"base_model_name_or_path"\s*:\s*"({local})"',
        lambda m: (
            '"base_model_name_or_path": "'
            + hub_base_for_local_path(m.group(1), namespace=namespace, size=size)
            + '"'
        ),
        text,
    )
    return text


@contextlib.contextmanager
def sanitized_staging(source: Path, *, namespace: str, size: str):
    """Hard-link weights; rewrite Hub-invalid local base_model paths in text metadata."""
    staging_root = ROOT / "results" / ".publish_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
        staged = Path(temporary) / source.name
        shutil.copytree(source, staged, copy_function=os.link)
        for path in staged.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.stat().st_size > 10 * 1024 * 1024:
                continue
            original = path.read_text(encoding="utf-8", errors="replace")
            redacted = rewrite_local_base_models(original, namespace=namespace, size=size)
            if redacted != original:
                path.unlink()
                path.write_text(redacted, encoding="utf-8")
        yield staged


def publish_folder(
    api: HfApi,
    local: Path,
    repo_id: str,
    *,
    repo_type: str,
    private: bool,
    namespace: str,
    size: str,
) -> None:
    if not local.exists():
        print(f"SKIP missing {local}", flush=True)
        return
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    if local.is_file():
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=local.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {local.name} ({utc_now()})",
        )
    elif repo_type == "model":
        with sanitized_staging(local, namespace=namespace, size=size) as staged:
            api.upload_folder(
                folder_path=str(staged),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload {local.name} ({utc_now()})",
            )
    else:
        api.upload_folder(
            folder_path=str(local),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {local.name} ({utc_now()})",
        )
    print(f"published {local} -> {repo_id} ({repo_type})", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)
    private = not args.public
    ns = args.namespace
    size = args.size
    which = set(args.which)
    if "all" in which:
        which = {
            "datasets",
            "prefs",
            "tokenizer",
            "ext_init",
            "cpt_adapter",
            "cpt_merged",
            "sft_cpt_adapter",
            "sft_instruct_adapter",
            "instruct_merged",
            "bpb",
            "state",
        }

    catalog: list[tuple[str, Path, str, str]] = []
    if "datasets" in which:
        catalog.extend(
            [
                (
                    "datasets",
                    ROOT / "data" / "ortho_ns_mixed_v1" / "cpt",
                    f"{ns}/Qwen3.5-Wolof-NTP-ortho_ns_mixed_v1",
                    "dataset",
                ),
                (
                    "datasets",
                    ROOT / "data" / "ortho_ns_mixed_v1" / "sft_stratified",
                    f"{ns}/Qwen3.5-Wolof-SFT-ortho_ns_mixed_stratified_v1",
                    "dataset",
                ),
                (
                    "datasets",
                    ROOT / "data" / "sft" / "prepared_anchored_v1",
                    f"{ns}/Qwen3.5-Wolof-SFT-prepared_anchored_v1",
                    "dataset",
                ),
            ]
        )
    if "prefs" in which:
        catalog.append(
            (
                "prefs",
                ROOT / "data" / "sft" / "preferences_v1.parquet",
                f"{ns}/Qwen3.5-Wolof-prefs_v1",
                "dataset",
            )
        )
    if "tokenizer" in which:
        catalog.append(
            (
                "tokenizer",
                ROOT / "tokenizers" / "qwen35-wolof-ext-anchors",
                f"{ns}/Qwen3.5-Wolof-tokenizer-ext-anchors-v1",
                "model",
            )
        )
    if "ext_init" in which:
        catalog.append(
            (
                "ext_init",
                ROOT / "models" / "qwen35-0.8b-ext-init"
                if size == "08b"
                else ROOT / "models" / f"qwen35-{size}-ext-init",
                f"{ns}/Qwen3.5-{size.upper()}-Wolof-ext-init-v1",
                "model",
            )
        )
    if "cpt_adapter" in which:
        for cand in (
            ROOT / "results" / "smoke" / f"qwen35_{size}_cpt_smoke-cpt" / "adapter",
            ROOT / f"runs/clm/{size}/v1_ntp70_ortho_lora" / "adapter",
        ):
            if cand.is_dir():
                catalog.append(
                    (
                        "cpt_adapter",
                        cand,
                        f"{ns}/Qwen3.5-{size.upper()}-Wolof-CPT-adapter-v1",
                        "model",
                    )
                )
                break
    if "cpt_merged" in which:
        catalog.append(
            (
                "cpt_merged",
                ROOT / f"models/qwen35-{size}-cpt-v1-merged",
                f"{ns}/Qwen3.5-{size.upper()}-Wolof-CPT-v1",
                "model",
            )
        )
    if "sft_cpt_adapter" in which:
        for cand in (
            ROOT / "results" / "smoke" / f"qwen35_{size}_sft_from_cpt_v1-sft" / "adapter",
            ROOT / f"runs/sft/{size}_from_cpt_v1" / "adapter",
        ):
            if cand.is_dir():
                catalog.append(
                    (
                        "sft_cpt_adapter",
                        cand,
                        f"{ns}/Qwen3.5-{size.upper()}-Wolof-Instruct-adapter-v1",
                        "model",
                    )
                )
                break
    if "sft_instruct_adapter" in which:
        for cand in (
            ROOT
            / "results"
            / "smoke"
            / f"qwen35_{size}_sft_from_instruct_v1-sft"
            / "adapter",
            ROOT / f"runs/sft/{size}_from_instruct_v1" / "adapter",
        ):
            if cand.is_dir():
                catalog.append(
                    (
                        "sft_instruct_adapter",
                        cand,
                        f"{ns}/Qwen3.5-{size.upper()}-Wolof-InstructStock-adapter-v1",
                        "model",
                    )
                )
                break
    if "instruct_merged" in which:
        catalog.append(
            (
                "instruct_merged",
                ROOT / f"models/qwen35-{size}-instruct-from-cpt-merged",
                f"{ns}/Qwen3.5-{size.upper()}-Wolof-Instruct-v1",
                "model",
            )
        )
    if "bpb" in which:
        catalog.append(
            (
                "bpb",
                ROOT / "results" / "bpb",
                f"{ns}/Qwen3.5-Wolof-eval-bpb-v1",
                "dataset",
            )
        )
    if "state" in which:
        catalog.append(
            (
                "state",
                ROOT / "results" / "full_pipeline",
                f"{ns}/Qwen3.5-Wolof-pipeline-state-v1",
                "dataset",
            )
        )

    for _kind, local, repo_id, repo_type in catalog:
        publish_folder(
            api,
            local,
            repo_id,
            repo_type=repo_type,
            private=private,
            namespace=ns,
            size=size,
        )
    print("PUBLISH_ARTIFACTS_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
