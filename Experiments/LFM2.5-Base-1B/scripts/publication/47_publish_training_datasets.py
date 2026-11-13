#!/usr/bin/env python3
"""Publish used CPT/SFT training datasets (mixture + prepared) to private Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_private_dataset(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if info.private is not True:
        raise RuntimeError(f"refusing non-private dataset: {repo_id}")


def write_readme(path: Path, *, title: str, body: str, tags: list[str]) -> None:
    front = "\n".join(
        [
            "---",
            "language:",
            "- wo",
            "- fr",
            "- en",
            "license: other",
            "pretty_name: " + title,
            "task_categories:",
            "- text-generation",
            "tags:",
            *[f"- {tag}" for tag in tags],
            "---",
            "",
        ]
    )
    path.write_text(front + body.rstrip() + "\n", encoding="utf-8")


def publish_folder(api: HfApi, repo_id: str, folder: Path, *, message: str) -> str:
    ensure_private_dataset(api, repo_id)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
    return commit.oid


def publish_cpt_card(api: HfApi, namespace: str) -> dict:
    """CPT used Hub cleaned_data_for_clm directly; publish a provenance card + pointer."""
    repo_id = f"{namespace}/LFM2.5-Wolof-CPT-datamix-used"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_readme(
            root / "README.md",
            title="LFM2.5 Wolof CPT datamix (used)",
            tags=["wolof", "cpt", "lfm2.5", "provenance"],
            body=f"""
# LFM2.5 Wolof CPT datamix (used)

Continual pretraining for all LFM2.5 Wolof CPT models used this Hub dataset
**directly** (no local re-mix):

- Source: [`galsenai/cleaned_data_for_clm`](https://huggingface.co/datasets/galsenai/cleaned_data_for_clm)
- Splits: `train` 958,646 · `test` 4,282
- Filtering already applied upstream: code-span strip from Code-170k lines,
  NFC normalization, seed-42 eval split, near-dup removal from eval (→ 4,282)

This repository is the **canonical provenance record** for LFM CPT runs under
`Tonic/LFM2.5-*-Wolof-CPT`. Rows are not re-uploaded here to avoid duplication;
load the source dataset for training/eval.

```python
from datasets import load_dataset
ds = load_dataset("galsenai/cleaned_data_for_clm")
```

Published: {utc_now()}
""",
        )
        meta = {
            "schema_version": 1,
            "role": "cpt",
            "used_by": "LFM2.5 Wolof CPT family",
            "source_dataset": "galsenai/cleaned_data_for_clm",
            "n_train": 958646,
            "n_test": 4282,
            "local_remix": False,
            "updated_at": utc_now(),
        }
        (root / "datamix_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        rev = publish_folder(
            api,
            repo_id,
            root,
            message="Publish CPT datamix provenance for LFM2.5 Wolof",
        )
    return {"repo_id": repo_id, "revision": rev, "kind": "cpt-provenance"}


def publish_parquet_dataset(
    api: HfApi,
    *,
    repo_id: str,
    source_dir: Path,
    title: str,
    role: str,
    extra_meta: dict | None = None,
    include_globs: tuple[str, ...] = ("*.parquet", "*.json", "*.jinja"),
) -> dict:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        copied = []
        for pattern in include_globs:
            for path in sorted(source_dir.glob(pattern)):
                if path.name == "trackio.json":
                    continue
                dest = data_dir / path.name if path.suffix == ".parquet" else root / path.name
                shutil.copy2(path, dest)
                copied.append(dest.relative_to(root).as_posix())
        if not any(p.suffix == ".parquet" for p in (root / "data").glob("*.parquet")):
            # also allow parquet at root of source
            for path in sorted(source_dir.glob("*.parquet")):
                shutil.copy2(path, data_dir / path.name)
                copied.append(f"data/{path.name}")
        meta = {
            "schema_version": 1,
            "role": role,
            "source_dir": str(source_dir),
            "files": copied,
            "updated_at": utc_now(),
            **(extra_meta or {}),
        }
        (root / "datamix_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Prefer Hub dataset viewer layout for train/val/test parquet names.
        train = data_dir / "train.parquet"
        val = data_dir / "val.parquet"
        test = data_dir / "test.parquet"
        configs = []
        if train.is_file():
            configs.append(("train", "data/train.parquet"))
        if val.is_file():
            configs.append(("validation", "data/val.parquet"))
        if test.is_file():
            configs.append(("test", "data/test.parquet"))
        config_yaml = ""
        if configs:
            config_yaml = "configs:\n- config_name: default\n  data_files:\n"
            for split, path in configs:
                config_yaml += f"  - split: {split}\n    path: {path}\n"
        write_readme(
            root / "README.md",
            title=title,
            tags=["wolof", "sft", "lfm2.5", role],
            body=f"""
# {title}

Private snapshot of the **exact** local artifacts used (or prepared) for LFM2.5
Wolof training.

{config_yaml}
## Files

```text
{chr(10).join(copied)}
```

## Meta

```json
{json.dumps(meta, indent=2, ensure_ascii=False)}
```

Published: {utc_now()}
""",
        )
        rev = publish_folder(
            api,
            repo_id,
            root,
            message=f"Publish {role} datamix snapshot",
        )
    return {"repo_id": repo_id, "revision": rev, "kind": role, "files": copied}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "training_datasets_manifest.json",
    )
    parser.add_argument("--skip-cpt", action="store_true")
    parser.add_argument("--skip-sft-used", action="store_true")
    parser.add_argument("--include-stratified-prepared", action="store_true")
    parser.add_argument("--include-ortho-ns-mixed", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)

    results = []
    if not args.skip_cpt:
        results.append(publish_cpt_card(api, args.namespace))

    if not args.skip_sft_used:
        mixture = ROOT / "data" / "sft" / "mixture_v1"
        prepared = ROOT / "data" / "sft" / "prepared_mixture_v1"
        prepared_128k = ROOT / "data" / "sft" / "prepared_mixture_v1_128k"
        meta_path = mixture / "mixture_meta.json"
        extra = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        results.append(
            publish_parquet_dataset(
                api,
                repo_id=f"{args.namespace}/LFM2.5-Wolof-SFT-mixture_v1-used",
                source_dir=mixture,
                title="LFM2.5 Wolof SFT mixture_v1 (used)",
                role="sft-mixture-used",
                extra_meta={"mixture_meta": extra},
            )
        )
        results.append(
            publish_parquet_dataset(
                api,
                repo_id=f"{args.namespace}/LFM2.5-Wolof-SFT-prepared_mixture_v1-used",
                source_dir=prepared,
                title="LFM2.5 Wolof SFT prepared_mixture_v1 65k (used)",
                role="sft-prepared-65k-used",
            )
        )
        results.append(
            publish_parquet_dataset(
                api,
                repo_id=f"{args.namespace}/LFM2.5-Wolof-SFT-prepared_mixture_v1_128k-used",
                source_dir=prepared_128k,
                title="LFM2.5 Wolof SFT prepared_mixture_v1_128k (used)",
                role="sft-prepared-128k-used",
            )
        )

    if args.include_stratified_prepared:
        for local, repo_suffix, role in [
            (
                ROOT / "data" / "sft" / "stratified_v1",
                "LFM2.5-Wolof-SFT-stratified_v1-raw",
                "sft-stratified-raw",
            ),
            (
                ROOT / "data" / "sft" / "prepared_stratified_v1",
                "LFM2.5-Wolof-SFT-prepared_stratified_v1-65k",
                "sft-prepared-stratified-65k",
            ),
            (
                ROOT / "data" / "sft" / "prepared_stratified_v1_128k",
                "LFM2.5-Wolof-SFT-prepared_stratified_v1-128k",
                "sft-prepared-stratified-128k",
            ),
        ]:
            if local.is_dir():
                results.append(
                    publish_parquet_dataset(
                        api,
                        repo_id=f"{args.namespace}/{repo_suffix}",
                        source_dir=local,
                        title=repo_suffix,
                        role=role,
                    )
                )

    if args.include_ortho_ns_mixed:
        for local, repo_suffix, role in [
            (
                ROOT / "data" / "clm" / "ortho_ns_mixed_v1",
                "LFM2.5-Wolof-CPT-ortho_ns_mixed_v1",
                "cpt-ortho-ns-mixed",
            ),
            (
                ROOT / "data" / "sft" / "ortho_ns_mixed_raw_v1",
                "LFM2.5-Wolof-SFT-ortho_ns_mixed_raw_v1",
                "sft-ortho-ns-mixed-raw",
            ),
            (
                ROOT / "data" / "sft" / "ortho_ns_mixed_stratified_v1",
                "LFM2.5-Wolof-SFT-ortho_ns_mixed_stratified_v1",
                "sft-ortho-ns-mixed-stratified",
            ),
            (
                ROOT / "data" / "sft" / "prepared_ortho_ns_mixed_v1",
                "LFM2.5-Wolof-SFT-prepared_ortho_ns_mixed_v1-65k",
                "sft-prepared-ortho-ns-mixed-65k",
            ),
            (
                ROOT / "data" / "sft" / "prepared_ortho_ns_mixed_v1_128k",
                "LFM2.5-Wolof-SFT-prepared_ortho_ns_mixed_v1-128k",
                "sft-prepared-ortho-ns-mixed-128k",
            ),
        ]:
            if local.is_dir() and any(local.glob("*.parquet")):
                results.append(
                    publish_parquet_dataset(
                        api,
                        repo_id=f"{args.namespace}/{repo_suffix}",
                        source_dir=local,
                        title=repo_suffix,
                        role=role,
                        extra_meta={
                            "filter_policy": "ortho_label in {non_standard, mixed}",
                            "label_method": "procedural_heuristic_v1",
                        },
                    )
                )

    report = {"updated_at": utc_now(), "results": results}
    out = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(out, report)
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
