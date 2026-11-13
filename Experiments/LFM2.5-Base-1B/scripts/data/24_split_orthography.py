#!/usr/bin/env python3
"""Procedurally split Wolof Hub datasets by orthography and publish partitions.

Sources (defaults):
  - galsenai/cleaned_data_for_clm
  - galsenai/wolof_centalized_corpus
  - galsenai/wolof_sft

Outputs (under --namespace, default Tonic):
  - {ns}/wolof_ortho_partitions   configs: clm_*, raw_*, sft_* × labels
  - {ns}/cleaned_data_for_clm_ortho_labeled  (train/test + ortho_* columns)
  - {ns}/wolof_sft_ortho_labeled
  - {ns}/wolof_centalized_corpus_ortho_labeled

Example:
  uv run python scripts/data/24_split_orthography.py --smoke
  uv run python scripts/data/24_split_orthography.py --push --private
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.orthography import (  # noqa: E402
    LABELS,
    classify_orthography,
    extract_sft_text,
)

DEFAULTS = {
    "clm": "galsenai/cleaned_data_for_clm",
    "raw": "galsenai/wolof_centalized_corpus",
    "sft": "galsenai/wolof_sft",
}


def _token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def _kw(token: str | None) -> dict[str, Any]:
    return {"token": token} if token else {}


def _label_text_column(ds: Dataset, text_col: str) -> Dataset:
    def _map(batch):
        labels, scores_s, scores_n, scores_f, reasons = [], [], [], [], []
        for text in batch[text_col]:
            d = classify_orthography(str(text) if text is not None else "")
            labels.append(d.label)
            scores_s.append(d.score_standard)
            scores_n.append(d.score_non_standard)
            scores_f.append(d.score_foreign)
            reasons.append(d.reason)
        return {
            "ortho_label": labels,
            "ortho_score_standard": scores_s,
            "ortho_score_non_standard": scores_n,
            "ortho_score_foreign": scores_f,
            "ortho_reason": reasons,
        }

    return ds.map(_map, batched=True, batch_size=512, desc=f"ortho:{text_col}")


def _label_sft(ds: Dataset) -> Dataset:
    def _map(batch):
        labels, scores_s, scores_n, scores_f, reasons, texts = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for messages in batch["messages"]:
            text = extract_sft_text(messages)
            texts.append(text)
            d = classify_orthography(text)
            labels.append(d.label)
            scores_s.append(d.score_standard)
            scores_n.append(d.score_non_standard)
            scores_f.append(d.score_foreign)
            reasons.append(d.reason)
        return {
            "ortho_text": texts,
            "ortho_label": labels,
            "ortho_score_standard": scores_s,
            "ortho_score_non_standard": scores_n,
            "ortho_score_foreign": scores_f,
            "ortho_reason": reasons,
        }

    return ds.map(_map, batched=True, batch_size=256, desc="ortho:sft")


def _histogram(ds: Dataset) -> dict[str, int]:
    c = Counter(ds["ortho_label"])
    return {lab: int(c.get(lab, 0)) for lab in LABELS}


def _filter_label(ds: Dataset, label: str) -> Dataset:
    return ds.filter(lambda ex: ex["ortho_label"] == label, desc=f"filter:{label}")


def _maybe_slice(ds: Dataset, max_rows: int | None) -> Dataset:
    if max_rows is None or max_rows <= 0 or len(ds) <= max_rows:
        return ds
    return ds.select(range(max_rows))


def process_clm(
    dataset_id: str, token: str | None, max_rows: int | None
) -> tuple[DatasetDict, dict[str, Any]]:
    raw = load_dataset(dataset_id, **_kw(token))
    out = DatasetDict()
    stats: dict[str, Any] = {"dataset_id": dataset_id, "splits": {}}
    for split in raw:
        part = _maybe_slice(raw[split], max_rows)
        labeled = _label_text_column(part, "text")
        out[split] = labeled
        stats["splits"][split] = {
            "n": len(labeled),
            "labels": _histogram(labeled),
        }
    return out, stats


def process_raw(
    dataset_id: str, token: str | None, max_rows: int | None
) -> tuple[DatasetDict, dict[str, Any]]:
    # Single-split corpus
    try:
        ds = load_dataset(dataset_id, split="train", **_kw(token))
        split_name = "train"
    except Exception:
        full = load_dataset(dataset_id, **_kw(token))
        split_name = list(full.keys())[0]
        ds = full[split_name]
    ds = _maybe_slice(ds, max_rows)
    labeled = _label_text_column(ds, "text")
    out = DatasetDict({split_name: labeled})
    stats = {
        "dataset_id": dataset_id,
        "splits": {split_name: {"n": len(labeled), "labels": _histogram(labeled)}},
    }
    return out, stats


def process_sft(
    dataset_id: str, token: str | None, max_rows: int | None
) -> tuple[DatasetDict, dict[str, Any]]:
    raw = load_dataset(dataset_id, **_kw(token))
    out = DatasetDict()
    stats: dict[str, Any] = {"dataset_id": dataset_id, "splits": {}}
    for split in raw:
        part = _maybe_slice(raw[split], max_rows)
        labeled = _label_sft(part)
        out[split] = labeled
        stats["splits"][split] = {
            "n": len(labeled),
            "labels": _histogram(labeled),
        }
    return out, stats


def build_partitions(
    prefix: str, labeled: DatasetDict
) -> dict[str, Dataset]:
    """configs like clm_standard / clm_non_standard (train concat if multi-split)."""
    configs: dict[str, Dataset] = {}
    for label in LABELS:
        pieces = []
        for split, ds in labeled.items():
            filt = _filter_label(ds, label)
            if len(filt) == 0:
                continue
            # keep original columns + ortho_* ; add source_split
            filt = filt.add_column("source_split", [split] * len(filt))
            pieces.append(filt)
        if not pieces:
            # empty placeholder so config exists
            configs[f"{prefix}_{label}"] = Dataset.from_dict(
                {"text": [], "ortho_label": [], "source_split": []}
            )
            continue
        configs[f"{prefix}_{label}"] = concatenate_datasets(pieces)
    return configs


def push_dataset(
    ds: Dataset | DatasetDict,
    repo_id: str,
    token: str,
    private: bool,
    config_name: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if isinstance(ds, DatasetDict):
        ds.push_to_hub(repo_id, private=private, token=token)
    else:
        ds.push_to_hub(
            repo_id,
            config_name=config_name,
            private=private,
            token=token,
        )


def write_card(repo_id: str, token: str, body: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=body.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card (orthography partitions)",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default=os.environ.get("ORTHO_NAMESPACE", "Tonic"))
    p.add_argument("--clm", default=DEFAULTS["clm"])
    p.add_argument("--raw", default=DEFAULTS["raw"])
    p.add_argument("--sft", default=DEFAULTS["sft"])
    p.add_argument("--skip-clm", action="store_true")
    p.add_argument("--skip-raw", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--max-rows", type=int, default=None, help="Per-split cap")
    p.add_argument("--smoke", action="store_true", help="Alias for --max-rows 200")
    p.add_argument("--push", action="store_true")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", action="store_true", help="Upload public datasets")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "ortho_split",
        help="Local parquet/json stats dump",
    )
    args = p.parse_args()
    if args.smoke and args.max_rows is None:
        args.max_rows = 200
    private = not args.public
    token = _token()
    if args.push and not token:
        print("ERROR: HF_TOKEN required for --push", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "namespace": args.namespace,
        "max_rows": args.max_rows,
        "method": "procedural_heuristic_v1",
        "labels": list(LABELS),
        "sources": {},
        "outputs": [],
    }

    partition_configs: dict[str, Dataset] = {}

    if not args.skip_clm:
        print(f"[clm] labeling {args.clm} …")
        labeled, stats = process_clm(args.clm, token, args.max_rows)
        report["sources"]["clm"] = stats
        labeled.save_to_disk(str(args.out_dir / "clm_labeled"))
        partition_configs.update(build_partitions("clm", labeled))
        if args.push:
            repo = f"{args.namespace}/cleaned_data_for_clm_ortho_labeled"
            print(f"[push] {repo}")
            push_dataset(labeled, repo, token, private)
            report["outputs"].append(repo)

    if not args.skip_raw:
        print(f"[raw] labeling {args.raw} …")
        labeled, stats = process_raw(args.raw, token, args.max_rows)
        report["sources"]["raw"] = stats
        labeled.save_to_disk(str(args.out_dir / "raw_labeled"))
        partition_configs.update(build_partitions("raw", labeled))
        if args.push:
            repo = f"{args.namespace}/wolof_centalized_corpus_ortho_labeled"
            print(f"[push] {repo}")
            push_dataset(labeled, repo, token, private)
            report["outputs"].append(repo)

    if not args.skip_sft:
        print(f"[sft] labeling {args.sft} …")
        labeled, stats = process_sft(args.sft, token, args.max_rows)
        report["sources"]["sft"] = stats
        labeled.save_to_disk(str(args.out_dir / "sft_labeled"))
        partition_configs.update(build_partitions("sft", labeled))
        if args.push:
            repo = f"{args.namespace}/wolof_sft_ortho_labeled"
            print(f"[push] {repo}")
            push_dataset(labeled, repo, token, private)
            report["outputs"].append(repo)

    # Umbrella partitions repo
    if args.push and partition_configs:
        umbrella = f"{args.namespace}/wolof_ortho_partitions"
        print(f"[push] {umbrella} ({len(partition_configs)} configs)")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(umbrella, repo_type="dataset", private=private, exist_ok=True)
        for cfg_name, ds in sorted(partition_configs.items()):
            print(f"  · config {cfg_name}: {len(ds):,} rows")
            ds.push_to_hub(
                umbrella,
                config_name=cfg_name,
                private=private,
                token=token,
            )
        card = _card_md(report, umbrella)
        write_card(umbrella, token, card)
        report["outputs"].append(umbrella)

    stats_path = args.out_dir / "ortho_split_report.json"
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {stats_path}")
    return 0


def _card_md(report: dict[str, Any], umbrella: str) -> str:
    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "- wo",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- wolof",
        "- orthography",
        "- procedural-split",
        "pretty_name: Wolof Orthography Partitions",
        "---",
        "",
        "# Wolof orthography partitions (procedural)",
        "",
        "Automatic split of GalsenAI Wolof corpora into "
        "`standard` / `non_standard` / `mixed` / `not_wolof` using deterministic "
        "heuristics (not volunteer gold).",
        "",
        f"- Method: `{report.get('method')}`",
        f"- Created: `{report.get('created_at')}`",
        "",
        "## Sources",
        "",
    ]
    for key, stats in (report.get("sources") or {}).items():
        lines.append(f"### {key}: `{stats.get('dataset_id')}`")
        for split, info in (stats.get("splits") or {}).items():
            lines.append(f"- **{split}** n={info['n']:,} → `{info['labels']}`")
        lines.append("")
    lines += [
        "## Configs",
        "",
        "Each config is `{source}_{label}` where source ∈ {clm, raw, sft}.",
        "",
        f"Load example:",
        "",
        "```python",
        "from datasets import load_dataset",
        f"ds = load_dataset('{umbrella}', 'clm_standard', split='train')",
        "```",
        "",
        "For human-validated gold, use the crowdsource Space "
        "`wolof-orthography-annotate` consensus dataset instead.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
