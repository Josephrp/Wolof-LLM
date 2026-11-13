# Job main body — concatenated after common/orthography.py by _build_ortho_job.py

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

DEFAULTS = {
    "clm": "galsenai/cleaned_data_for_clm",
    "raw": "galsenai/wolof_centalized_corpus",
    "sft": "galsenai/wolof_sft",
}


def _token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN is required")
    return tok


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _max_rows() -> int | None:
    raw = os.environ.get("ORTHO_MAX_ROWS", "").strip()
    return int(raw) if raw else None


def _label_text(ds: Dataset, col: str) -> Dataset:
    def _map(batch):
        labels, ss, sn, sf, reasons = [], [], [], [], []
        for text in batch[col]:
            d = classify_orthography(str(text) if text is not None else "")
            labels.append(d.label)
            ss.append(d.score_standard)
            sn.append(d.score_non_standard)
            sf.append(d.score_foreign)
            reasons.append(d.reason)
        return {
            "ortho_label": labels,
            "ortho_score_standard": ss,
            "ortho_score_non_standard": sn,
            "ortho_score_foreign": sf,
            "ortho_reason": reasons,
        }

    return ds.map(_map, batched=True, batch_size=512)


def _label_sft(ds: Dataset) -> Dataset:
    def _map(batch):
        labels, ss, sn, sf, reasons, texts = [], [], [], [], [], []
        for messages in batch["messages"]:
            text = extract_sft_text(messages)
            texts.append(text)
            d = classify_orthography(text)
            labels.append(d.label)
            ss.append(d.score_standard)
            sn.append(d.score_non_standard)
            sf.append(d.score_foreign)
            reasons.append(d.reason)
        return {
            "ortho_text": texts,
            "ortho_label": labels,
            "ortho_score_standard": ss,
            "ortho_score_non_standard": sn,
            "ortho_score_foreign": sf,
            "ortho_reason": reasons,
        }

    return ds.map(_map, batched=True, batch_size=256)


def _hist(ds: Dataset) -> dict[str, int]:
    c = Counter(ds["ortho_label"])
    return {lab: int(c.get(lab, 0)) for lab in LABELS}


def _slice(ds: Dataset, n: int | None) -> Dataset:
    if n is None or n <= 0 or len(ds) <= n:
        return ds
    return ds.select(range(n))


def build_partitions(prefix: str, labeled: DatasetDict) -> dict[str, Dataset]:
    configs: dict[str, Dataset] = {}
    for label in LABELS:
        pieces = []
        for split, ds in labeled.items():
            filt = ds.filter(lambda ex, lab=label: ex["ortho_label"] == lab)
            if len(filt) == 0:
                continue
            filt = filt.add_column("source_split", [split] * len(filt))
            pieces.append(filt)
        if pieces:
            configs[f"{prefix}_{label}"] = concatenate_datasets(pieces)
        else:
            configs[f"{prefix}_{label}"] = Dataset.from_dict(
                {"text": [], "ortho_label": [], "source_split": []}
            )
    return configs


def push_dd(ds: DatasetDict, repo: str, token: str, private: bool) -> None:
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    ds.push_to_hub(repo, private=private, token=token)


def main() -> None:
    token = _token()
    ns = os.environ.get("ORTHO_NAMESPACE", "Tonic").strip() or "Tonic"
    private = not _env_flag("ORTHO_PUBLIC")
    max_rows = _max_rows()
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "namespace": ns,
        "max_rows": max_rows,
        "method": "procedural_heuristic_v1",
        "sources": {},
        "outputs": [],
    }
    partitions: dict[str, Dataset] = {}
    kw = {"token": token}

    if not _env_flag("ORTHO_SKIP_CLM"):
        print("[clm] loading…")
        raw = load_dataset(DEFAULTS["clm"], **kw)
        out = DatasetDict()
        stats: dict[str, Any] = {"dataset_id": DEFAULTS["clm"], "splits": {}}
        for split in raw:
            part = _label_text(_slice(raw[split], max_rows), "text")
            out[split] = part
            stats["splits"][split] = {"n": len(part), "labels": _hist(part)}
            print(f"  {split}: {stats['splits'][split]}")
        report["sources"]["clm"] = stats
        repo = f"{ns}/cleaned_data_for_clm_ortho_labeled"
        print(f"[push] {repo}")
        push_dd(out, repo, token, private)
        report["outputs"].append(repo)
        partitions.update(build_partitions("clm", out))

    if not _env_flag("ORTHO_SKIP_RAW"):
        print("[raw] loading…")
        try:
            ds = load_dataset(DEFAULTS["raw"], split="train", **kw)
            split_name = "train"
        except Exception:
            full = load_dataset(DEFAULTS["raw"], **kw)
            split_name = list(full.keys())[0]
            ds = full[split_name]
        labeled = DatasetDict({split_name: _label_text(_slice(ds, max_rows), "text")})
        stats = {
            "dataset_id": DEFAULTS["raw"],
            "splits": {
                split_name: {
                    "n": len(labeled[split_name]),
                    "labels": _hist(labeled[split_name]),
                }
            },
        }
        print(f"  {split_name}: {stats['splits'][split_name]}")
        report["sources"]["raw"] = stats
        repo = f"{ns}/wolof_centalized_corpus_ortho_labeled"
        print(f"[push] {repo}")
        push_dd(labeled, repo, token, private)
        report["outputs"].append(repo)
        partitions.update(build_partitions("raw", labeled))

    if not _env_flag("ORTHO_SKIP_SFT"):
        print("[sft] loading…")
        raw = load_dataset(DEFAULTS["sft"], **kw)
        out = DatasetDict()
        stats = {"dataset_id": DEFAULTS["sft"], "splits": {}}
        for split in raw:
            part = _label_sft(_slice(raw[split], max_rows))
            out[split] = part
            stats["splits"][split] = {"n": len(part), "labels": _hist(part)}
            print(f"  {split}: {stats['splits'][split]}")
        report["sources"]["sft"] = stats
        repo = f"{ns}/wolof_sft_ortho_labeled"
        print(f"[push] {repo}")
        push_dd(out, repo, token, private)
        report["outputs"].append(repo)
        partitions.update(build_partitions("sft", out))

    umbrella = f"{ns}/wolof_ortho_partitions"
    print(f"[push] {umbrella}")
    api = HfApi(token=token)
    api.create_repo(umbrella, repo_type="dataset", private=private, exist_ok=True)
    for name, ds in sorted(partitions.items()):
        print(f"  config {name}: {len(ds):,}")
        ds.push_to_hub(umbrella, config_name=name, private=private, token=token)
    report["outputs"].append(umbrella)

    card = (
        "---\nlicense: apache-2.0\nlanguage: [wo]\ntags: [wolof, orthography]\n"
        "pretty_name: Wolof Orthography Partitions\n---\n\n"
        "# Wolof orthography partitions (procedural)\n\n"
        f"```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```\n"
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=umbrella,
        repo_type="dataset",
        commit_message="Orthography partition card",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
