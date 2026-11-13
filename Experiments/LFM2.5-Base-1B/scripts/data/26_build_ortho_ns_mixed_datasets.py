#!/usr/bin/env python3
"""Filter CPT/SFT ortho labels to non_standard+mixed, clean/stratify SFT, publish Hub v3 sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write

KEEP_LABELS = frozenset({"non_standard", "mixed"})
CPT_SOURCE = "Tonic/cleaned_data_for_clm_ortho_labeled"
SFT_SOURCE = "Tonic/wolof_sft_ortho_labeled"
LABEL_METHOD = "procedural_heuristic_v1"
CODE_SOURCE_SUBSTR = "Code-170k"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def require_token() -> str:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    return token


def _label_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(series.astype(str)).items()}


def filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "ortho_label" not in df.columns:
        raise SystemExit("missing ortho_label column")
    return df[df["ortho_label"].astype(str).isin(KEEP_LABELS)].copy()


def messages_fingerprint(messages) -> str:
    if isinstance(messages, str):
        raw = messages
    else:
        raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clean_sft(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply cleaned_v1-style caps: drop Code-170k, long chats, short answers, dedupe."""
    n0 = len(df)
    dropped = {
        "code_170k": 0,
        "too_many_turns": 0,
        "short_assistant": 0,
        "dedupe": 0,
        "bad_messages": 0,
    }
    rows = []
    seen = set()
    for row in df.to_dict(orient="records"):
        source = str(row.get("source") or "")
        if CODE_SOURCE_SUBSTR in source:
            dropped["code_170k"] += 1
            continue
        messages = row.get("messages")
        try:
            if isinstance(messages, str):
                messages = json.loads(messages)
            elif hasattr(messages, "tolist"):
                messages = messages.tolist()
            if not isinstance(messages, list) or not messages:
                dropped["bad_messages"] += 1
                continue
            if len(messages) > 20:
                dropped["too_many_turns"] += 1
                continue
            assistants = [
                str(m.get("content") or m.get("value") or "")
                for m in messages
                if (m.get("role") in {"assistant", "gpt"} or m.get("from") == "gpt")
            ]
            if not assistants or min(len(a.strip()) for a in assistants) < 2:
                dropped["short_assistant"] += 1
                continue
            fp = messages_fingerprint(messages)
            if fp in seen:
                dropped["dedupe"] += 1
                continue
            seen.add(fp)
            row = dict(row)
            row["messages"] = messages
            rows.append(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            dropped["bad_messages"] += 1
            continue
    out = pd.DataFrame(rows)
    meta = {
        "n_in": n0,
        "n_out": len(out),
        "dropped": dropped,
        "rules": [
            "exclude source containing Code-170k",
            "drop conversations with >20 turns",
            "drop if any assistant reply <2 chars",
            "dedupe by sha256(messages JSON)",
        ],
    }
    return out, meta


def stratify_90_10(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic 90/10 by task (same spirit as wolof_sft_cleaned_v1_stratified)."""
    train_parts, test_parts = [], []
    for _, group in df.groupby("task", sort=False):
        if len(group) < 2:
            train_parts.append(group)
            continue
        n_test = max(1, round(len(group) * 0.10))
        n_test = min(n_test, len(group) - 1)
        test = group.sample(n=n_test, random_state=seed)
        train = group.drop(test.index)
        train_parts.append(train)
        test_parts.append(test)
    train = (
        pd.concat(train_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    test = (
        pd.concat(test_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed + 1)
        .reset_index(drop=True)
        if test_parts
        else df.iloc[0:0].copy()
    )
    return train, test


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


def ensure_private_dataset(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if info.private is not True:
        raise RuntimeError(f"refusing non-private dataset: {repo_id}")


def publish_parquet_dir(
    api: HfApi,
    *,
    repo_id: str,
    source_dir: Path,
    title: str,
    role: str,
    body_extra: str,
    split_map: dict[str, str] | None = None,
) -> str:
    ensure_private_dataset(api, repo_id)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data"
        data.mkdir(parents=True)
        copied = []
        for path in sorted(source_dir.glob("*.parquet")):
            shutil.copy2(path, data / path.name)
            copied.append(f"data/{path.name}")
        for path in sorted(source_dir.glob("*.json")):
            shutil.copy2(path, root / path.name)
            copied.append(path.name)
        configs = []
        mapping = split_map or {
            "train": "train.parquet",
            "validation": "val.parquet",
            "test": "test.parquet",
        }
        for split, fname in mapping.items():
            if (data / fname).is_file():
                configs.append((split, f"data/{fname}"))
        config_yaml = ""
        if configs:
            config_yaml = "configs:\n- config_name: default\n  data_files:\n"
            for split, path in configs:
                config_yaml += f"  - split: {split}\n    path: {path}\n"
        write_readme(
            root / "README.md",
            title=title,
            tags=["wolof", "lfm2.5", "ortho", "ns_mixed", role],
            body=f"""
# {title}

Filter policy: keep `ortho_label ∈ {{non_standard, mixed}}` from procedural labels
(`{LABEL_METHOD}`). Drop `standard` and `not_wolof`.

{body_extra}

{config_yaml}
## Files

```text
{chr(10).join(copied)}
```

Published: {utc_now()}
""",
        )
        commit = api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Publish {role} ortho_ns_mixed_v1",
        )
    return commit.oid


def build_cpt(token: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(CPT_SOURCE, token=token)
    metas = {}
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        before = _label_counts(df["ortho_label"])
        kept = filter_labels(df)
        after = _label_counts(kept["ortho_label"])
        # Training loader uses text column; keep ortho columns for audit.
        kept.to_parquet(out / f"{split}.parquet", index=False)
        # SmolLM3 local names for charger_splits_local / tokenizer fallback.
        if split == "train":
            kept.to_parquet(out / "wolof_bpe_train_v1.parquet", index=False)
        else:
            kept.to_parquet(out / "wolof_eval_clean_v1.parquet", index=False)
        metas[split] = {
            "n_before": int(len(df)),
            "n_after": int(len(kept)),
            "labels_before": before,
            "labels_after": after,
        }
    # Eval CSV for Evaluation/evaluate_clm.py primary BPB.
    eval_root = Path(
        os.environ.get(
            "EVALUATION_ROOT",
            str(ROOT.parents[1] / "Evaluation"),
        )
    )
    eval_data = eval_root / "data"
    eval_data.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_parquet(out / "test.parquet")
    csv_path = eval_data / "corpus_test_clm_ortho_ns_mixed.csv"
    test_df[["text"]].to_csv(csv_path, index=False)
    meta = {
        "schema_version": 1,
        "role": "cpt-ortho-ns-mixed",
        "source_dataset": CPT_SOURCE,
        "label_method": LABEL_METHOD,
        "keep_labels": sorted(KEEP_LABELS),
        "splits": metas,
        "primary_eval_csv": str(csv_path),
        "updated_at": utc_now(),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def build_sft_raw(token: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(SFT_SOURCE, token=token)
    metas = {}
    drop_by_task: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        before = _label_counts(df["ortho_label"])
        kept = filter_labels(df)
        after = _label_counts(kept["ortho_label"])
        # Log drop rates per task (full → kept).
        task_before = Counter(df["task"].astype(str))
        task_after = Counter(kept["task"].astype(str))
        drop_by_task[split] = {
            task: int(task_before[task] - task_after.get(task, 0))
            for task in sorted(task_before)
        }
        kept.to_parquet(out / f"{split}.parquet", index=False)
        metas[split] = {
            "n_before": int(len(df)),
            "n_after": int(len(kept)),
            "labels_before": before,
            "labels_after": after,
            "task_counts_after": dict(task_after),
        }
    meta = {
        "schema_version": 1,
        "role": "sft-ortho-ns-mixed-raw",
        "source_dataset": SFT_SOURCE,
        "label_method": LABEL_METHOD,
        "keep_labels": sorted(KEEP_LABELS),
        "splits": metas,
        "drop_by_task": drop_by_task,
        "updated_at": utc_now(),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def build_sft_stratified(raw_dir: Path, out: Path, seed: int = 42) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_parquet(raw_dir / "train.parquet")
    cleaned, clean_meta = clean_sft(train_raw)
    train, val = stratify_90_10(cleaned, seed=seed)
    # Hub stratified used test as val; keep both names for prepare_sft / materialize.
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    val.to_parquet(out / "test.parquet", index=False)
    meta = {
        "schema_version": 1,
        "role": "sft-ortho-ns-mixed-stratified",
        "source_raw": str(raw_dir),
        "seed": seed,
        "clean": clean_meta,
        "n_train": len(train),
        "n_val": len(val),
        "task_counts_train": _label_counts(train["task"]) if "task" in train else {},
        "task_counts_val": _label_counts(val["task"]) if "task" in val else {},
        "ortho_counts_train": _label_counts(train["ortho_label"]),
        "ortho_counts_val": _label_counts(val["ortho_label"]),
        "updated_at": utc_now(),
    }
    # Fix task counts (not labels).
    meta["task_counts_train"] = {
        str(k): int(v) for k, v in Counter(train["task"].astype(str)).items()
    }
    meta["task_counts_val"] = {
        str(k): int(v) for k, v in Counter(val["task"].astype(str)).items()
    }
    (out / "mixture_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def publish_all(api: HfApi, namespace: str, paths: dict[str, Path], metas: dict) -> dict:
    results = []
    results.append(
        {
            "repo_id": f"{namespace}/LFM2.5-Wolof-CPT-ortho_ns_mixed_v1",
            "revision": publish_parquet_dir(
                api,
                repo_id=f"{namespace}/LFM2.5-Wolof-CPT-ortho_ns_mixed_v1",
                source_dir=paths["cpt"],
                title="LFM2.5 Wolof CPT ortho_ns_mixed_v1",
                role="cpt",
                body_extra=f"""
Source: [`{CPT_SOURCE}`](https://huggingface.co/datasets/{CPT_SOURCE})

Keep sizes (train): non_standard+mixed ≈ {metas['cpt']['splits']['train']['n_after']:,}
(from {metas['cpt']['splits']['train']['n_before']:,}).

Primary CLM BPB should use this dataset's `test` split.
""",
                split_map={"train": "train.parquet", "test": "test.parquet"},
            ),
            "kind": "cpt",
        }
    )
    results.append(
        {
            "repo_id": f"{namespace}/LFM2.5-Wolof-SFT-ortho_ns_mixed_raw_v1",
            "revision": publish_parquet_dir(
                api,
                repo_id=f"{namespace}/LFM2.5-Wolof-SFT-ortho_ns_mixed_raw_v1",
                source_dir=paths["sft_raw"],
                title="LFM2.5 Wolof SFT ortho_ns_mixed_raw_v1",
                role="sft-raw",
                body_extra=f"""
Source: [`{SFT_SOURCE}`](https://huggingface.co/datasets/{SFT_SOURCE})

Raw filter only (no clean/stratify). Train keep ≈ {metas['sft_raw']['splits']['train']['n_after']:,}.
""",
                split_map={"train": "train.parquet", "test": "test.parquet"},
            ),
            "kind": "sft-raw",
        }
    )
    results.append(
        {
            "repo_id": f"{namespace}/LFM2.5-Wolof-SFT-ortho_ns_mixed_stratified_v1",
            "revision": publish_parquet_dir(
                api,
                repo_id=f"{namespace}/LFM2.5-Wolof-SFT-ortho_ns_mixed_stratified_v1",
                source_dir=paths["sft_strat"],
                title="LFM2.5 Wolof SFT ortho_ns_mixed_stratified_v1",
                role="sft-stratified",
                body_extra=f"""
Cleaned (Code-170k drop, turn/answer caps, dedupe) then 90/10 stratified by `task`.

Train/val: {metas['sft_strat']['n_train']:,} / {metas['sft_strat']['n_val']:,}.
""",
                split_map={
                    "train": "train.parquet",
                    "validation": "val.parquet",
                    "test": "test.parquet",
                },
            ),
            "kind": "sft-stratified",
        }
    )
    for key, suffix, role in [
        ("prepared_65k", "LFM2.5-Wolof-SFT-prepared_ortho_ns_mixed_v1-65k", "sft-prepared-65k"),
        ("prepared_128k", "LFM2.5-Wolof-SFT-prepared_ortho_ns_mixed_v1-128k", "sft-prepared-128k"),
    ]:
        local = paths.get(key)
        if local and local.is_dir() and any(local.glob("*.parquet")):
            results.append(
                {
                    "repo_id": f"{namespace}/{suffix}",
                    "revision": publish_parquet_dir(
                        api,
                        repo_id=f"{namespace}/{suffix}",
                        source_dir=local,
                        title=suffix,
                        role=role,
                        body_extra="Prepared ChatML rows for Instruct-v3 SFT.",
                        split_map={
                            "train": "train.parquet",
                            "validation": "val.parquet",
                        },
                    ),
                    "kind": role,
                }
            )
    return {"updated_at": utc_now(), "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Also run 10_prepare_sft for 65k/128k")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    token = require_token()
    paths = {
        "cpt": ROOT / "data" / "clm" / "ortho_ns_mixed_v1",
        "sft_raw": ROOT / "data" / "sft" / "ortho_ns_mixed_raw_v1",
        "sft_strat": ROOT / "data" / "sft" / "ortho_ns_mixed_stratified_v1",
        "prepared_65k": ROOT / "data" / "sft" / "prepared_ortho_ns_mixed_v1",
        "prepared_128k": ROOT / "data" / "sft" / "prepared_ortho_ns_mixed_v1_128k",
    }

    print("Building CPT ortho_ns_mixed …", flush=True)
    cpt_meta = build_cpt(token, paths["cpt"])
    print(json.dumps(cpt_meta["splits"], indent=2), flush=True)

    print("Building SFT raw ortho_ns_mixed …", flush=True)
    sft_raw_meta = build_sft_raw(token, paths["sft_raw"])
    print(json.dumps(sft_raw_meta["splits"], indent=2)[:2000], flush=True)

    print("Cleaning + stratifying SFT …", flush=True)
    sft_strat_meta = build_sft_stratified(paths["sft_raw"], paths["sft_strat"], seed=args.seed)
    print(
        json.dumps(
            {
                "n_train": sft_strat_meta["n_train"],
                "n_val": sft_strat_meta["n_val"],
                "clean": sft_strat_meta["clean"],
            },
            indent=2,
        ),
        flush=True,
    )

    if args.prepare:
        py = sys.executable
        import subprocess

        cmd65 = [
            py,
            "scripts/sft/10_prepare_sft.py",
            "--input",
            str(paths["sft_strat"].relative_to(ROOT)),
            "--output",
            str(paths["prepared_65k"].relative_to(ROOT)),
            "--tokenizer-model",
            "models/lfm25-wolof-ext4000-ortho-ns-mixed-init",
        ]
        tok65 = ROOT / "models" / "lfm25-wolof-ext4000-ortho-ns-mixed-init"
        if not tok65.is_dir():
            # Fall back to existing ext4000 init for prepare-before-rebuild smoke.
            cmd65[-1] = "models/lfm25-wolof-ext4000-init"
        if args.smoke:
            cmd65.append("--smoke")
        print("+", " ".join(cmd65), flush=True)
        subprocess.run(cmd65, cwd=ROOT, check=True)

        cmd128 = [
            py,
            "scripts/sft/10_prepare_sft.py",
            "--input",
            str(paths["sft_strat"].relative_to(ROOT)),
            "--output",
            str(paths["prepared_128k"].relative_to(ROOT)),
            "--template-model",
            "LiquidAI/LFM2.5-2.6B",
            "--tokenizer-model",
            "models/lfm25-2.6b-wolof-ext4000-ortho-ns-mixed-init",
        ]
        tok128 = ROOT / "models" / "lfm25-2.6b-wolof-ext4000-ortho-ns-mixed-init"
        if not tok128.is_dir():
            cmd128[-1] = "models/lfm25-2.6b-wolof-ext4000-init"
        if args.smoke:
            cmd128.append("--smoke")
        print("+", " ".join(cmd128), flush=True)
        subprocess.run(cmd128, cwd=ROOT, check=True)

    metas = {"cpt": cpt_meta, "sft_raw": sft_raw_meta, "sft_strat": sft_strat_meta}
    report = {"updated_at": utc_now(), "local": {k: str(v) for k, v in paths.items()}, "metas": metas}
    if not args.skip_publish:
        api = HfApi(token=token)
        report["hub"] = publish_all(api, args.namespace, paths, metas)
    out = ROOT / "results" / "full_pipeline" / "ortho_ns_mixed_datasets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
