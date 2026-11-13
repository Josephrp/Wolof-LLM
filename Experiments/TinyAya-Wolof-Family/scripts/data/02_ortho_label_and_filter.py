#!/usr/bin/env python3
"""Ortho-label NTP+SFT into ns and/or std Wolof orthography variants.

Variants (see common/ortho_variants.py):
  ns  — keep non_standard+mixed; drop standard + not_wolof
  std — keep standard+mixed; drop non_standard + not_wolof

NTP: filter only Wolof rows; keep FR/EN unchanged for multilingual CPT.
SFT: filter all rows by message text; then clean + optional restratify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.ortho_variants import ORTHO_VARIANTS, get_variant, resolve_variants
from common.orthography import classify_orthography, extract_sft_text
from common.pipeline import atomic_json_write, utc_now

CODE_SOURCE_SUBSTR = "Code-170k"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ortho label + filter Wolof orthography variants")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--ortho",
        default="both",
        help="ns | std | both (default both)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override output root (single variant only). Default: data/<variant dirname>",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _label_text(text: str) -> dict:
    decision = classify_orthography(str(text or ""))
    return {
        "ortho_label": decision.label,
        "ortho_score_standard": decision.score_standard,
        "ortho_score_non_standard": decision.score_non_standard,
        "ortho_score_foreign": decision.score_foreign,
        "ortho_reason": decision.reason,
    }


def messages_fingerprint(messages) -> str:
    if isinstance(messages, str):
        raw = messages
    else:
        raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clean_sft(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    dropped = {
        "code_170k": 0,
        "too_many_turns": 0,
        "short_assistant": 0,
        "dedupe": 0,
        "bad_messages": 0,
    }
    rows = []
    seen: set[str] = set()
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
    meta = {"n_in": n0, "n_out": len(out), "dropped": dropped}
    return out, meta


def stratify_90_10(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def process_ntp(
    raw: Path,
    out: Path,
    *,
    keep_labels: frozenset[str],
    variant: str,
    smoke: bool,
) -> dict:
    train = pd.read_parquet(raw / "ntp_70_20_10" / "train.parquet")
    test = pd.read_parquet(raw / "ntp_70_20_10" / "test.parquet")
    prov = pd.read_parquet(raw / "ntp_70_20_10" / "provenance.parquet")
    if smoke:
        train = train.head(3000)
        test = test.head(800)
        prov = prov.head(max(len(train), len(test)) + 100)

    prov_train = prov[prov["split"].astype(str) == "train"].reset_index(drop=True)
    if len(prov_train) >= len(train):
        langs = prov_train.loc[: len(train) - 1, "lang"].astype(str).tolist()
    else:
        langs = ["wo"] * len(train)

    labeled_rows = []
    kept = 0
    dropped_other_ortho = 0
    dropped_not_wolof = 0
    kept_fr_en = 0
    for i, row in enumerate(train.to_dict(orient="records")):
        lang = langs[i] if i < len(langs) else "wo"
        text = str(row.get("text") or "")
        if lang in {"fr", "en"}:
            rec = dict(row)
            rec.update(
                {
                    "lang": lang,
                    "ortho_label": "pass_through",
                    "ortho_score_standard": 0.0,
                    "ortho_score_non_standard": 0.0,
                    "ortho_score_foreign": 0.0,
                    "ortho_reason": "multilingual_keep",
                    "ortho_variant": variant,
                }
            )
            labeled_rows.append(rec)
            kept_fr_en += 1
            kept += 1
            continue
        meta = _label_text(text)
        if meta["ortho_label"] == "not_wolof":
            dropped_not_wolof += 1
            continue
        if meta["ortho_label"] not in keep_labels:
            dropped_other_ortho += 1
            continue
        rec = dict(row)
        rec["lang"] = lang
        rec["ortho_variant"] = variant
        rec.update(meta)
        labeled_rows.append(rec)
        kept += 1

    train_out = pd.DataFrame(labeled_rows)
    test_rows = []
    prov_test = prov[prov["split"].astype(str) == "test"].reset_index(drop=True)
    test_langs = (
        prov_test["lang"].astype(str).tolist()
        if len(prov_test) >= len(test)
        else ["wo"] * len(test)
    )
    for i, row in enumerate(test.to_dict(orient="records")):
        lang = test_langs[i] if i < len(test_langs) else "wo"
        text = str(row.get("text") or "")
        meta = (
            {
                "ortho_label": "pass_through",
                "ortho_score_standard": 0.0,
                "ortho_score_non_standard": 0.0,
                "ortho_score_foreign": 0.0,
                "ortho_reason": "multilingual_keep",
            }
            if lang in {"fr", "en"}
            else _label_text(text)
        )
        rec = dict(row)
        rec["lang"] = lang
        rec["ortho_variant"] = variant
        rec.update(meta)
        test_rows.append(rec)
    test_out = pd.DataFrame(test_rows)

    cpt_dir = out / "cpt"
    cpt_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(cpt_dir / "train.parquet", index=False)
    test_out.to_parquet(cpt_dir / "test.parquet", index=False)
    train_out[["text"]].to_parquet(cpt_dir / "wolof_bpe_train_v1.parquet", index=False)
    test_out[["text"]].to_parquet(cpt_dir / "wolof_eval_clean_v1.parquet", index=False)

    # Primary eval slice: kept Wolof labels only (exclude FR/EN pass-through).
    eval_wo = test_out[test_out["ortho_label"].astype(str).isin(keep_labels)].copy()
    if len(eval_wo):
        eval_wo.to_parquet(cpt_dir / "test_ortho_kept.parquet", index=False)

    forbidden = {"not_wolof"} | (
        set(ORTHO_VARIANTS["ns"]["keep_labels"] | ORTHO_VARIANTS["std"]["keep_labels"])
        - set(keep_labels)
    )
    # "mixed" is in both keep sets — not forbidden. Only the opposite pure label is.
    opposite = {"standard", "non_standard"} - set(keep_labels)
    forbidden_remaining = int(train_out["ortho_label"].astype(str).isin(opposite | {"not_wolof"}).sum())

    stats = {
        "variant": variant,
        "keep_labels": sorted(keep_labels),
        "n_train_in": int(len(train)),
        "n_train_out": int(len(train_out)),
        "kept_fr_en": kept_fr_en,
        "dropped_other_ortho": dropped_other_ortho,
        "dropped_not_wolof": dropped_not_wolof,
        "label_counts_train": dict(Counter(train_out["ortho_label"].astype(str))),
        "forbidden_remaining": forbidden_remaining,
        "n_test_ortho_kept": int(len(eval_wo)),
    }
    del forbidden  # computed via opposite above
    atomic_json_write(cpt_dir / "filter_meta.json", stats)
    if forbidden_remaining != 0:
        raise SystemExit(
            f"G1 fail ({variant}): opposite-ortho/not_wolof rows remain in CPT train"
        )
    return stats


def process_sft(
    raw: Path,
    out: Path,
    *,
    keep_labels: frozenset[str],
    variant: str,
    seed: int,
    smoke: bool,
) -> dict:
    train = pd.read_parquet(raw / "sft_stratified_v1" / "train.parquet")
    test = pd.read_parquet(raw / "sft_stratified_v1" / "test.parquet")
    if smoke:
        train = train.head(2500)
        test = test.head(400)

    def label_frame(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in df.to_dict(orient="records"):
            text = extract_sft_text(row.get("messages"))
            meta = _label_text(text)
            rec = dict(row)
            rec["ortho_text"] = text
            rec["ortho_variant"] = variant
            rec.update(meta)
            rows.append(rec)
        return pd.DataFrame(rows)

    train_l = label_frame(train)
    test_l = label_frame(test)
    before = len(train_l)
    train_f = train_l[train_l["ortho_label"].astype(str).isin(keep_labels)].copy()
    test_f = test_l[test_l["ortho_label"].astype(str).isin(keep_labels)].copy()
    opposite = {"standard", "non_standard"} - set(keep_labels)
    dropped_other = int(train_l["ortho_label"].astype(str).isin(opposite).sum())
    train_c, clean_meta = clean_sft(train_f)
    if "task" in train_c.columns and len(train_c) >= 20:
        train_s, test_s = stratify_90_10(train_c, seed=seed)
        test_final = test_s if len(test_s) else test_f
        train_final = train_s
    else:
        train_final, test_final = train_c, test_f

    sft_dir = out / "sft_stratified"
    sft_dir.mkdir(parents=True, exist_ok=True)
    train_final.to_parquet(sft_dir / "train.parquet", index=False)
    test_final.to_parquet(sft_dir / "test.parquet", index=False)
    test_final.to_parquet(sft_dir / "val.parquet", index=False)

    forbidden_remaining = (
        int(train_final["ortho_label"].astype(str).isin(opposite | {"not_wolof"}).sum())
        if len(train_final)
        else 0
    )
    stats = {
        "variant": variant,
        "keep_labels": sorted(keep_labels),
        "n_train_in": before,
        "n_train_after_ortho": int(len(train_f)),
        "n_train_after_clean": int(len(train_final)),
        "n_test": int(len(test_final)),
        "dropped_other_ortho": dropped_other,
        "clean": clean_meta,
        "label_counts_train": dict(Counter(train_final["ortho_label"].astype(str)))
        if len(train_final)
        else {},
        "forbidden_remaining": forbidden_remaining,
    }
    atomic_json_write(sft_dir / "filter_meta.json", stats)
    if forbidden_remaining != 0:
        raise SystemExit(
            f"G1 fail ({variant}): opposite-ortho/not_wolof rows remain in SFT train"
        )
    return stats


def run_variant(
    *,
    raw: Path,
    variant_key: str,
    out_root: Path | None,
    seed: int,
    smoke: bool,
) -> dict:
    variant = get_variant(variant_key)
    out = out_root if out_root is not None else ROOT / "data" / variant["data_dirname"]
    out.mkdir(parents=True, exist_ok=True)
    keep = frozenset(variant["keep_labels"])
    cpt_stats = process_ntp(
        raw, out, keep_labels=keep, variant=variant_key, smoke=smoke
    )
    sft_stats = process_sft(
        raw, out, keep_labels=keep, variant=variant_key, seed=seed, smoke=smoke
    )
    summary = {
        "created_at": utc_now(),
        "variant": variant_key,
        "title": variant["title"],
        "keep_labels": sorted(keep),
        "anchor_token": variant["anchor_token"],
        "out_root": str(out),
        "cpt": cpt_stats,
        "sft": sft_stats,
        "smoke": smoke,
    }
    atomic_json_write(out / "ortho_filter_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    keys = resolve_variants(args.ortho)
    if args.out_root is not None and len(keys) != 1:
        raise SystemExit("--out-root requires a single --ortho (ns or std)")
    summaries = []
    for key in keys:
        summaries.append(
            run_variant(
                raw=args.raw_root,
                variant_key=key,
                out_root=args.out_root,
                seed=args.seed,
                smoke=args.smoke,
            )
        )
    if len(summaries) > 1:
        atomic_json_write(
            ROOT / "data" / "ortho_variants_summary.json",
            {"created_at": utc_now(), "variants": summaries, "smoke": args.smoke},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
