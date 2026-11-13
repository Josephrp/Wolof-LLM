#!/usr/bin/env python3
"""P0.1 — Audit Hub datasets for LFM2.5 Wolof CPT/SFT.

Writes results/data_audit.json, results/data_audit.md, and
results/audit/clm_samples.jsonl.

Example:
  uv run python scripts/data/20_audit_datasets.py --out results/data_audit.json
  uv run python scripts/data/20_audit_datasets.py --max-rows 500 --clm --sft
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]

EXPECTED = {
    "clm": {"train": 958_646, "test": 4_282, "tol": 0},
    "raw": {"rows": 1_018_520, "tol": 5_000},
    "sft": {"train": 990_320, "test": 2_250, "tol_pct": 0.01},
}

DEFAULT_CLM = "galsenai/cleaned_data_for_clm"
DEFAULT_RAW = "galsenai/wolof_centalized_corpus"
DEFAULT_SFT = "galsenai/wolof_sft"


def _token():
    env = os.environ.get("_HF_TOKEN_ENV_NAME", "HF_TOKEN")
    return os.environ.get(env) or os.environ.get("HF_TOKEN") or None


def _pctiles(values, ps=(50, 90, 99)):
    if not values:
        return {f"p{p}": None for p in ps}
    sorted_v = sorted(values)
    n = len(sorted_v)

    def _p(p):
        if n == 1:
            return sorted_v[0]
        idx = min(n - 1, max(0, round((p / 100) * (n - 1))))
        return sorted_v[idx]

    return {f"p{p}": _p(p) for p in ps}


def _assert_size(name, got, expected, tol=0, soft=False):
    ok = abs(got - expected) <= tol
    msg = f"{name}: got {got:,}, expected {expected:,} (±{tol})"
    if ok:
        return {"ok": True, "message": msg}
    if soft:
        return {"ok": False, "soft": True, "message": f"WARN {msg}"}
    return {"ok": False, "soft": False, "message": f"FAIL {msg}"}


def _safe_json_loads(value, default=None):
    if value is None:
        return default, True
    if isinstance(value, (dict, list)):
        return value, True
    if not isinstance(value, str):
        return default, False
    s = value.strip()
    if not s:
        return default, True
    try:
        return json.loads(s), True
    except json.JSONDecodeError:
        return default, False


def audit_clm(dataset_id, token, max_rows, out_dir: Path):
    report = {"dataset_id": dataset_id}
    kwargs = {}
    if token:
        kwargs["token"] = token
    ds = load_dataset(dataset_id, **kwargs)
    report["splits"] = {k: len(ds[k]) for k in ds}
    checks = []
    checks.append(
        _assert_size(
            "clm.train",
            len(ds["train"]),
            EXPECTED["clm"]["train"],
            tol=EXPECTED["clm"]["tol"] if max_rows is None else 10**12,
            soft=max_rows is not None,
        )
    )
    checks.append(
        _assert_size(
            "clm.test",
            len(ds["test"]),
            EXPECTED["clm"]["test"],
            tol=EXPECTED["clm"]["tol"] if max_rows is None else 10**12,
            soft=max_rows is not None,
        )
    )

    train = ds["train"]
    test = ds["test"]
    if max_rows is not None:
        train = train.select(range(min(max_rows, len(train))))
        test = test.select(range(min(max_rows, len(test))))

    report["columns"] = {
        "train": train.column_names,
        "test": test.column_names,
    }
    for split_name, split in (("train", train), ("test", test)):
        if "text" not in split.column_names:
            checks.append({
                "ok": False,
                "soft": False,
                "message": f"FAIL {split_name}: missing 'text'",
            })
            continue
        texts = split["text"]
        nulls = sum(1 for t in texts if t is None)
        checks.append({
            "ok": nulls == 0,
            "soft": False,
            "message": f"{split_name} null text rate: {nulls}/{len(texts)}",
        })

    # Length histogram on (possibly sampled) train+test combined for stats,
    # but sample rows from test preferentially for human skim.
    sample_texts = list(test["text"]) + list(train["text"])
    chars = [len(str(t)) for t in sample_texts if t is not None]
    toks = [len(str(t).split()) for t in sample_texts if t is not None]
    report["length"] = {
        "n": len(chars),
        "chars": {
            "mean": statistics.fmean(chars) if chars else None,
            **_pctiles(chars),
        },
        "whitespace_tokens": {
            "mean": statistics.fmean(toks) if toks else None,
            **_pctiles(toks),
        },
    }

    samples_path = out_dir / "audit" / "clm_samples.jsonl"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"i": i, "text": str(t)[:2000]}, ensure_ascii=False) + "\n" for i, t in enumerate(list(test["text"])[:20]))
    report["samples_path"] = str(samples_path.relative_to(ROOT))
    report["checks"] = checks
    report["test_texts_for_overlap"] = [str(t) for t in list(test["text"])[:5000]]
    return report


def audit_raw(dataset_id, token, max_rows):
    report = {"dataset_id": dataset_id}
    kwargs = {}
    if token:
        kwargs["token"] = token
    ds = load_dataset(dataset_id, **kwargs)
    # Prefer train / default first split
    split_name = "train" if "train" in ds else next(iter(ds))
    split = ds[split_name]
    report["split"] = split_name
    report["n_rows_full"] = len(split)
    checks = [
        _assert_size(
            "raw.rows",
            len(split),
            EXPECTED["raw"]["rows"],
            tol=EXPECTED["raw"]["tol"] if max_rows is None else 10**12,
            soft=max_rows is not None,
        )
    ]
    if max_rows is not None:
        split = split.select(range(min(max_rows, len(split))))
    report["columns"] = split.column_names
    for col in ("text", "sources"):
        if col not in split.column_names:
            checks.append({
                "ok": False,
                "soft": False,
                "message": f"FAIL missing column '{col}'",
            })

    source_counter = Counter()
    n_gt_2000 = n_gt_8000 = 0
    code_flag = 0
    for row in split:
        text = row.get("text")
        sources = row.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        for s in sources:
            source_counter[str(s)] += 1
            if "Code-170k-wolof" in str(s) or "code" in str(s).lower():
                code_flag += 1
        if text is not None:
            L = len(str(text))
            if L > 2000:
                n_gt_2000 += 1
            if L > 8000:
                n_gt_8000 += 1

    n = max(len(split), 1)
    report["source_freq_top20"] = source_counter.most_common(20)
    report["long_context"] = {
        "pct_len_gt_2000": 100.0 * n_gt_2000 / n,
        "pct_len_gt_8000": 100.0 * n_gt_8000 / n,
        "n_sampled": n,
    }
    report["code_source_rows_sampled"] = code_flag
    report["orthography_note"] = (
        "Mixed standard / non-standard Latin Wolof orthography; "
        "GlotLID ~97% Wolof on card sample — optional LID pass later."
    )
    report["checks"] = checks
    return report


def audit_sft(dataset_id, token, max_rows):
    report = {"dataset_id": dataset_id}
    kwargs = {}
    if token:
        kwargs["token"] = token
    ds = load_dataset(dataset_id, **kwargs)
    report["splits"] = {k: len(ds[k]) for k in ds}
    checks = []
    for split, expected in (("train", EXPECTED["sft"]["train"]),
                            ("test", EXPECTED["sft"]["test"])):
        if split not in ds:
            checks.append({
                "ok": False,
                "soft": False,
                "message": f"FAIL missing split {split}",
            })
            continue
        got = len(ds[split])
        tol = int(expected * EXPECTED["sft"]["tol_pct"]) if max_rows is None else 10**12
        checks.append(
            _assert_size(f"sft.{split}", got, expected, tol=tol, soft=max_rows is not None)
        )

    train = ds["train"]
    test = ds.get("test")
    if max_rows is not None:
        train = train.select(range(min(max_rows, len(train))))
        if test is not None:
            test = test.select(range(min(max_rows, len(test))))

    task_c = Counter()
    lang_c = Counter()
    multi_turn = 0
    tool_call_n = 0
    meta_ok = meta_fail = 0
    args_ok = args_fail = 0
    parse_budget = min(1000, len(train))

    assistant_samples = []
    for i, row in enumerate(train):
        task = row.get("task") or "unknown"
        task_c[str(task)] += 1
        lang_c[str(row.get("prompt_lang") or "unknown")] += 1
        messages = row.get("messages")
        if isinstance(messages, str):
            messages, ok = _safe_json_loads(messages)
            if not ok:
                messages = []
        if isinstance(messages, list):
            roles = [m.get("role") for m in messages if isinstance(m, dict)]
            if roles.count("user") + roles.count("assistant") > 2:
                multi_turn += 1
            for m in messages:
                if not isinstance(m, dict):
                    continue
                if m.get("tool_calls"):
                    tool_call_n += 1
                    for tc in m["tool_calls"]:
                        if i >= parse_budget:
                            break
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                        args = fn.get("arguments") if isinstance(fn, dict) else None
                        if args is None:
                            continue
                        _, ok = _safe_json_loads(args)
                        if ok:
                            args_ok += 1
                        else:
                            args_fail += 1
                if (
                    m.get("role") == "assistant"
                    and m.get("content")
                    and len(assistant_samples) < 2000
                ):
                    assistant_samples.append(str(m["content"])[:500])
        if i < parse_budget:
            _, ok = _safe_json_loads(row.get("meta"), default={})
            if ok:
                meta_ok += 1
            else:
                meta_fail += 1

    n_train = max(len(train), 1)
    translation_pct = 100.0 * task_c.get("translation", 0) / n_train
    if translation_pct > 50:
        checks.append({
            "ok": False,
            "soft": True,
            "message": (
                f"WARN translation={translation_pct:.1f}% > 50% "
                "(soft — reweight mixture required)"
            ),
        })
    else:
        checks.append({
            "ok": True,
            "message": f"translation={translation_pct:.1f}% <= 50%",
        })

    test_sources = Counter()
    if test is not None:
        for row in test:
            test_sources[str(row.get("source") or "unknown")] += 1

    report["task_histogram"] = dict(task_c.most_common())
    report["translation_pct"] = translation_pct
    report["prompt_lang_histogram"] = dict(lang_c.most_common())
    report["multi_turn_count"] = multi_turn
    report["tool_call_count"] = tool_call_n
    report["json_parse_sample"] = {
        "n": parse_budget,
        "meta_ok": meta_ok,
        "meta_fail": meta_fail,
        "tool_args_ok": args_ok,
        "tool_args_fail": args_fail,
    }
    report["test_sources"] = dict(test_sources.most_common())
    report["leakage_caveat"] = (
        "Hub test includes belebele / afrimmlu / afrimgsm / afrixnli held out of "
        "SFT train, but source sentences may still appear in the pretrain corpus. "
        "Do not claim zero leakage without an explicit CLM↔SFT overlap study."
    )
    report["checks"] = checks
    report["assistant_samples_for_overlap"] = assistant_samples
    return report


def overlap_sample(clm_report, sft_report, n_clm=2000, n_sft=2000):
    """Exact-overlap sample: SFT assistant text vs CLM test (rate)."""
    clm_set = set(
        (clm_report.get("test_texts_for_overlap") or [])[:n_clm]
    )
    sft_as = (sft_report.get("assistant_samples_for_overlap") or [])[:n_sft]
    if not clm_set or not sft_as:
        return {"n_sft": 0, "n_hits": 0, "rate": None, "note": "insufficient sample"}
    hits = sum(1 for t in sft_as if t in clm_set)
    return {
        "n_clm_test_sampled": len(clm_set),
        "n_sft_assistant_sampled": len(sft_as),
        "n_hits": hits,
        "rate": hits / max(len(sft_as), 1),
        "note": "Exact string match on truncated assistant content vs CLM test sample.",
    }


def _md_summary(audit: dict) -> str:
    lines = ["# Data audit summary", ""]
    for key in ("clm", "raw", "sft"):
        if key not in audit:
            continue
        block = audit[key]
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- dataset: `{block.get('dataset_id')}`")
        if "splits" in block:
            lines.append(f"- splits: `{block['splits']}`")
        if "n_rows_full" in block:
            lines.append(f"- rows: {block['n_rows_full']:,}")
        if "translation_pct" in block:
            lines.append(f"- translation %: {block['translation_pct']:.1f}")
        if "task_histogram" in block:
            lines.append(f"- tasks: `{block['task_histogram']}`")
        if "length" in block:
            lines.append(f"- length chars: `{block['length'].get('chars')}`")
        for c in block.get("checks", []):
            lines.append(f"- check: {c.get('message')}")
        if key == "sft" and block.get("leakage_caveat"):
            lines.append(f"- leakage: {block['leakage_caveat']}")
        lines.append("")
    if "overlap" in audit:
        lines.append("## SFT vs CLM overlap (sample)")
        lines.append("")
        lines.append(f"- `{audit['overlap']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Audit LFM2.5 Wolof Hub datasets")
    ap.add_argument("--clm", action="store_true", help="Audit cleaned CLM dataset")
    ap.add_argument("--sft", action="store_true", help="Audit wolof_sft")
    ap.add_argument("--raw", action="store_true", help="Audit centralized corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "data_audit.json")
    ap.add_argument("--hf-token-env", default="HF_TOKEN",
                    help="Env var name holding HF token (default HF_TOKEN)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Limit rows per split for sampling / dry-run")
    ap.add_argument("--clm-id", default=DEFAULT_CLM)
    ap.add_argument("--raw-id", default=DEFAULT_RAW)
    ap.add_argument("--sft-id", default=DEFAULT_SFT)
    args = ap.parse_args()

    os.environ["_HF_TOKEN_ENV_NAME"] = args.hf_token_env
    token = os.environ.get(args.hf_token_env) or None

    if args.clm or args.sft or args.raw:
        do_clm, do_sft, do_raw = args.clm, args.sft, args.raw
    else:
        do_clm = do_sft = do_raw = True

    out_json = args.out if args.out.is_absolute() else ROOT / args.out
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = out_json.with_suffix(".md")

    audit: dict = {"max_rows": args.max_rows, "token_env": args.hf_token_env}
    hard_fail = False

    if do_clm:
        print(f"Auditing CLM: {args.clm_id}")
        audit["clm"] = audit_clm(args.clm_id, token, args.max_rows, ROOT / "results")
        for c in audit["clm"]["checks"]:
            print(" ", c["message"])
            if not c["ok"] and not c.get("soft"):
                hard_fail = True

    if do_raw:
        print(f"Auditing raw: {args.raw_id}")
        audit["raw"] = audit_raw(args.raw_id, token, args.max_rows)
        for c in audit["raw"]["checks"]:
            print(" ", c["message"])
            if not c["ok"] and not c.get("soft"):
                hard_fail = True

    if do_sft:
        print(f"Auditing SFT: {args.sft_id}")
        audit["sft"] = audit_sft(args.sft_id, token, args.max_rows)
        for c in audit["sft"]["checks"]:
            print(" ", c["message"])
            if not c["ok"] and not c.get("soft"):
                hard_fail = True

    if "clm" in audit and "sft" in audit:
        audit["overlap"] = overlap_sample(audit["clm"], audit["sft"])
        print(" overlap:", audit["overlap"])

    # Drop heavy sample buffers from JSON dump
    for key in ("clm", "sft"):
        if key in audit:
            audit[key].pop("test_texts_for_overlap", None)
            audit[key].pop("assistant_samples_for_overlap", None)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(_md_summary(audit))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")

    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
