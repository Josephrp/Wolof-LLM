#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.26.0",
#   "pyarrow",
# ]
# ///
"""HF Jobs entrypoint: procedural Wolof orthography split + Hub upload.

Submit via:
  uv run python scripts/data/25_submit_ortho_job.py
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

LABELS = ("standard", "non_standard", "mixed", "not_wolof")

# Letters that strongly mark school / CLAD Latin Wolof (not generic French).
_WOLOF_STRONG_CHARS = set("ëñŋ")
# Accents shared with French/Portuguese — only count with Wolof lexical cues.
_WOLOF_WEAK_CHARS = set("àâéèêîïóôùûć")
_STANDARD_CHARS = _WOLOF_STRONG_CHARS | _WOLOF_WEAK_CHARS

# High-precision informal substitutions observed in Wolof SMS / social.
_NONSTD_PATTERNS = (
    re.compile(r"(?<![a-z])gn[aeiouyàéèë]", re.I),  # ñ → gn
    re.compile(r"(?<![a-z])kh[aeiouy]", re.I),  # x → kh
    re.compile(r"\b[a-z]{14,}\b", re.I),  # long fused tokens
    re.compile(r"\b(jaa?pp|japp|nopp|xam|danga|naka)\b", re.I),  # bare common lemmas
)

# Very common French function words (not_wolof / FR-heavy cue).
_FR_STOP = {
    "le",
    "la",
    "les",
    "des",
    "une",
    "un",
    "et",
    "est",
    "dans",
    "pour",
    "avec",
    "sur",
    "que",
    "qui",
    "pas",
    "plus",
    "cette",
    "ces",
    "aux",
    "du",
    "au",
    "par",
    "sont",
    "nous",
    "vous",
    "ils",
    "elle",
    "mais",
    "comme",
    "tout",
    "aussi",
}

# Distinctive Wolof tokens (avoid FR-overlapping function words like "la").
_WO_STOP = {
    "ci",
    "ak",
    "bu",
    "bi",
    "yi",
    "ñu",
    "nu",
    "mu",
    "nga",
    "ngay",
    "danga",
    "naka",
    "lan",
    "koom",
    "woon",
    "nekk",
    "def",
    "xam",
    "wax",
    "yaw",
    "moom",
    "leen",
    "baax",
    "jëf",
    "jef",
    "waaw",
    "déedéet",
    "deedeet",
    "mbaa",
    "ndax",
    "ndaxte",
    "japp",
    "jàpp",
    "llakk",
    "réew",
    "reew",
}

_CODE_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bfunction\s*\("),
    re.compile(r"[{};<>]=|</?\w+>"),
    re.compile(r"https?://"),
)

_TOKEN_RE = re.compile(r"[A-Za-zÀ-öø-ÿŊŋÑñËëÓóÀàÉé]+", re.UNICODE)


@dataclass(frozen=True)
class OrthoDecision:
    label: str
    score_standard: float
    score_non_standard: float
    score_foreign: float
    reason: str


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def strip_diacritics(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def classify_orthography(text: str) -> OrthoDecision:
    """Return a deterministic orthography label for one string."""
    t = normalize_text(text)
    if not t or len(t) < 2:
        return OrthoDecision("not_wolof", 0.0, 0.0, 1.0, "empty_or_tiny")

    # Non-Latin dominant (Arabic, etc.)
    letters = [ch for ch in t if ch.isalpha()]
    if letters:
        latin = sum(1 for ch in letters if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in _STANDARD_CHARS or ord(ch) < 0x0250)
        if latin / len(letters) < 0.55:
            return OrthoDecision("not_wolof", 0.0, 0.0, 1.0, "non_latin_script")

    if any(p.search(t) for p in _CODE_PATTERNS[:4]) and t.count("\n") >= 2:
        return OrthoDecision("not_wolof", 0.0, 0.0, 0.9, "code_like")

    toks = _tokens(t)
    if not toks:
        return OrthoDecision("not_wolof", 0.0, 0.0, 1.0, "no_alpha_tokens")

    n = len(toks)
    lowered = t.lower()
    strong_hits = sum(1 for ch in lowered if ch in _WOLOF_STRONG_CHARS)
    weak_hits = sum(1 for ch in lowered if ch in _WOLOF_WEAK_CHARS)
    std_token_frac = (
        sum(1 for tok in toks if any(ch in _WOLOF_STRONG_CHARS for ch in tok)) / n
    )
    nonstd_hits = sum(1 for p in _NONSTD_PATTERNS if p.search(t))
    alpha = [ch for ch in t if ch.isalpha()]
    ascii_alpha_frac = (
        sum(1 for ch in alpha if ch.isascii()) / len(alpha) if alpha else 1.0
    )

    wo_hits = sum(
        1
        for tok in toks
        if (tok in _WO_STOP or strip_diacritics(tok) in _WO_STOP)
        and tok not in _FR_STOP
    )
    fr_hits = sum(1 for tok in toks if tok in _FR_STOP)
    wo_frac = wo_hits / n
    fr_frac = fr_hits / n

    # Weak French-shared accents only boost standard when Wolof lexicon is present.
    weak_boost = min(weak_hits / 4.0, 1.0) * (1.0 if wo_frac >= 0.06 else 0.15)
    score_std = min(
        1.0,
        0.55 * min(strong_hits / 2.0, 1.0) + 0.30 * std_token_frac + 0.15 * weak_boost,
    )
    score_non = min(
        1.0,
        0.35 * min(nonstd_hits / 2.0, 1.0)
        + 0.35 * (1.0 if ascii_alpha_frac > 0.98 and wo_frac >= 0.08 else 0.0)
        + 0.30 * min(wo_frac * 2.0, 1.0) * (1.0 if strong_hits == 0 else 0.25),
    )
    score_foreign = min(
        1.0,
        fr_frac * 2.2
        + (0.35 if ascii_alpha_frac > 0.99 and wo_frac < 0.04 else 0.0)
        + (0.25 if fr_frac > wo_frac and strong_hits == 0 else 0.0),
    )

    # Foreign / not Wolof when French dominates and Wolof cues are weak.
    if score_foreign >= 0.45 and wo_frac < 0.08 and strong_hits == 0:
        return OrthoDecision(
            "not_wolof", score_std, score_non, score_foreign, "french_heavy"
        )

    # Strong standard with little informal noise
    if score_std >= 0.35 and nonstd_hits == 0 and wo_frac >= 0.04:
        return OrthoDecision(
            "standard", score_std, score_non, score_foreign, "diacritic_standard"
        )

    # Both systems
    if score_std >= 0.25 and nonstd_hits >= 1:
        return OrthoDecision(
            "mixed", score_std, score_non, score_foreign, "both_systems"
        )

    # Informal Wolof without special letters
    if score_non >= 0.35 and score_std < 0.20:
        return OrthoDecision(
            "non_standard", score_std, score_non, score_foreign, "informal_cues"
        )

    # Ascii Wolof-ish → non_standard; ascii foreign → not_wolof
    if ascii_alpha_frac > 0.98 and score_std < 0.1:
        if wo_frac >= 0.08 or nonstd_hits >= 1:
            return OrthoDecision(
                "non_standard",
                score_std,
                max(score_non, 0.4),
                score_foreign,
                "ascii_wolof",
            )
        if fr_frac >= 0.10 or wo_frac < 0.03:
            return OrthoDecision(
                "not_wolof", score_std, score_non, score_foreign, "ascii_foreign"
            )

    # Weak standard diacritics only when Wolof lexicon supports it
    if score_std >= 0.20 and wo_frac >= 0.05:
        return OrthoDecision(
            "standard", score_std, score_non, score_foreign, "weak_standard"
        )

    if wo_frac >= 0.06:
        return OrthoDecision(
            "non_standard", score_std, max(score_non, 0.3), score_foreign, "wolof_tokens"
        )

    if fr_frac >= 0.08 and strong_hits == 0:
        return OrthoDecision(
            "not_wolof", score_std, score_non, score_foreign, "foreign_fallback"
        )

    return OrthoDecision("mixed", score_std, score_non, score_foreign, "ambiguous")


def classify_label(text: str) -> str:
    return classify_orthography(text).label


def extract_sft_text(messages) -> str:
    """Flatten ChatML messages to a single string for labeling."""
    if messages is None:
        return ""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


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
