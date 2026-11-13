"""Lightweight orthography heuristics shown as hints (not ground truth)."""

from __future__ import annotations

import re
import unicodedata

# Strong CLAD / school Latin Wolof markers (avoid counting French accents alone).
_STANDARD_CHARS = set("ëñŋ")

# Informal / phone-style cues (weak signals only).
_NONSTD_PATTERNS = (
    re.compile(r"\bgn[aeiouàéè]", re.I),  # ñ → gn
    re.compile(r"\bkh[aeiou]", re.I),  # x → kh
    re.compile(r"[a-z]{12,}", re.I),  # long fused tokens
)


def normalize_for_id(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def has_standard_diacritics(text: str) -> bool:
    lowered = text.lower()
    return any(ch in _STANDARD_CHARS for ch in lowered)


def heuristic_hint(text: str) -> str:
    """Return a short volunteer-facing hint; never auto-labels."""
    t = normalize_for_id(text)
    if not t:
        return "Empty text."
    std = has_standard_diacritics(t)
    nonstd_hits = sum(1 for p in _NONSTD_PATTERNS if p.search(t))
    latin_alpha = sum(ch.isalpha() for ch in t)
    ascii_alpha = sum(ch.isascii() and ch.isalpha() for ch in t)

    bits: list[str] = []
    if std:
        bits.append("contains Wolof special letters (leans standard)")
    elif latin_alpha and ascii_alpha == latin_alpha:
        bits.append("no special letters (may be non_standard or French)")
    if nonstd_hits:
        bits.append(f"informal patterns detected ({nonstd_hits})")
    if not bits:
        bits.append("no strong automatic cue — use your judgment")
    return " · ".join(bits)
