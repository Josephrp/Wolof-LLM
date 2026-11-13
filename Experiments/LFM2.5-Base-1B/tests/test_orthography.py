"""Tests for procedural Wolof orthography classifier."""

from __future__ import annotations

from common.orthography import classify_label, classify_orthography, extract_sft_text


def test_standard_diacritics():
    assert classify_label("Jàpp nañu ci jëf ju baax.") == "standard"


def test_non_standard_gn():
    assert classify_label("Danga xam ne gniow dyangore.") in {
        "non_standard",
        "mixed",
    }


def test_french_not_wolof():
    text = "Le président de la république a déclaré que les mesures sont nécessaires."
    assert classify_label(text) == "not_wolof"


def test_empty_not_wolof():
    assert classify_label("  ") == "not_wolof"


def test_mixed_both():
    # diacritics + informal gn pattern
    d = classify_orthography("Ñu ngi dem ak gniow ci jëf.")
    assert d.label in {"mixed", "standard", "non_standard"}


def test_extract_sft():
    messages = [
        {"role": "user", "content": "Naka nga def?"},
        {"role": "assistant", "content": "Maa ngi fi."},
    ]
    assert "Naka" in extract_sft_text(messages)
