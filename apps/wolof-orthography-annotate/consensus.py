"""Consensus rules for orthography labels.

A row is sealed when at least ``threshold`` *distinct* volunteers
chose the same label. Ties (two labels at threshold) do not seal —
need a unique winning label with ``count >= threshold``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def tally_labels(votes: list[dict[str, Any]]) -> dict[str, int]:
    """Count votes per label (one vote per stored row)."""
    return dict(Counter(str(v["label"]) for v in votes))


def agreeing_volunteers(
    votes: list[dict[str, Any]], label: str
) -> list[str]:
    """Distinct volunteer ids who chose ``label``, sorted."""
    return sorted(
        {str(v["volunteer_id"]) for v in votes if str(v["label"]) == label}
    )


def winning_label(
    votes: list[dict[str, Any]], threshold: int
) -> tuple[str, list[str]] | None:
    """Return (label, agreeing_ids) if a unique consensus exists."""
    if threshold < 2:
        raise ValueError("threshold must be >= 2")
    if not votes:
        return None

    by_label: dict[str, set[str]] = {}
    for v in votes:
        lab = str(v["label"])
        by_label.setdefault(lab, set()).add(str(v["volunteer_id"]))

    qualified = [
        (lab, sorted(ids))
        for lab, ids in by_label.items()
        if len(ids) >= threshold
    ]
    if not qualified:
        return None
    # Prefer the label with most agreeing volunteers; refuse ties.
    qualified.sort(key=lambda item: (-len(item[1]), item[0]))
    best_lab, best_ids = qualified[0]
    if len(qualified) > 1 and len(qualified[1][1]) == len(best_ids):
        return None
    return best_lab, best_ids
