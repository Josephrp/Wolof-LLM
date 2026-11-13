"""Unit tests for consensus sealing (no Hub I/O)."""

from __future__ import annotations

from consensus import tally_labels, winning_label


def _v(vid: str, label: str) -> dict:
    return {"volunteer_id": vid, "label": label}


def test_threshold_two_agrees():
    votes = [_v("a", "standard"), _v("b", "standard")]
    assert winning_label(votes, 2) == ("standard", ["a", "b"])


def test_threshold_three_needs_three():
    votes = [_v("a", "standard"), _v("b", "standard"), _v("c", "mixed")]
    assert winning_label(votes, 3) is None
    votes.append(_v("d", "standard"))
    assert winning_label(votes, 3) == ("standard", ["a", "b", "d"])


def test_tie_does_not_seal():
    votes = [
        _v("a", "standard"),
        _v("b", "standard"),
        _v("c", "non_standard"),
        _v("d", "non_standard"),
    ]
    assert winning_label(votes, 2) is None


def test_same_volunteer_counted_once_per_label_set():
    # Storage prevents double-vote; if duplicates slipped in, set collapses them.
    votes = [_v("a", "standard"), _v("a", "standard"), _v("b", "standard")]
    assert winning_label(votes, 2) == ("standard", ["a", "b"])


def test_tally():
    votes = [_v("a", "mixed"), _v("b", "mixed"), _v("c", "standard")]
    assert tally_labels(votes) == {"mixed": 2, "standard": 1}
