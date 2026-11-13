"""Guard against silent LFM short-conv slow-path fallbacks."""

from __future__ import annotations

import os

import pytest


def test_require_causal_conv1d_fails_closed_without_kernel(monkeypatch):
    try:
        from common import model
    except ImportError as exc:
        pytest.skip(f"train deps missing: {exc}")

    monkeypatch.setattr(model, "causal_conv1d_fn", None)
    monkeypatch.setattr(model, "causal_conv1d_update", None)
    monkeypatch.setattr(model.torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("LFM_ALLOW_SLOW_CONV", raising=False)

    with pytest.raises(RuntimeError, match="causal_conv1d"):
        model.require_causal_conv1d()


def test_require_causal_conv1d_allows_emergency_override(monkeypatch):
    try:
        from common import model
    except ImportError as exc:
        pytest.skip(f"train deps missing: {exc}")

    monkeypatch.setattr(model, "causal_conv1d_fn", None)
    monkeypatch.setattr(model, "causal_conv1d_update", None)
    monkeypatch.setattr(model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("LFM_ALLOW_SLOW_CONV", "1")

    model.require_causal_conv1d()


def test_causal_conv1d_available_matches_imports():
    try:
        from common import model
    except ImportError as exc:
        pytest.skip(f"train deps missing: {exc}")

    expected = (
        model.causal_conv1d_fn is not None and model.causal_conv1d_update is not None
    )
    assert model.causal_conv1d_available() is expected
    # Windows/dev hosts may lack the Linux CUDA wheel; GPU train hosts must have it.
    if os.name != "nt" and model.torch.cuda.is_available():
        assert expected is True
