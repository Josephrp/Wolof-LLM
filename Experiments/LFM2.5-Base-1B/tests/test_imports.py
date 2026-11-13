"""Import smoke for the common package."""


def test_import_common():
    import common
    import common.data
    import common.journal  # noqa: F401


def test_import_model_metrics_optional():
    """model/metrics need torch; skip soft if train group not installed."""
    try:
        import common.metrics
        import common.model  # noqa: F401
    except ImportError as e:
        import pytest

        pytest.skip(f"train deps missing: {e}")
