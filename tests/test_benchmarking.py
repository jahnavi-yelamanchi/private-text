import pytest

from app.benchmarking import percentile


def test_percentile_interpolates_between_observations() -> None:
    assert percentile([1.0, 2.0, 4.0, 10.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 4.0, 10.0], 0.95) == pytest.approx(9.1)
