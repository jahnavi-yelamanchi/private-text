"""Small runtime-independent helpers for recorded benchmark summaries."""

from __future__ import annotations

from collections.abc import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linear-interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("Cannot calculate a percentile from no values.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile fraction must be between zero and one.")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
