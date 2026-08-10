"""Uncertainty estimates for aggregate metrics.

With 61 scored questions, a three-point difference between configurations is
within sampling noise. Reporting an interval rather than a point estimate is
what makes a comparison honest.
"""

import numpy as np

DEFAULT_RESAMPLES = 1000
CONFIDENCE = 0.95


def bootstrap_ci(
    values: list[float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lower bound, upper bound).

    Questions are resampled with replacement to estimate how much the mean
    would move if a different set of questions had been written.
    """
    if not values:
        return 0.0, 0.0, 0.0

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(resamples, len(array)))].mean(
        axis=1
    )

    tail = (1.0 - confidence) / 2.0
    return (
        float(array.mean()),
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )
