"""Uncertainty estimates for aggregate metrics.

With 61 scored questions, a three-point difference between configurations is
within sampling noise. Reporting an interval rather than a point estimate is
what makes a comparison honest.
"""

import numpy as np
from scipy.stats import binomtest, wilcoxon

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


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> dict:
    """Agreement between two raters, corrected for agreement by chance.

    Raw agreement flatters a judge on a skewed set: if 85% of answers are
    correct, a judge that says "correct" every time agrees 85% of the time
    while carrying no information. Kappa subtracts the agreement two raters
    would reach by guessing at the observed rates, so 0 means no better than
    chance and 1 means perfect.
    """
    if not rater_a:
        return {"kappa": 0.0, "agreement": 0.0, "n": 0}

    labels = sorted(set(rater_a) | set(rater_b))
    n = len(rater_a)

    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    expected = sum(
        (rater_a.count(label) / n) * (rater_b.count(label) / n) for label in labels
    )

    kappa = 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "kappa": float(kappa),
        "agreement": float(observed),
        "n": n,
        "confusion": {
            f"{a}->{b}": sum(
                1 for x, y in zip(rater_a, rater_b, strict=True) if x == a and y == b
            )
            for a in labels
            for b in labels
        },
    }


def mcnemar(baseline: list[bool], variant: list[bool]) -> dict:
    """Paired comparison of two systems on the same questions.

    Only questions where the two disagree carry information. If the variant
    fixes many that the baseline missed and breaks few that it solved, the
    improvement is real regardless of how much the aggregate intervals overlap.
    """
    fixed = sum(1 for b, v in zip(baseline, variant, strict=True) if not b and v)
    broken = sum(1 for b, v in zip(baseline, variant, strict=True) if b and not v)

    if fixed + broken == 0:
        return {"fixed": 0, "broken": 0, "p_value": 1.0}

    result = binomtest(fixed, fixed + broken, 0.5)
    return {"fixed": fixed, "broken": broken, "p_value": float(result.pvalue)}


def wilcoxon_paired(baseline: list[float], variant: list[float]) -> dict:
    """Paired test on a continuous metric.

    Binarising a ranking into hit-or-miss discards the information that
    matters most when a reranker moves answers from rank 9 to rank 6. The
    signed-rank test uses the magnitude of each per-question change.
    """
    deltas = [v - b for b, v in zip(baseline, variant, strict=True)]
    if not any(deltas):
        return {"mean_delta": 0.0, "p_value": 1.0, "improved": 0, "worsened": 0}

    result = wilcoxon(baseline, variant)
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "p_value": float(result.pvalue),
        "improved": sum(1 for d in deltas if d > 0),
        "worsened": sum(1 for d in deltas if d < 0),
    }
