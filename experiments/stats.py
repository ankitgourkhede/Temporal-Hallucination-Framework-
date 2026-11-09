"""Paired statistical tests, bootstrap intervals, and Holm correction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import binomtest, chi2


def mcnemar(b: int, c: int) -> dict[str, float | str]:
    if b < 0 or c < 0:
        raise ValueError("discordant counts cannot be negative")
    total = b + c
    if total == 0:
        return {"method": "exact", "statistic": 0.0, "p_value": 1.0}
    if total < 25:
        result = binomtest(min(b, c), total, 0.5, alternative="two-sided")
        return {"method": "exact", "statistic": float(min(b, c)), "p_value": float(result.pvalue)}
    statistic = (abs(b - c) - 1) ** 2 / total
    return {"method": "chi2_continuity_corrected", "statistic": float(statistic), "p_value": float(chi2.sf(statistic, 1))}


def paired_bootstrap(
    left: Iterable[float], right: Iterable[float], *, resamples: int = 10_000, seed: int = 42
) -> dict[str, float]:
    left_values, right_values = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    if left_values.shape != right_values.shape or not left_values.size:
        raise ValueError("paired arrays must be non-empty and have the same shape")
    differences = left_values - right_values
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[draws].mean(axis=1)
    # Two-sided bootstrap p-value for H0: mean difference = 0 (share of resampled means on the far side of 0).
    p_value = min(1.0, 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean())))
    if not differences.any():
        p_value = 1.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_value": p_value,
    }


def bootstrap_ci(values: Iterable[float], *, resamples: int = 1_000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if not values.size:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running_max = max(running_max, (count - index) * p_value)
        adjusted[name] = min(1.0, running_max)
    return adjusted