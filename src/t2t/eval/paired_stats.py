"""Paired bootstrap statistics for comparing generation runs.

All runs in a comparison must have been evaluated on the SAME items (the
battery configs share dataset/split/limit/sample_seed, so per-item metrics
align by example id). Uncertainty is quantified by bootstrap resampling of
items: each resample simulates "what if a different sample of games had been
drawn". Paired differences resample the same item indices in both runs, so
item difficulty cancels out.

Deterministic: every function takes a seed (default 13, matching the runs).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Stat = Callable[..., float]


def mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def ratio_of_sums(num: np.ndarray, den: np.ndarray) -> float:
    """Micro-average rate: sum(num) / sum(den). NaN when the denominator is 0
    (e.g. no numeric claims in a resample)."""
    d = float(np.sum(den))
    return float(np.sum(num)) / d if d else float("nan")


def prop_positive(x: np.ndarray) -> float:
    """Share of items with x > 0 (e.g. items with >=1 relational error)."""
    return float(np.mean(np.asarray(x) > 0))


def _as_cols(columns: Sequence[Sequence[float]]) -> list[np.ndarray]:
    cols = [np.asarray(c, dtype=float) for c in columns]
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("all columns must have the same length")
    if n == 0:
        raise ValueError("empty columns")
    return cols


def bootstrap_ci(
    stat: Stat,
    columns: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for stat(*columns) resampled over items.

    Returns (point, lo, hi). NaN resamples (empty denominators) are dropped.
    """
    cols = _as_cols(columns)
    n = len(cols[0])
    rng = np.random.default_rng(seed)
    point = stat(*cols)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat(*(c[idx] for c in cols))
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(
    stat: Stat,
    columns_a: Sequence[Sequence[float]],
    columns_b: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """CI for stat(run A) - stat(run B), resampling the SAME items in both.

    Columns of A and B must be aligned item-by-item (same ids, same order).
    Returns (diff, lo, hi); the CI excluding 0 marks a robust difference.
    """
    ca, cb = _as_cols(columns_a), _as_cols(columns_b)
    if len(ca[0]) != len(cb[0]):
        raise ValueError("paired runs must have the same number of items")
    n = len(ca[0])
    rng = np.random.default_rng(seed)
    point = stat(*ca) - stat(*cb)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat(*(c[idx] for c in ca)) - stat(*(c[idx] for c in cb))
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def fold_split(n_items: int, n_folds: int, seed: int = 13) -> list[np.ndarray]:
    """Shuffle items once and cut into n_folds disjoint folds (sizes differ by
    at most 1). Deterministic for a given (n_items, n_folds, seed)."""
    if not 2 <= n_folds <= n_items:
        raise ValueError("need 2 <= n_folds <= n_items")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    return [np.sort(f) for f in np.array_split(order, n_folds)]


def fold_stats(
    stat: Stat,
    columns: Sequence[Sequence[float]],
    folds: Sequence[np.ndarray],
) -> list[float]:
    """stat(*columns) restricted to each fold."""
    cols = _as_cols(columns)
    return [stat(*(c[f] for c in cols)) for f in folds]


def rank(values: Sequence[float], higher_better: bool) -> list[int]:
    """1-based ranks (1 = best). NaN ranks last. Ties keep input order."""
    keyed = [
        (float("inf") if np.isnan(v) else (-v if higher_better else v), i)
        for i, v in enumerate(values)
    ]
    out = [0] * len(values)
    for r, (_, i) in enumerate(sorted(keyed), start=1):
        out[i] = r
    return out
