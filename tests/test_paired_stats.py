import numpy as np
import pytest

from t2t.eval.paired_stats import (
    bootstrap_ci,
    fold_split,
    fold_stats,
    mean,
    paired_bootstrap_diff,
    prop_positive,
    rank,
    ratio_of_sums,
)


def test_stats_basics():
    assert mean(np.array([1.0, 2.0, 3.0])) == 2.0
    assert ratio_of_sums(np.array([1, 1]), np.array([2, 2])) == 0.5
    assert np.isnan(ratio_of_sums(np.array([0.0]), np.array([0.0])))
    assert prop_positive(np.array([0, 0, 2, 5])) == 0.5


def test_bootstrap_ci_deterministic_and_sane():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 1.0, 400)
    a = bootstrap_ci(mean, [x], n_boot=2000, seed=13)
    b = bootstrap_ci(mean, [x], n_boot=2000, seed=13)
    assert a == b  # seeded => reproducible
    point, lo, hi = a
    assert lo < point < hi
    assert point == pytest.approx(10.0, abs=0.2)
    assert (hi - lo) == pytest.approx(2 * 1.96 / 20, rel=0.35)  # ~4*SE/2


def test_paired_diff_identical_runs_is_zero():
    x = np.arange(50, dtype=float)
    d, lo, hi = paired_bootstrap_diff(mean, [x], [x], n_boot=500, seed=13)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_diff_detects_constant_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 5.0, 200)
    d, lo, hi = paired_bootstrap_diff(mean, [x + 1.0], [x], n_boot=2000, seed=13)
    assert d == pytest.approx(1.0)
    # pairing cancels the item variance: CI is tight around 1 and excludes 0
    assert lo > 0.9 and hi < 1.1


def test_fold_split_partitions():
    folds = fold_split(150, 3, seed=13)
    assert [len(f) for f in folds] == [50, 50, 50]
    all_idx = np.sort(np.concatenate(folds))
    assert np.array_equal(all_idx, np.arange(150))
    assert [f.tolist() for f in fold_split(150, 3, seed=13)] == \
           [f.tolist() for f in folds]


def test_fold_stats_and_rank():
    x = np.array([1.0] * 10 + [3.0] * 10)
    folds = [np.arange(10), np.arange(10, 20)]
    assert fold_stats(mean, [x], folds) == [1.0, 3.0]
    assert rank([0.1, 0.3, 0.2], higher_better=False) == [1, 3, 2]
    assert rank([0.1, 0.3, 0.2], higher_better=True) == [3, 1, 2]
    assert rank([0.1, float("nan"), 0.2], higher_better=False) == [1, 3, 2]
