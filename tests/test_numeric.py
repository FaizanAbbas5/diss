"""Numeric featurisation and per-column statistics."""
import math

from t2t.data.types import Example, Section, Table
from t2t.encode.numeric import compute_column_stats, numeric_features


def test_z_score_uses_column_stats():
    z, _, _, _ = numeric_features(14, mean=10.0, std=2.0)
    assert z == 2.0
    z, _, _, _ = numeric_features(10, mean=10.0, std=2.0)
    assert z == 0.0


def test_zero_std_guard_gives_zero_z():
    assert numeric_features(42, mean=42.0, std=0.0)[0] == 0.0
    assert numeric_features(42)[0] == 0.0  # no stats at all


def test_z_is_clamped_against_outliers():
    z, _, _, _ = numeric_features(1e9, mean=0.0, std=1.0)
    assert z == 10.0
    z, _, _, _ = numeric_features(-1e9, mean=0.0, std=1.0)
    assert z == -10.0


def test_log_sign_isint_features():
    _, log_v, sign, is_int = numeric_features(99)
    assert math.isclose(log_v, math.log1p(99))
    assert sign == 1.0 and is_int == 1.0
    _, _, sign, is_int = numeric_features(-2.5)
    assert sign == -1.0 and is_int == 0.0
    _, log_v, sign, _ = numeric_features(0)
    assert log_v == 0.0 and sign == 0.0


def _example(sections):
    return Example(id="x", table=Table(sections))


def test_stats_are_per_column_and_section_scoped():
    exs = [
        _example(
            [
                Section("teams", ["TEAM", "PTS"], [["A Ants", 100], ["B Bees", 110]]),
                Section("players", ["PLAYER_NAME", "PTS"], [["Al Ant", 10], ["Bo Bee", 30]]),
            ]
        )
    ]
    stats = compute_column_stats(exs)
    assert stats["teams|PTS"] == [105.0, 5.0]
    assert stats["players|PTS"] == [20.0, 10.0]
    assert "teams|TEAM" not in stats  # strings contribute no numeric stats


def test_stats_ignore_none_and_pool_across_examples():
    exs = [
        _example([Section("s", ["v"], [[2]])]),
        _example([Section("s", ["v"], [[None]])]),
        _example([Section("s", ["v"], [[4]])]),
    ]
    stats = compute_column_stats(exs)
    mean, std = stats["s|v"]
    assert mean == 3.0 and std == 1.0
