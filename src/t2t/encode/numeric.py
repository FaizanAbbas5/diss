"""Numeric value featurisation and per-column z-statistics.

Numeric cells get first-class treatment (plan.md: exact-number recovery is
the failure mode): each numeric value becomes the 4-vector
[z_scored_value, log1p(|v|), sign, is_int]. Z-stats are computed per column
on the *train* split and saved inside the checkpoint, so evaluation uses
exactly the scaling the encoder was trained with.
"""
from __future__ import annotations

import math

from ..data.types import Example

N_FEATURES = 4

# Clamp z-scores so a single outlier cell cannot blow up the encoder input.
_Z_CLAMP = 10.0


def column_key(section_name: str, column_name: str) -> str:
    """Columns are scoped by section: RotoWire 'teams' and 'players' both
    have point-like columns that must not share statistics."""
    return f"{section_name}|{column_name}"


def numeric_features(value: float, mean: float = 0.0, std: float = 0.0) -> list[float]:
    z = (float(value) - mean) / std if std > 0 else 0.0
    z = max(-_Z_CLAMP, min(_Z_CLAMP, z))
    sign = float(value > 0) - float(value < 0)
    return [z, math.log1p(abs(float(value))), sign, float(float(value).is_integer())]


def compute_column_stats(examples: list[Example]) -> dict[str, list[float]]:
    """Per-column [mean, population_std] over all numeric cells in `examples`.

    Keyed by column_key(); JSON-serialisable so it can live in the checkpoint.
    Columns never seen numeric (or seen once, std 0) fall back to z=0 via the
    std<=0 guard in numeric_features.
    """
    values: dict[str, list[float]] = {}
    for ex in examples:
        for sec in ex.table.sections:
            for row in sec.rows:
                for col, cell in zip(sec.columns, row):
                    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                        values.setdefault(column_key(sec.name, col), []).append(float(cell))
    stats: dict[str, list[float]] = {}
    for key, vals in values.items():
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[key] = [mean, math.sqrt(var)]
    return stats
