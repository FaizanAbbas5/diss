"""Learned table representation for Arm 2: encoder -> soft tokens -> projector."""
from __future__ import annotations

from .align import alignment_labels, filter_reference
from .numeric import compute_column_stats, numeric_features
from .table_encoder import (
    ColumnVocab,
    Projector,
    SoftTableModel,
    TableEncoder,
    featurise_table,
    string_bucket,
)

__all__ = [
    "alignment_labels",
    "filter_reference",
    "compute_column_stats",
    "numeric_features",
    "ColumnVocab",
    "Projector",
    "SoftTableModel",
    "TableEncoder",
    "featurise_table",
    "string_bucket",
]
