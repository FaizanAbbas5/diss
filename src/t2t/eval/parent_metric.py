"""PARENT metric (Dhingra et al. 2019) on canonical Tables.

Uses the vendored reference implementation in `_parent_impl` (see its
docstring for provenance). The table is flattened to (attribute-tokens,
value-tokens) pairs; multi-row sections prefix the attribute with the
row's entity so values stay disambiguated. Everything is lowercased and
whitespace-tokenised, matching common PARENT usage.
"""
from __future__ import annotations

from ..data.types import Table
from ._parent_impl import parent


def _tok(text: str) -> list[str]:
    return text.lower().split()


def table_to_parent_format(table: Table) -> list[tuple[list[str], list[str]]]:
    pairs: list[tuple[list[str], list[str]]] = []
    for sec in table.sections:
        for row in sec.rows:
            entity = str(row[sec.key_column])
            for col, val in zip(sec.columns, row):
                if val is None:
                    continue
                attr = col if len(sec.rows) == 1 else f"{entity} {col}"
                pairs.append((_tok(attr), _tok(str(val))))
    return pairs


def compute_parent(
    predictions: list[str],
    references: list[list[str]],
    tables: list[Table],
) -> dict:
    """references[i] is the list of reference texts for item i (multi-ref
    is supported natively: PARENT keeps the best-F reference per item)."""
    precision, recall, f1, per_item_f = parent(
        [_tok(p) for p in predictions],
        [[_tok(r) for r in refs] for refs in references],
        [table_to_parent_format(t) for t in tables],
    )
    return {
        "parent_precision": precision,
        "parent_recall": recall,
        "parent_f1": f1,
        "per_item_f1": per_item_f,
    }
