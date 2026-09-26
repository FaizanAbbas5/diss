"""Number-accuracy checker: every number in the generated text must appear
in the table or be derivable by simple arithmetic (same-column pairwise
sums and absolute differences, e.g. victory margins or combined points).

Known limitations: digit numerals only (spelled-out numbers such as
"three-pointer" are ignored); derivations limited to pairwise same-column
sums and differences.
"""
from __future__ import annotations

import re

from ..data.types import Table

_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+|\.\d+|\d+")


def extract_numbers(text: str) -> list[float]:
    # ".500" (shooting idiom) parses as 0.5, not 500 (audit C8)
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(text)]


def _key(x: float) -> float:
    return round(x, 6)


def table_numbers(table: Table) -> set[float]:
    nums: set[float] = set()
    for _, _, cell in table.cells():
        if isinstance(cell, (int, float)):
            nums.add(_key(float(cell)))
        elif isinstance(cell, str):
            nums.update(_key(v) for v in extract_numbers(cell))
    return nums


def derived_numbers(table: Table, max_column_cells: int = 40) -> set[float]:
    derived: set[float] = set()
    for sec in table.sections:
        for ci in range(len(sec.columns)):
            vals = [float(row[ci]) for row in sec.rows if isinstance(row[ci], (int, float))]
            if len(vals) < 2 or len(vals) > max_column_cells:
                continue
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    derived.add(_key(vals[i] + vals[j]))
                    derived.add(_key(abs(vals[i] - vals[j])))
    return derived


def score_text(text: str, table: Table, allow_derived: bool = True) -> dict:
    """Tiered support: 'exact' (number appears in a cell) is a stronger
    signal than 'derived' (reachable via some same-column sum/difference);
    on dense tables the derived set covers most small integers, so the two
    tiers are reported separately."""
    exact_set = table_numbers(table)
    derived_set = derived_numbers(table) if allow_derived else set()
    found = extract_numbers(text)
    # fraction/percent duality: ".500" is supported by a 50 in a _PCT cell
    exact_set |= {_key(v / 100) for v in list(exact_set)}
    n_exact = sum(1 for n in found if _key(n) in exact_set)
    derived = [n for n in found if _key(n) not in exact_set and _key(n) in derived_set]
    unsupported = [
        n for n in found if _key(n) not in exact_set and _key(n) not in derived_set
    ]
    n_supported = n_exact + len(derived)
    return {
        "n_numbers": len(found),
        "n_exact": n_exact,
        "n_derived": len(derived),
        "n_supported": n_supported,
        "unsupported": unsupported,
        "support_rate": n_supported / len(found) if found else None,
        "exact_rate": n_exact / len(found) if found else None,
    }


def score_run(outputs: dict[str, str], tables: dict[str, Table]) -> dict:
    """outputs/tables keyed by example id (must share keys)."""
    per_item = {i: score_text(text, tables[i]) for i, text in outputs.items()}
    with_numbers = [s for s in per_item.values() if s["n_numbers"] > 0]
    total = sum(s["n_numbers"] for s in with_numbers)
    supported = sum(s["n_supported"] for s in with_numbers)
    exact = sum(s["n_exact"] for s in with_numbers)
    return {
        "per_item": per_item,
        "aggregate": {
            "items": len(per_item),
            "items_with_numbers": len(with_numbers),
            "number_support_rate_micro": supported / total if total else None,
            "number_exact_rate_micro": exact / total if total else None,
            "pct_items_with_unsupported_number": (
                sum(1 for s in with_numbers if s["unsupported"]) / len(per_item)
                if per_item
                else None
            ),
        },
    }
