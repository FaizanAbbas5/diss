"""Dataset registry: t2t.data.load(name, split) -> list[Example]."""
from __future__ import annotations

from .types import Cell, Example, Section, Table

__all__ = ["Cell", "Example", "Section", "Table", "load"]


def load(
    name: str,
    split: str,
    data_dir: str | None = None,
    limit: int | None = None,
) -> list[Example]:
    if name in ("e2e", "e2e_cleaned"):
        from . import e2e

        examples = e2e.load(split, data_dir or "data/e2e")
    elif name == "rotowire":
        from . import rotowire

        examples = rotowire.load(split, data_dir or "data/rotowire")
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    return examples[:limit] if limit else examples
