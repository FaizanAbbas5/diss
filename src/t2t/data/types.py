"""Canonical table representation shared by all datasets, the serialiser,
the metrics, and (later) the neural encoder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

Cell = int | float | str | None


@dataclass
class Section:
    """One named sub-table (e.g. 'teams', 'players').

    key_column identifies the column naming the row's entity, used later
    for claim retrieval and PARENT table formatting.
    """

    name: str
    columns: list[str]
    rows: list[list[Cell]]
    key_column: int = 0

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Section {self.name!r}: row width {len(row)} != {len(self.columns)} columns"
                )


@dataclass
class Table:
    sections: list[Section]

    def cells(self) -> Iterator[tuple[str, str, Cell]]:
        """Yield (section_name, column_name, value) for every cell."""
        for sec in self.sections:
            for row in sec.rows:
                for col, val in zip(sec.columns, row):
                    yield sec.name, col, val


@dataclass
class Example:
    id: str
    table: Table
    references: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
