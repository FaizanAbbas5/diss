"""Deterministic Table -> markdown serialisation.

This format is frozen across all experiment arms; any change invalidates
comparability of cached runs (the config hash will not catch it).
"""
from __future__ import annotations

from ..data.types import Cell, Section, Table


def _fmt(cell: Cell) -> str:
    if cell is None:
        return "N/A"
    return str(cell).replace("|", "\\|")


def _section_md(sec: Section) -> str:
    lines = [f"### {sec.name}"]
    if len(sec.rows) == 1:
        # Single-row sections read better transposed as attribute/value pairs.
        lines += ["| attribute | value |", "| --- | --- |"]
        lines += [f"| {c} | {_fmt(v)} |" for c, v in zip(sec.columns, sec.rows[0])]
    else:
        lines.append("| " + " | ".join(sec.columns) + " |")
        lines.append("|" + " --- |" * len(sec.columns))
        lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in sec.rows]
    return "\n".join(lines)


def to_markdown(table: Table) -> str:
    return "\n\n".join(_section_md(s) for s in table.sections)
