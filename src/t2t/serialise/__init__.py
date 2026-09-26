"""Serialiser registry.

`markdown` is the frozen baseline serialisation; `factsheet` is the
template arm's deterministic preparation. Generation backends resolve the
serialiser from the config key `serialisation`; absent means markdown, so
every pre-registry config hash is untouched.
"""
from __future__ import annotations

from typing import Callable

from ..data.types import Table
from .markdown import to_markdown

__all__ = ["to_markdown", "get_serialiser"]


def get_serialiser(name: str = "markdown") -> Callable[[Table], str]:
    if name == "markdown":
        return to_markdown
    if name == "factsheet":
        from ..factsheet import to_factsheet

        return to_factsheet
    raise KeyError(f"Unknown serialisation {name!r}; expected 'markdown' or 'factsheet'")
