"""Dataset registry: t2t.data.load(name, split) -> list[Example]."""
from __future__ import annotations

from .types import Cell, Example, Section, Table

__all__ = ["Cell", "Example", "Section", "Table", "load"]


def load(
    name: str,
    split: str,
    data_dir: str | None = None,
    limit: int | None = None,
    sample_seed: int | None = None,
) -> list[Example]:
    """With sample_seed, `limit` items are drawn as a seeded random sample
    (original order preserved) — required for final runs, since first-N on
    RotoWire is temporally biased. Without it, first-N (dev/smoke only)."""
    if name in ("e2e", "e2e_cleaned"):
        from ....diss_code.src.t2t.data import e2e

        examples = e2e.load(split, data_dir or "data/e2e")
    elif name == "rotowire":
        from ....diss_code.src.t2t.data import rotowire

        examples = rotowire.load(split, data_dir or "data/rotowire")
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    if limit and sample_seed is not None and limit < len(examples):
        import random

        idx = sorted(random.Random(sample_seed).sample(range(len(examples)), limit))
        return [examples[i] for i in idx]
    return examples[:limit] if limit else examples
