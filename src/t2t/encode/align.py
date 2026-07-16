"""Reference <-> cell alignment: aux-loss labels and reference filtering.

A cell counts as "mentioned" in a reference if its numeric value appears in
the text (t2t.eval.numbers.extract_numbers) or its string value matches on
a word boundary. Entity cells (key columns) additionally go through the
alias machinery of t2t.facts (nicknames, unique surnames) when the table is
a box score, so "DeRozan" credits the "DeMar DeRozan" cell.

filter_reference implements the plan.md answer to the 45%-extrinsic
problem: keep only reference sentences that contain at least one aligned
fact and no relational error (t2t.eval.relations). Both filtered and
unfiltered paths stay runnable — a dissertation decision point.
"""
from __future__ import annotations

import re

import torch

from ..data.types import Table
from ..eval.claims import split_sentences
from ..eval.numbers import extract_numbers
from ..eval.relations import score_text_relations
from ..facts import GameFacts


def _key(x: float) -> float:
    return round(float(x), 6)


def _word_boundary_match(value: str, text_lower: str) -> bool:
    """Lookarounds rather than \\b so values with punctuation still anchor
    ('£20-25', '5 out of 5') — same approach as GameFacts._find_alias."""
    value = value.strip().lower()
    if not value:
        return False
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text_lower) is not None


def alignment_labels(
    table: Table, reference: str, facts: GameFacts | None = None
) -> torch.Tensor:
    """Per-cell 0/1 'mentioned in reference' labels, [R, C], padded with 0.

    Row/column layout matches featurise_table exactly (sections in order,
    C = widest row) so labels align with encoder cells one-to-one.
    """
    if facts is None:
        facts = GameFacts(table)
    text_lower = reference.lower()
    ref_nums = {_key(n) for n in extract_numbers(reference)}
    if facts.ok:
        mentioned_entities = {n for _, n in facts.find_teams(text_lower)}
        mentioned_entities.update(facts.find_players(text_lower))
    else:
        mentioned_entities = set()

    n_rows = sum(len(sec.rows) for sec in table.sections)
    n_cols = max(len(sec.columns) for sec in table.sections)
    labels = torch.zeros(n_rows, n_cols)
    r = 0
    for sec in table.sections:
        for row in sec.rows:
            for c, cell in enumerate(row):
                if cell is None:
                    continue
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    if _key(cell) in ref_nums:
                        labels[r, c] = 1.0
                elif c == sec.key_column and str(cell) in mentioned_entities:
                    labels[r, c] = 1.0
                elif _word_boundary_match(str(cell), text_lower):
                    labels[r, c] = 1.0
            r += 1
    return labels


def filter_reference(reference: str, table: Table, facts: GameFacts | None = None) -> str:
    """Keep only sentences with >=1 aligned fact and no relational error.

    Returns the surviving sentences joined with spaces; '' if none survive
    (callers drop such items). Relational checking is a no-op on tables the
    checker does not recognise (e.g. E2E), by construction of GameFacts.ok.
    """
    if facts is None:
        facts = GameFacts(table)
    kept = []
    for sentence in split_sentences(reference):
        if score_text_relations(sentence, table)["n_wrong"]:
            continue
        if alignment_labels(table, sentence, facts).any():
            kept.append(sentence)
    return " ".join(kept)
