"""Claim-level NLI verification (classifier-based faithfulness metric).

Each output sentence is treated as a claim and classified as entailed /
neutral / contradicted against a premise built from the table rows the
sentence actually mentions (entity-match retrieval). Sentence-level
decomposition + retrieval answers the granularity-mismatch critique of
whole-text NLI (Huang et al. 2025, section 4.1.2).

This catches the hallucinations the number checker cannot see: unsupported
qualitative claims ("upscale eatery") and contradictions ("family-friendly"
when the table says familyFriendly: no).

Known limitations: regex sentence splitting; premise capped at
`max_premise_chars` (very long row sets are truncated in retrieval
order); NLI operates on "column: value" premises rather than fluent
sentences.
"""
from __future__ import annotations

import re
from typing import Callable

from ..data.types import Section, Table

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Split on sentence-final punctuation followed by whitespace and a capital
# or digit.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

_STOPWORDS = {"the", "and", "for", "out", "city", "new"}

ScorerFn = Callable[[list[tuple[str, str]]], list[dict[str, float]]]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


# NLI hypotheses are single clauses, not compound sentences: the evaluator
# audit found factually-correct "X led ..., while Y paced ..." sentences
# labelled CONTRADICTION. Split at ';', ', while', and ', and' only when a
# new capitalised subject follows (mirrors the relational clause splitter).
_NLI_CLAUSE_RE = re.compile(r";\s+|,\s+while\s+|,\s+and\s+(?=[A-Z])")


def split_claim_units(text: str) -> list[str]:
    """Claim units for NLI scoring: clause-level, with degenerate fragments
    (e.g. "23).") dropped. The single tokenisation used by both the batch
    path (score_run_claims) and the per-item path (score_text_claims), so
    all consumers report identical numbers."""
    units = []
    for s in split_sentences(text):
        for c in _NLI_CLAUSE_RE.split(s):
            c = c.strip(" ,")
            if c and len(re.findall(r"[A-Za-z]{2,}", c)) >= 3:
                units.append(c)
    return units


def _entity_tokens(entity: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", entity.lower())
        if len(t) >= 3 and t not in _STOPWORDS
    }


def _row_facts(sec: Section, row: list) -> str:
    entity = str(row[sec.key_column])
    facts = "; ".join(
        f"{col}: {val}"
        for i, (col, val) in enumerate(zip(sec.columns, row))
        if val is not None and i != sec.key_column
    )
    return f"{sec.name} {entity} -- {facts}"


def _derived_result_facts(table: Table) -> list[str]:
    """Explicit game-result sentences for two-row sections with a points
    column. NLI models cannot compare 122 > 95 themselves, so without this
    every true "X defeated Y" sentence is judged unverifiable."""
    facts = []
    for sec in table.sections:
        # Two competitors with a points column, not e.g. a 2-player section.
        if len(sec.rows) != 2 or "team" not in sec.name.lower():
            continue
        for ci, col in enumerate(sec.columns):
            if col.upper().endswith("PTS"):
                a, b = sec.rows[0][ci], sec.rows[1][ci]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a != b:
                    (wi, wp), (li, lp) = ((0, a), (1, b)) if a > b else ((1, b), (0, a))
                    winner = sec.rows[wi][sec.key_column]
                    loser = sec.rows[li][sec.key_column]
                    facts.append(
                        f"The {winner} defeated the {loser} {wp} to {lp} ; "
                        f"the {winner} won the game and the {loser} lost the game"
                    )
                break
    return facts


def build_premise(table: Table, sentence: str, max_premise_chars: int = 2000) -> str:
    """Facts of every single-row section plus multi-row rows whose entity
    is mentioned in the sentence; falls back to the first multi-row section
    (e.g. team totals) when the sentence names no entity. Derived result
    facts are prepended so premise truncation can never drop them."""
    sent_l = sentence.lower()
    parts: list[str] = list(_derived_result_facts(table))
    matched_multi = False
    for sec in table.sections:
        if len(sec.rows) == 1:
            parts.append(_row_facts(sec, sec.rows[0]))
            continue
        for row in sec.rows:
            if _entity_tokens(str(row[sec.key_column])) & set(re.findall(r"[a-z0-9]+", sent_l)):
                parts.append(_row_facts(sec, row))
                matched_multi = True
    if not matched_multi:
        for sec in table.sections:
            if len(sec.rows) > 1:
                parts.extend(_row_facts(sec, row) for row in sec.rows[:2])
                break
    return " . ".join(parts)[:max_premise_chars]


class NLIScorer:
    """Batched (premise, hypothesis) -> label probabilities. Label indices
    are read from the model config, not hardcoded."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        batch_size: int = 16,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.labels = {
            i: label.lower() for i, label in self.model.config.id2label.items()
        }

    def __call__(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        import time

        import torch

        results: list[dict[str, float]] = []
        t0 = time.time()
        for start in range(0, len(pairs), self.batch_size):
            if start:
                rate = start / (time.time() - t0)
                remaining = (len(pairs) - start) / rate if rate else 0
                print(
                    f"NLI: {start}/{len(pairs)} pairs, ~{remaining:.0f}s left",
                    flush=True,
                )
            batch = pairs[start : start + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                probs = self.model(**enc).logits.softmax(dim=-1)
            for row in probs:
                results.append(
                    {self.labels[i]: round(float(p), 4) for i, p in enumerate(row)}
                )
        return results


def score_text_claims(text: str, table: Table, scorer: ScorerFn) -> dict:
    units = split_claim_units(text)
    if not units:
        return {"n_claims": 0, "entailed_rate": None, "claims": []}
    probs = scorer([(build_premise(table, u), u) for u in units])
    claims = []
    for sentence, p in zip(units, probs):
        label = max(p, key=p.get)
        claims.append({"sentence": sentence, "label": label, "probs": p})
    n_entailed = sum(1 for c in claims if c["label"] == "entailment")
    return {
        "n_claims": len(claims),
        "n_entailed": n_entailed,
        "n_contradicted": sum(1 for c in claims if c["label"] == "contradiction"),
        "entailed_rate": n_entailed / len(claims),
        "claims": claims,
    }


def score_run_claims(
    outputs: dict[str, str],
    tables: dict[str, Table],
    scorer: ScorerFn | None = None,
    model_name: str = DEFAULT_NLI_MODEL,
) -> dict:
    """Batch all (premise, sentence) pairs across items through one scorer."""
    if scorer is None:
        scorer = NLIScorer(model_name)

    pair_index: list[tuple[str, str]] = []  # (item_id, claim unit)
    pairs: list[tuple[str, str]] = []
    for i, text in outputs.items():
        for s in split_claim_units(text):
            pair_index.append((i, s))
            pairs.append((build_premise(tables[i], s), s))
    all_probs = scorer(pairs)

    per_item: dict[str, dict] = {
        i: {"n_claims": 0, "n_entailed": 0, "n_contradicted": 0, "claims": []}
        for i in outputs
    }
    for (i, sentence), p in zip(pair_index, all_probs):
        label = max(p, key=p.get)
        item = per_item[i]
        item["n_claims"] += 1
        item["n_entailed"] += label == "entailment"
        item["n_contradicted"] += label == "contradiction"
        item["claims"].append({"sentence": sentence, "label": label, "probs": p})
    for item in per_item.values():
        item["entailed_rate"] = (
            item["n_entailed"] / item["n_claims"] if item["n_claims"] else None
        )

    scored = [v for v in per_item.values() if v["n_claims"] > 0]
    total = sum(v["n_claims"] for v in scored)
    return {
        "per_item": per_item,
        "aggregate": {
            "items": len(per_item),
            "total_claims": total,
            "claim_entailed_rate_micro": (
                sum(v["n_entailed"] for v in scored) / total if total else None
            ),
            "pct_items_with_contradiction": (
                sum(1 for v in scored if v["n_contradicted"]) / len(per_item)
                if per_item
                else None
            ),
        },
    }
