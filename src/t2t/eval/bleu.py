"""Corpus BLEU (Papineni et al. 2002), reported for comparability with the
data-to-text literature, which quotes BLEU on RotoWire universally
(Wiseman et al. 2017; Puduppully & Lapata 2019).

BLEU measures n-gram overlap with the human reference, i.e. how closely
the output imitates journalistic prose. It is not a faithfulness metric
and can rank a hallucinating system above a faithful one whose phrasing
is more literal, which is the motivation for PARENT (Dhingra et al. 2019)
and for the relational checker. BLEU is reported as a fluency anchor and
a literature bridge, never as evidence of groundedness.

Implemented locally (no sacrebleu/nltk dependency, matching the vendored
PARENT precedent) and structured so that per-item n-gram counts can be
summed: corpus BLEU is a function of summed counts, so the existing paired
bootstrap machinery gives it confidence intervals by resampling items
(Koehn, EMNLP 2004).

Tokenisation is the punctuation-splitting tokeniser used by PARENT, applied
to BOTH sides: RotoWire references are pre-tokenised ("76ers , 122 - 95")
while model output is raw ("76ers, 122-95"), and whitespace splitting would
score identical content as different n-grams (audit finding C9).
"""
from __future__ import annotations

import math
import re
from collections import Counter

MAX_N = 4


def tokenise(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def item_counts(prediction: str, references: list[str]) -> dict:
    """Clipped n-gram match counts, candidate n-gram totals, and lengths for
    ONE item. Summing these across items and calling `bleu_from_counts`
    yields corpus BLEU."""
    hyp = tokenise(prediction)
    refs = [tokenise(r) for r in references if r]
    out: dict[str, float] = {}
    for n in range(1, MAX_N + 1):
        hyp_ng = _ngrams(hyp, n)
        # clip each candidate n-gram by its max count in any single reference
        max_ref: Counter = Counter()
        for ref in refs:
            ref_ng = _ngrams(ref, n)
            for g, c in ref_ng.items():
                if c > max_ref[g]:
                    max_ref[g] = c
        out[f"bleu_match_{n}"] = sum(min(c, max_ref[g]) for g, c in hyp_ng.items())
        out[f"bleu_total_{n}"] = max(0, len(hyp) - n + 1)
    out["bleu_hyp_len"] = len(hyp)
    # effective reference length: the reference closest in length to the
    # candidate (standard multi-reference BLEU; a no-op for single-reference)
    out["bleu_ref_len"] = (
        min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1] if refs else 0
    )
    return out


def bleu_from_counts(
    matches: list[float], totals: list[float], hyp_len: float, ref_len: float
) -> float:
    """Corpus BLEU from summed counts. Returns a 0-100 score (the convention
    in the data-to-text literature)."""
    if hyp_len == 0 or any(t == 0 for t in totals):
        return 0.0
    log_p = 0.0
    for m, t in zip(matches, totals):
        if m == 0:
            return 0.0  # a zero n-gram precision zeroes corpus BLEU
        log_p += math.log(m / t) / MAX_N
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * math.exp(log_p)


def compute_bleu(predictions: list[str], references: list[list[str]]) -> dict:
    """references[i] is the list of reference texts for item i."""
    per_item = [item_counts(p, r) for p, r in zip(predictions, references)]
    if not per_item:
        return {"bleu": None, "per_item": []}
    agg = {k: sum(d[k] for d in per_item) for k in per_item[0]}
    score = bleu_from_counts(
        [agg[f"bleu_match_{n}"] for n in range(1, MAX_N + 1)],
        [agg[f"bleu_total_{n}"] for n in range(1, MAX_N + 1)],
        agg["bleu_hyp_len"],
        agg["bleu_ref_len"],
    )
    return {
        "bleu": round(score, 2),
        "brevity_penalty": round(
            1.0
            if agg["bleu_hyp_len"] > agg["bleu_ref_len"]
            else math.exp(1 - agg["bleu_ref_len"] / max(1, agg["bleu_hyp_len"])),
            4,
        ),
        "hyp_len": agg["bleu_hyp_len"],
        "ref_len": agg["bleu_ref_len"],
        "per_item": per_item,
    }
