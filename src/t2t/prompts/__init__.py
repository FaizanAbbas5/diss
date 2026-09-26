"""Prompt registry.

The baseline prompt is frozen and reused verbatim by every arm; only the
content of the {table} slot differs between arms.

A prompt spec may carry an "extract_marker": everything before the last
occurrence of the marker is treated as scratch work (e.g. a fact list) and
stripped from the recorded output. The raw text is kept alongside for
debugging.
"""
from __future__ import annotations

import re

_SYSTEM = (
    "You are a careful assistant that writes concise, factual text from structured data."
)

_FAITHFUL_SUFFIX = (
    " Use only facts that are explicitly stated in the data above."
    " Do not add, infer, or embellish any information, and do not include opinions."
)

_E2E_USER = (
    "Here is structured information about a restaurant:\n\n{table}\n\n"
    "Write a one- or two-sentence natural-sounding description of this restaurant."
)

_E2E_STRICT_SUFFIX = _FAITHFUL_SUFFIX + (
    " Every attribute value you mention must match the table exactly."
    " If you are unsure whether something is supported by the table, leave it out."
)

_E2E_STRUCTURED = (
    "Here is structured information about a restaurant:\n\n{table}\n\n"
    "First, list every attribute-value pair from the table as short bullet"
    " points, one per line, prefixed with '- '. Then, on a new line beginning"
    " with 'SUMMARY:', write a one- or two-sentence natural-sounding"
    " description of the restaurant that uses only those listed facts."
)

_ROTOWIRE_USER = (
    "Here are the box-score statistics of an NBA game:\n\n{table}\n\n"
    "Write a short game summary of roughly 6-8 sentences in the style of a sports report."
)

_ROTOWIRE_STRICT_SUFFIX = _FAITHFUL_SUFFIX + (
    " Every number and every player or team name you mention must appear in"
    " the table. Do not estimate, round, or invent statistics. If you are"
    " unsure of a value, omit it."
)

_ROTOWIRE_STRUCTURED = (
    "Here are the box-score statistics of an NBA game:\n\n{table}\n\n"
    "First, list 6-10 key facts from the box score (winner, final score, top"
    " performers) as short bullet points, one per line, prefixed with '- ',"
    " copying all numbers exactly from the table. Then, on a new line"
    " beginning with 'SUMMARY:', write a game summary of roughly 6-8 sentences"
    " in the style of a sports report, using only those listed facts."
)

PROMPTS: dict[str, dict[str, str]] = {
    "e2e/plain_v1": {"system": _SYSTEM, "user": _E2E_USER},
    "e2e/faithful_v1": {"system": _SYSTEM, "user": _E2E_USER + _FAITHFUL_SUFFIX},
    "e2e/strict_v1": {"system": _SYSTEM, "user": _E2E_USER + _E2E_STRICT_SUFFIX},
    "e2e/structured_v1": {
        "system": _SYSTEM,
        "user": _E2E_STRUCTURED,
        "extract_marker": "SUMMARY:",
    },
    "rotowire/plain_v1": {"system": _SYSTEM, "user": _ROTOWIRE_USER},
    "rotowire/faithful_v1": {"system": _SYSTEM, "user": _ROTOWIRE_USER + _FAITHFUL_SUFFIX},
    "rotowire/strict_v1": {"system": _SYSTEM, "user": _ROTOWIRE_USER + _ROTOWIRE_STRICT_SUFFIX},
    "rotowire/structured_v1": {
        "system": _SYSTEM,
        "user": _ROTOWIRE_STRUCTURED,
        "extract_marker": "SUMMARY:",
    },
}


def get_spec(name: str) -> dict[str, str]:
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt {name!r}; available: {sorted(PROMPTS)}")
    return PROMPTS[name]


def get_messages(name: str, table_text: str) -> list[dict[str, str]]:
    spec = get_spec(name)
    return [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": spec["user"].format(table=table_text)},
    ]


def extract_final(text: str, spec: dict[str, str]) -> str:
    """Strip scratch work before the spec's extract_marker, if it has one.

    Matching is tolerant: case-insensitive and allowing markdown decoration
    around the marker (models emit variants like '**Summary:**'). The last
    occurrence wins.
    """
    marker = spec.get("extract_marker")
    if not marker:
        return text.strip()
    pattern = re.compile(
        re.escape(marker.rstrip(":")) + r"[*_\s]*:[*_\s]*", re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    if matches:
        return text[matches[-1].end() :].strip()
    return text.strip()
