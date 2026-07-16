"""Deterministic relational fact checker ("RG-lite").

The week-2 pilot showed the dominant hallucination class on box scores is
relational — wrong winner, false leader claims, misattributed stats — while
digits are copied exactly. NLI cannot compare integers, so relational
claims are checked deterministically: relations are derived from the table
and pattern-extracted claims are verified against them. A lightweight,
model-free variant of the RG metric of Wiseman et al. (2017).

Extraction is precision-first: a clause naming several players, or any
otherwise ambiguous construction, yields no claim rather than a guessed
one. Recall is therefore partial by design; the week-3 manual calibration
measures what it misses.

The derivation core (winner/margin/leaders + alias machinery) lives in
t2t.facts, shared with the fact-sheet serialiser; only the claim-extraction
patterns live here.
"""
from __future__ import annotations

import re

from ..data.types import Table
from ..facts import GameFacts as _GameFacts
from .claims import split_sentences

# category word -> player-section column
CATEGORIES = {
    "point": "PTS",
    "rebound": "REB",
    "board": "REB",
    "assist": "AST",
    "steal": "STL",
    "block": "BLK",
}

_STAT_RE = re.compile(
    r"(\d+)\s+(points?|rebounds?|boards?|assists?|steals?|blocks?)\b", re.IGNORECASE
)
_LED_RE = re.compile(r"\b(led|leads|leading|paced)\b", re.IGNORECASE)
_DEFEAT_RE = re.compile(
    r"\b(defeated|beat|topped|downed|routed|edged|crushed|blew out|cruised past|held off)\b",
    re.IGNORECASE,
)
_LOST_TO_RE = re.compile(r"\b(fell to|lost to)\b", re.IGNORECASE)
_WON_RE = re.compile(r"\b(won|victory|victorious|picked up the win)\b", re.IGNORECASE)
_MARGIN_RE = re.compile(r"(\d+)\s*-\s*point\s+(victory|win|loss|defeat)\b", re.IGNORECASE)
# split clauses at ';', ', while', 'while', ', and' — but never bare ' and ',
# which would cut stat lists like "20 points and 11 rebounds"
_CLAUSE_RE = re.compile(r";|,?\s+while\s+|,\s+and\s+|,\s+(?=[A-Z])")


def _check_winner_claims(sentence: str, facts: _GameFacts) -> list[dict]:
    s_l = sentence.lower()
    teams = facts.find_teams(s_l)
    claims = []

    def claim(claimed_winner: str, kind: str = "winner") -> None:
        if facts.tie:
            return
        claims.append(
            {
                "type": kind,
                "sentence": sentence,
                "claimed": claimed_winner,
                "expected": facts.winner,
                "correct": claimed_winner == facts.winner,
            }
        )

    m = _DEFEAT_RE.search(s_l)
    if m and len(teams) == 2:
        before = [n for p, n in teams if p < m.start()]
        after = [n for p, n in teams if p >= m.end()]
        if len(before) == 1 and len(after) == 1:
            claim(before[0])
            return claims
    m = _LOST_TO_RE.search(s_l)
    if m and len(teams) == 2:
        before = [n for p, n in teams if p < m.start()]
        after = [n for p, n in teams if p >= m.end()]
        if len(before) == 1 and len(after) == 1:
            claim(after[0])
            return claims
    if _WON_RE.search(s_l) and len(teams) == 1 and not _LOST_TO_RE.search(s_l):
        claim(teams[0][1])

    m = _MARGIN_RE.search(sentence)
    if m and facts.margin:
        claims.append(
            {
                "type": "margin",
                "sentence": sentence,
                "claimed": int(m.group(1)),
                "expected": facts.margin,
                "correct": int(m.group(1)) == facts.margin,
            }
        )
    return claims


def _check_player_claims(sentence: str, facts: _GameFacts) -> list[dict]:
    claims = []
    for clause in _CLAUSE_RE.split(sentence):
        c_l = clause.lower()
        players = facts.find_players(c_l)
        if len(players) != 1:
            continue  # precision-first: skip multi-player clauses
        name = players[0]
        info = facts.player_rows[name]
        stats = _STAT_RE.findall(clause)

        # team-label check: "The Nuggets' Karl-Anthony Towns ..."
        teams_here = facts.find_teams(c_l)
        actual_team = facts.team_of_player(name)
        if len(teams_here) == 1 and actual_team:
            claims.append(
                {
                    "type": "team-attribution",
                    "sentence": clause.strip(),
                    "claimed": f"{name} on {teams_here[0][1]}",
                    "expected": f"{name} on {actual_team}",
                    "correct": teams_here[0][1] == actual_team,
                }
            )

        for num, cat_word in stats:
            cat = CATEGORIES[cat_word.lower().rstrip("s")]
            actual = info["stats"].get(cat)
            if not isinstance(actual, (int, float)):
                continue
            claims.append(
                {
                    "type": "attribution",
                    "sentence": clause.strip(),
                    "claimed": f"{name} {num} {cat}",
                    "expected": f"{name} {actual} {cat}",
                    "correct": float(num) == float(actual),
                }
            )

        if stats and _LED_RE.search(c_l):
            cat = CATEGORIES[stats[0][1].lower().rstrip("s")]
            scope = (
                "*"
                if re.search(r"all scorers|the game|game-high", c_l)
                else info["city"].lower()
            )
            leaders = facts.leaders.get((scope, cat))
            if leaders:
                claims.append(
                    {
                        "type": "leader",
                        "sentence": clause.strip(),
                        "claimed": f"{name} led ({cat})",
                        "expected": f"led by {sorted(leaders)} ({cat})",
                        "correct": name in leaders,
                    }
                )
    return claims


def score_text_relations(text: str, table: Table) -> dict:
    facts = _GameFacts(table)
    if not facts.ok:
        return {"n_claims": 0, "n_wrong": 0, "wrong": []}
    claims = []
    for sentence in split_sentences(text):
        claims += _check_winner_claims(sentence, facts)
        claims += _check_player_claims(sentence, facts)
    wrong = [c for c in claims if not c["correct"]]
    return {"n_claims": len(claims), "n_wrong": len(wrong), "wrong": wrong}


def score_run_relations(outputs: dict[str, str], tables: dict[str, Table]) -> dict:
    per_item = {i: score_text_relations(text, tables[i]) for i, text in outputs.items()}
    total = sum(v["n_claims"] for v in per_item.values())
    wrong = sum(v["n_wrong"] for v in per_item.values())
    by_type: dict[str, dict[str, int]] = {}
    for v in per_item.values():
        for c in v["wrong"]:
            t = by_type.setdefault(c["type"], {"wrong": 0})
            t["wrong"] += 1
    return {
        "per_item": per_item,
        "aggregate": {
            "items": len(per_item),
            "relational_claims": total,
            "relational_wrong": wrong,
            "relational_wrong_rate": wrong / total if total else None,
            "pct_items_with_relational_error": (
                sum(1 for v in per_item.values() if v["n_wrong"]) / len(per_item)
                if per_item
                else None
            ),
            "wrong_by_type": by_type,
        },
    }
