"""Deterministic relational fact checker ("RG-lite").

The dominant hallucination class on box scores is relational (wrong
winner, false leader claims, misattributed stats) while digits are copied
exactly. NLI cannot compare integers, so relational claims are checked
deterministically: relations are derived from the table and
pattern-extracted claims are verified against them. A lightweight,
model-free variant of the RG metric of Wiseman et al. (2017).

Extraction is precision-first: ambiguous constructions yield no claim
rather than a guessed one. A clause naming several players is segmented at
the player mentions (each stat pattern is attributed to the player whose
span it sits in) unless it pools stats across players ("combined for 45"),
which stays excluded. Recall is partial by design.

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
# Passive inversion: "the 76ers were defeated by the Raptors"; the team
# before the verb is the loser (audit C1; common in gold text).
_PASSIVE_DEFEAT_RE = re.compile(
    r"\b(?:was|were)\s+(?:defeated|beaten|downed|routed|edged|topped)\s+by\b",
    re.IGNORECASE,
)
_LOST_TO_RE = re.compile(r"\b(fell to|lost to)\b", re.IGNORECASE)
_WON_RE = re.compile(r"\b(won|victory|victorious|picked up the win)\b", re.IGNORECASE)
# season-record phrasing, not a game-result claim: "have won 11 of 13"
_SEASON_WON_RE = re.compile(
    r"\b(?:have|has|had)\s+(?:now\s+)?won\b|\bwon\s+\d+\s+of\b"
    r"|\bwin(?:ning)?\s+streak\b|\bwon\s+\d+\s+(?:straight|consecutive|in a row)\b",
    re.IGNORECASE,
)
_MARGIN_RE = re.compile(r"(\d+)\s*-\s*point\s+(victory|win|loss|defeat)\b", re.IGNORECASE)
# Split clauses at ';', ', while', 'while', ', and', 'but', never at bare
# ' and ', which would cut stat lists like "20 points and 11 rebounds".
# 'but' starts a contrast clause whose team mentions are usually the
# opponent ("... but they couldn't contain the Clippers"); keeping it in
# the player's clause caused team-attribution false positives.
_CLAUSE_RE = re.compile(r";|,?\s+while\s+|,?\s+but\s+|,\s+and\s+|,\s+(?=[A-Z])")


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

    m = _PASSIVE_DEFEAT_RE.search(s_l)
    if m and len(teams) == 2:
        before = [n for p, n in teams if p < m.start()]
        after = [n for p, n in teams if p >= m.end()]
        if len(before) == 1 and len(after) == 1:
            claim(after[0])  # passive: the team after "defeated by" won
            return claims
    m = _DEFEAT_RE.search(s_l)
    if m and len(teams) == 2 and not _PASSIVE_DEFEAT_RE.search(s_l):
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
    if (
        _WON_RE.search(s_l)
        and len(teams) == 1
        and not _LOST_TO_RE.search(s_l)
        and not _SEASON_WON_RE.search(s_l)  # "have won 11 of 13" (audit C4)
    ):
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


# A multi-player clause is segmented at player positions unless it pools
# stats across players; then attribution is genuinely ambiguous and the
# clause is skipped (precision-first).
_POOLED_RE = re.compile(r"\b(combined|between them|total|altogether)\b", re.IGNORECASE)


# Stat written before its owner: "... and 12 points from Kevin Love".
_STAT_BEFORE_OWNER_RE = re.compile(
    r"(?:a\s+)?\d+\s+(?:points?|rebounds?|boards?|assists?|steals?|blocks?)"
    r"\s+(?:from|by|for)\s*$",
    re.IGNORECASE,
)


def _segment_start(clause_l: str, pos: int, facts: _GameFacts) -> int:
    """Material right before a player mention that belongs to that player,
    a possessive team reference ("... and the Raptors' Robert Covington")
    or a stat-before-owner construction ("... and 12 points from Kevin
    Love"), pulls the boundary back so it lands in the right span."""
    head = clause_l[:pos]
    best = pos
    for alias in facts.aliases:
        m = re.search(
            r"(?:the\s+)?" + re.escape(alias) + r"[‘’']s?\s*$", head
        )
        if m and m.start() < best:
            best = m.start()
    m = _STAT_BEFORE_OWNER_RE.search(head[:best] if best < pos else head)
    if m and m.start() < best:
        best = m.start()
    return best


def _player_segments(clause: str, facts: _GameFacts) -> list[tuple[str, str]]:
    """(player, sub-span) pairs. One distinct player: the whole clause.
    Several: the clause split at every player mention (repeat mentions
    included), so each stat pattern sits in the span of the player it
    belongs to."""
    c_l = clause.lower()
    mentions = facts.find_player_mentions(c_l)
    names = {n for _p, n in mentions}
    if len(names) == 1:
        return [(mentions[0][1], clause)]
    if not mentions or _POOLED_RE.search(c_l):
        return []
    starts = [_segment_start(c_l, p, facts) for p, _n in mentions]
    ends = starts[1:] + [len(clause)]
    return [
        (n, clause[s:e]) for (_p, n), s, e in zip(mentions, starts, ends)
    ]


def _team_stat_subject(pre_l: str, facts: _GameFacts) -> bool:
    """True when the text immediately before a stat pattern makes a team the
    stat's subject ("the Raptors put up <122 points>"); the number belongs
    to the team, not the segment's player. "led the Raptors with 24" keeps
    the player as subject ("with" intervenes) and is not matched."""
    for alias in facts.aliases:
        if re.search(
            re.escape(alias)
            + r"\s+(?:put up|posted|totaled|totalled|managed|shot|scored|added|grabbed|had)\s*$",
            pre_l,
        ):
            return True
    return False


def _linked_team(segment_l: str, facts: _GameFacts) -> str | None:
    """The team a membership-asserting pattern links to this segment's
    player: possessive before the name ("the Raptors' X"), "for/of the T",
    "led/paced the T", "the T was/were led/paced by", or "T guard X".
    Returns None when no pattern matches or two teams match (ambiguous)."""
    linked: set[str] = set()
    for alias, team in facts.aliases.items():
        a = re.escape(alias)
        if re.search(
            r"^\W*(?:the\s+)?" + a + r"[‘’']s?\s"          # possessive lead-in
            r"|\b(?:for|of)\s+the\s+" + a + r"\b"           # "for/of the T"
            r"|\b(?:led|leads|leading|paced)\s+the\s+" + a + r"\b"
            r"|\b" + a + r"\s+(?:was|were)\s+(?:led|paced)\b"
            r"|\b" + a + r"\s+(?:guard|forward|center|star)\b",
            segment_l,
        ):
            linked.add(team)
    return linked.pop() if len(linked) == 1 else None


def _check_player_claims(sentence: str, facts: _GameFacts) -> list[dict]:
    claims = []
    for full_clause in _CLAUSE_RE.split(sentence):
        for name, clause in _player_segments(full_clause, facts):
            c_l = clause.lower()
            info = facts.player_rows[name]
            stats = _STAT_RE.findall(clause)

            # Team-label check: only when the team is linked to the player
            # by a membership pattern; a bare team mention in the same
            # clause is often the opponent ("Despite X's effort, the Suns'
            # attack ...") and must not count as a membership claim.
            claimed_team = _linked_team(c_l, facts)
            actual_team = facts.team_of_player(name)
            if claimed_team and actual_team:
                claims.append(
                    {
                        "type": "team-attribution",
                        "sentence": clause.strip(),
                        "claimed": f"{name} on {claimed_team}",
                        "expected": f"{name} on {actual_team}",
                        "correct": claimed_team == actual_team,
                    }
                )

            for m in _STAT_RE.finditer(clause):
                num, cat_word = m.group(1), m.group(2)
                # team-total guard (audit C3): "<Team> put up 122 points"
                # inside a player's span is the TEAM's stat, not the player's
                if _team_stat_subject(c_l[: m.start()], facts):
                    continue
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

            # leader-scope guard (audit C5): "led the bench" is not a claim
            # about the full-team leader
            if (
                stats
                and _LED_RE.search(c_l)
                and not re.search(r"\b(?:bench|second unit|reserves)\b", c_l)
            ):
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
