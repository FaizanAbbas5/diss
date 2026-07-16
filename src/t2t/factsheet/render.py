"""Deterministic Table -> fact-sheet serialisation (template arm).

The selection + derivation policy below IS the engineered representation
under test (docs/factsheet-arm-design.md). No LLM is involved anywhere in
sheet construction: every number is copied from a table cell or derived by
the shared t2t.facts core (margin), so the sheet cannot contain
fabrications — pinned by the invariant test in tests/test_factsheet.py.

Policy is versioned: any change to selection rules, thresholds, ordering,
or wording bumps VERSION, and configs carry it as `factsheet_version` so
the config hash separates old runs from new.

Dispatch is structural: a table on which GameFacts derives (teams section
with numeric TEAM-PTS) renders as a box-score sheet (rotowire.j2); anything
else — E2E, and pathological box scores — renders as labelled
attribute/value lines (e2e.j2), a near-null transformation.

Sheet labels must not contain digits ("three-point %", never "3PT%"): a
digit in a label would be flagged by the number checker the sheet is
required to pass by construction.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..data.types import Section, Table
from ..facts import GameFacts, team_section

VERSION = "v1"

# Notability thresholds: 20+ points, or a 10+ line in rebounds/assists.
NOTABLE_PTS = 20
NOTABLE_REB = 10
NOTABLE_AST = 10
# Categories counting towards double-/triple-doubles (10+ each).
DOUBLE_CATS = ("PTS", "REB", "AST", "STL", "BLK")
N_SECONDARY = 2  # "next two scorers" per team

# teams-section column -> sheet label, in fixed sheet order.
TEAM_STATS = [
    ("TEAM-PTS", "points"),
    ("TEAM-FG_PCT", "field-goal %"),
    ("TEAM-FG3_PCT", "three-point %"),
    ("TEAM-FT_PCT", "free-throw %"),
    ("TEAM-REB", "rebounds"),
    ("TEAM-AST", "assists"),
    ("TEAM-TOV", "turnovers"),
]

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _num(x) -> bool:
    return isinstance(x, (int, float))


def _count(n, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _shooting(stats: dict) -> str | None:
    if _num(stats.get("FGM")) and _num(stats.get("FGA")):
        return f"{stats['FGM']}-of-{stats['FGA']} shooting"
    return None


def _scorer_phrase(names: list[str], facts: GameFacts) -> str:
    """'A with 24 points on 8-of-14 shooting'; ties list every player
    (shooting in parentheses) and append 'each'."""
    pts = facts.player_rows[names[0]]["stats"]["PTS"]
    if len(names) == 1:
        shot = _shooting(facts.player_rows[names[0]]["stats"])
        return f"{names[0]} with {_count(pts, 'point')}" + (f" on {shot}" if shot else "")
    with_shots = []
    for n in names:
        shot = _shooting(facts.player_rows[n]["stats"])
        with_shots.append(f"{n} ({shot})" if shot else n)
    return f"{_join(with_shots)} with {_count(pts, 'point')} each"


def _leader_lines(team: str, roster: list[str], facts: GameFacts) -> list[str]:
    scope = facts.player_rows[roster[0]]["city"].lower()
    lines = []
    scorers = facts.leaders.get((scope, "PTS"))
    if scorers:
        lines.append(f"{team} top scorer: {_scorer_phrase(sorted(scorers), facts)}.")
    for cat, label, word in (("REB", "top rebounder", "rebound"), ("AST", "top assists", "assist")):
        leaders = facts.leaders.get((scope, cat))
        if leaders:
            names = sorted(leaders)
            val = facts.player_rows[names[0]]["stats"][cat]
            each = " each" if len(names) > 1 else ""
            lines.append(f"{team} {label}: {_join(names)} with {_count(val, word)}{each}.")
    return lines


def _secondary_line(team: str, roster: list[str], facts: GameFacts) -> str | None:
    """Next N_SECONDARY scorers after the top-scorer set (PTS desc, name asc)."""
    scope = facts.player_rows[roster[0]]["city"].lower()
    top = facts.leaders.get((scope, "PTS"), set())
    rest = sorted(
        ((facts.player_rows[n]["stats"]["PTS"], n) for n in roster
         if n not in top and _num(facts.player_rows[n]["stats"].get("PTS"))),
        key=lambda t: (-t[0], t[1]),
    )[:N_SECONDARY]
    if not rest:
        return None
    return f"{team}: " + "; ".join(f"{n} {_count(p, 'point')}" for p, n in rest) + "."


def _notable_sentences(facts: GameFacts) -> list[str]:
    keyed = []
    for name, info in facts.player_rows.items():
        st = info["stats"]
        pts, reb, ast = st.get("PTS"), st.get("REB"), st.get("AST")
        if not (
            (_num(pts) and pts >= NOTABLE_PTS)
            or (_num(reb) and reb >= NOTABLE_REB)
            or (_num(ast) and ast >= NOTABLE_AST)
        ):
            continue
        parts = [
            _count(st[cat], word)
            for cat, word in (("PTS", "point"), ("REB", "rebound"), ("AST", "assist"))
            if _num(st.get(cat))
        ]
        # steals/blocks only when they cross the double-double threshold
        parts += [
            _count(st[cat], word)
            for cat, word in (("STL", "steal"), ("BLK", "block"))
            if _num(st.get(cat)) and st[cat] >= 10
        ]
        doubles = sum(1 for c in DOUBLE_CATS if _num(st.get(c)) and st[c] >= 10)
        badge = "triple-double" if doubles >= 3 else "double-double" if doubles == 2 else None
        sentence = f"{name} had {_join(parts)}" + (f", a {badge}" if badge else "") + "."
        keyed.append(((-(pts if _num(pts) else -1), name), sentence))
    return [s for _, s in sorted(keyed)]


def _boxscore_context(table: Table, facts: GameFacts) -> dict:
    teams = team_section(table)
    cols = {c.upper(): i for i, c in enumerate(teams.columns)}

    if facts.tie:
        result = (
            f"The {facts.team_names[0]} and the {facts.team_names[1]} finished tied"
            f" at {facts.pts[0]} points apiece."
        )
    else:
        loser = next(n for n in facts.team_names if n != facts.winner)
        result = (
            f"The {facts.winner} defeated the {loser} {max(facts.pts)} to"
            f" {min(facts.pts)} (margin {facts.margin})."
        )

    records = []
    for row in teams.rows:
        wins = row[cols["TEAM-WINS"]] if "TEAM-WINS" in cols else None
        losses = row[cols["TEAM-LOSSES"]] if "TEAM-LOSSES" in cols else None
        if _num(wins) and _num(losses):
            records.append(f"The {row[teams.key_column]} are now {wins}-{losses}.")

    team_lines = []
    for row in teams.rows:
        stats = "; ".join(
            f"{label}: {row[cols[col]]}"
            for col, label in TEAM_STATS
            if col in cols and row[cols[col]] is not None
        )
        if stats:
            team_lines.append(f"{row[teams.key_column]} -- {stats}")

    # per-team blocks only where players attribute to a unique team
    # (precision-first, same stance as the checker: shared-city games skip)
    by_team: dict[str, list[str]] = {}
    for name in facts.player_rows:
        team = facts.team_of_player(name)
        if team:
            by_team.setdefault(team, []).append(name)
    leader_lines, secondary_lines = [], []
    for team in facts.team_names:
        roster = by_team.get(team)
        if roster:
            leader_lines += _leader_lines(team, roster, facts)
            line = _secondary_line(team, roster, facts)
            if line:
                secondary_lines.append(line)

    return {
        "result": result,
        "records": records,
        "team_lines": team_lines,
        "leader_lines": leader_lines,
        "notables": _notable_sentences(facts),
        "secondary_lines": secondary_lines,
    }


def _generic_context(table: Table) -> dict:
    """Labelled attribute/value lines, nothing selected or derived."""
    show_headers = len(table.sections) > 1
    sections = []
    for sec in table.sections:
        lines: list[str] = []
        if len(sec.rows) == 1:
            lines += [
                f"{col}: {val}"
                for col, val in zip(sec.columns, sec.rows[0])
                if val is not None
            ]
        else:
            for row in sec.rows:
                facts = "; ".join(
                    f"{col}: {val}"
                    for i, (col, val) in enumerate(zip(sec.columns, row))
                    if val is not None and i != sec.key_column
                )
                lines.append(f"{row[sec.key_column]} -- {facts}")
        sections.append({"header": f"{sec.name}:" if show_headers else None, "lines": lines})
    return {"sections": sections}


def to_factsheet(table: Table) -> str:
    facts = GameFacts(table)
    if facts.ok:
        return _env.get_template("rotowire.j2").render(**_boxscore_context(table, facts)).strip()
    return _env.get_template("e2e.j2").render(**_generic_context(table)).strip()
