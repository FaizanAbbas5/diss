"""Shared deterministic derivations from a box-score Table.

Single source of truth for winner/margin/leader facts, used by both the
relational checker (t2t.eval.relations) and the fact-sheet serialiser
(t2t.factsheet). Sharing one derivation makes the facts the sheet asserts
and the facts the checker expects identical by construction; the shared-bug
risk this creates is mitigated by this module's tests and by the arbiter
metrics (NLI, PARENT) that share no code with it.

FGM/FGA appear in the per-player stats dict for the fact sheet's shooting
lines; the checker never looks them up.
"""
from __future__ import annotations

import re

from .data.types import Section, Table


_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def _surname(name: str) -> str:
    """Last non-suffix token: 'Tim Hardaway Jr.' -> 'hardaway', so suffixed
    players never alias to 'jr.'/'iii' (audit C6)."""
    tokens = [t for t in name.split() if t.lower() not in _NAME_SUFFIXES]
    return (tokens[-1] if tokens else name).lower()


def team_section(table: Table) -> Section | None:
    for sec in table.sections:
        if "team" in sec.name.lower() and len(sec.rows) == 2:
            return sec
    return None


def player_section(table: Table) -> Section | None:
    for sec in table.sections:
        if sec.name.lower() == "players":
            return sec
    return None


class GameFacts:
    """Derived relations plus alias tables for one game."""

    def __init__(self, table: Table):
        self.ok = False
        teams, players = team_section(table), player_section(table)
        if teams is None:
            return
        cols = {c.upper(): i for i, c in enumerate(teams.columns)}
        pts_i = cols.get("TEAM-PTS")
        if pts_i is None:
            return
        pts = [r[pts_i] for r in teams.rows]
        if not all(isinstance(p, (int, float)) for p in pts):
            return
        self.team_names = [str(r[teams.key_column]) for r in teams.rows]
        self.pts = pts
        self.tie = pts[0] == pts[1]
        self.winner = None if self.tie else self.team_names[pts.index(max(pts))]
        self.margin = abs(pts[0] - pts[1])

        # aliases: nickname (last token) always; city only if unambiguous
        self.aliases: dict[str, str] = {}
        cities = [" ".join(n.split()[:-1]).lower() for n in self.team_names]
        for name, city in zip(self.team_names, cities):
            self.aliases[name.lower()] = name
            self.aliases[name.split()[-1].lower()] = name
            if city and cities.count(city) == 1:
                self.aliases[city] = name

        # players: full-name always; surname only if unique in the game
        self.player_rows: dict[str, dict] = {}
        self.player_aliases: dict[str, str] = {}
        if players is not None:
            pcols = {c.upper(): i for i, c in enumerate(players.columns)}
            surnames: dict[str, int] = {}
            for row in players.rows:
                name = str(row[players.key_column])
                stats = {
                    c: row[pcols[c]]
                    for c in ("PTS", "REB", "AST", "STL", "BLK", "FGM", "FGA")
                    if c in pcols
                }
                city = row[pcols["TEAM_CITY"]] if "TEAM_CITY" in pcols else None
                self.player_rows[name] = {"stats": stats, "city": str(city or "")}
                self.player_aliases[name.lower()] = name
                surnames[_surname(name)] = surnames.get(_surname(name), 0) + 1
            for name in list(self.player_rows):
                s = _surname(name)
                if surnames[s] == 1:
                    self.player_aliases[s] = name

        # per-team leaders per category (ties -> set of names)
        self.leaders: dict[tuple[str, str], set[str]] = {}
        by_city: dict[str, list[str]] = {}
        for name, info in self.player_rows.items():
            by_city.setdefault(info["city"].lower(), []).append(name)
        scopes = {**{c: ns for c, ns in by_city.items()}, "*": list(self.player_rows)}
        for scope, names in scopes.items():
            for cat in ("PTS", "REB", "AST"):
                vals = [
                    (n, self.player_rows[n]["stats"].get(cat))
                    for n in names
                    if isinstance(self.player_rows[n]["stats"].get(cat), (int, float))
                ]
                if vals:
                    best = max(v for _, v in vals)
                    self.leaders[(scope, cat)] = {n for n, v in vals if v == best}
        self.ok = True

    def team_of_player(self, name: str) -> str | None:
        city = self.player_rows[name]["city"].lower()
        matches = [t for t in self.team_names if city and city in t.lower()]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _find_alias(alias: str, text_l: str) -> int:
        """Word-boundary alias search ('lin' must not match 'line');
        lookarounds rather than \\b so aliases with periods ('j.j.') work."""
        m = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_l)
        return m.start() if m else -1

    def find_teams(self, text_l: str) -> list[tuple[int, str]]:
        found = {}
        for alias, name in self.aliases.items():
            pos = self._find_alias(alias, text_l)
            if pos >= 0 and (name not in found or pos < found[name]):
                found[name] = pos
        return sorted((p, n) for n, p in found.items())

    def find_players(self, text_l: str) -> list[str]:
        return [n for _p, n in self.find_players_with_pos(text_l)]

    def find_players_with_pos(self, text_l: str) -> list[tuple[int, str]]:
        """(position, name) of each mentioned player's first mention;
        the positional analogue of find_teams."""
        first: dict[str, int] = {}
        for pos, name in self.find_player_mentions(text_l):
            if name not in first:
                first[name] = pos
        return sorted((p, n) for n, p in first.items())

    def find_player_mentions(self, text_l: str) -> list[tuple[int, str]]:
        """Every occurrence of every player alias, in text order; overlapping
        matches resolved in favour of the longest (full name beats the
        surname inside it)."""
        matches: list[tuple[int, int, str]] = []
        for alias, name in self.player_aliases.items():
            for m in re.finditer(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_l
            ):
                matches.append((m.start(), m.end(), name))
        matches.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        kept: list[tuple[int, int, str]] = []
        for s, e, n in matches:
            if all(e <= ks or s >= ke for ks, ke, _n in kept):
                kept.append((s, e, n))
        return [(s, n) for s, _e, n in sorted(kept)]
