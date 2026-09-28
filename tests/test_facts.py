"""Property tests for the shared derivation core (t2t.facts).

These carry extra weight: GameFacts feeds both the fact-sheet serialiser
and the relational checker, so a bug here would corrupt the sheet and then
score the model's faithful copy of it as correct. Behaviour is
additionally pinned by tests/test_relations.py, which exercises the same
class via the checker.
"""
import random

from t2t.data.types import Section, Table
from t2t.facts import GameFacts, player_section, team_section

TEAM_COLS = ["TEAM", "TEAM-PTS", "TEAM-WINS", "TEAM-LOSSES"]
PLAYER_COLS = ["PLAYER_NAME", "TEAM_CITY", "PTS", "REB", "AST", "STL", "BLK", "FGM", "FGA"]


def game(team_rows=None, player_rows=None):
    sections = [
        Section(
            "teams",
            TEAM_COLS,
            team_rows
            if team_rows is not None
            else [["Toronto Raptors", 122, 11, 6], ["Philadelphia 76ers", 95, 4, 14]],
        )
    ]
    if player_rows is not None:
        sections.append(Section("players", PLAYER_COLS, player_rows))
    return Table(sections)


PLAYERS = [
    ["Kyle Lowry", "Toronto", 24, 4, 8, 2, 0, 8, 14],
    ["DeMar DeRozan", "Toronto", 14, 5, 3, 1, 0, 6, 15],
    ["Robert Covington", "Philadelphia", 20, 6, 1, 3, 1, 7, 13],
    ["Jahlil Okafor", "Philadelphia", 15, 8, 1, 0, 1, 7, 12],
    ["Ersan Ilyasova", "Philadelphia", 15, 7, 2, 1, 0, 6, 11],
]


def test_winner_is_argmax_and_margin_is_abs_diff_for_all_orderings():
    for a, b in [(122, 95), (95, 122), (100, 99), (99, 100), (1, 150)]:
        f = GameFacts(game(team_rows=[["A Ants", a, 1, 1], ["B Bees", b, 1, 1]]))
        assert f.ok and not f.tie
        assert f.winner == ("A Ants" if a > b else "B Bees")
        assert f.margin == abs(a - b) > 0


def test_tie_has_no_winner_and_zero_margin():
    f = GameFacts(game(team_rows=[["A Ants", 100, 1, 1], ["B Bees", 100, 1, 1]]))
    assert f.ok and f.tie and f.winner is None and f.margin == 0


def test_missing_team_pts_column_not_ok():
    t = Table([Section("teams", ["TEAM", "TEAM-REB"], [["A Ants", 40], ["B Bees", 38]])])
    assert GameFacts(t).ok is False


def test_non_numeric_team_pts_not_ok():
    f = GameFacts(game(team_rows=[["A Ants", None, 1, 1], ["B Bees", 95, 1, 1]]))
    assert f.ok is False


def test_no_teams_section_not_ok():
    t = Table([Section("restaurant", ["name"], [["The Vaults"]])])
    assert GameFacts(t).ok is False
    assert team_section(t) is None and player_section(t) is None


def test_leaders_exist_for_every_category_with_values():
    f = GameFacts(game(player_rows=PLAYERS))
    for scope in ("toronto", "philadelphia", "*"):
        for cat in ("PTS", "REB", "AST"):
            assert (scope, cat) in f.leaders, (scope, cat)
    assert f.leaders[("toronto", "PTS")] == {"Kyle Lowry"}
    assert f.leaders[("philadelphia", "REB")] == {"Jahlil Okafor"}
    assert f.leaders[("*", "PTS")] == {"Kyle Lowry"}


def test_leader_ties_return_all_tied_players():
    f = GameFacts(game(player_rows=PLAYERS))
    assert f.leaders[("philadelphia", "PTS")] == {"Robert Covington"}
    # Okafor and Ilyasova tie on 15 only after removing Covington
    rows = [p for p in PLAYERS if p[0] != "Robert Covington"]
    f2 = GameFacts(game(player_rows=rows))
    assert f2.leaders[("philadelphia", "PTS")] == {"Jahlil Okafor", "Ersan Ilyasova"}


def test_all_none_category_yields_no_leader_entry():
    rows = [[n, c, pts, None, ast, s, b, m, a] for n, c, pts, _, ast, s, b, m, a in PLAYERS]
    f = GameFacts(game(player_rows=rows))
    assert not any(cat == "REB" for _, cat in f.leaders)
    assert ("toronto", "PTS") in f.leaders


def test_none_values_skipped_not_treated_as_zero():
    rows = [
        ["A One", "Toronto", None, 5, 1, 0, 0, 1, 2],
        ["B Two", "Toronto", 0, 3, 1, 0, 0, 0, 1],
    ]
    f = GameFacts(game(player_rows=rows))
    assert f.leaders[("toronto", "PTS")] == {"B Two"}
    assert f.leaders[("toronto", "REB")] == {"A One"}


def test_derivations_invariant_under_player_row_order():
    base = GameFacts(game(player_rows=PLAYERS))
    rows = list(PLAYERS)
    rng = random.Random(13)
    for _ in range(5):
        rng.shuffle(rows)
        f = GameFacts(game(player_rows=rows))
        assert f.leaders == base.leaders
        assert f.player_rows == base.player_rows
        assert (f.winner, f.margin, f.tie) == (base.winner, base.margin, base.tie)


def test_deterministic_across_instances():
    a, b = GameFacts(game(player_rows=PLAYERS)), GameFacts(game(player_rows=PLAYERS))
    assert a.leaders == b.leaders and a.aliases == b.aliases


def test_team_of_player_and_shared_city_ambiguity():
    f = GameFacts(game(player_rows=PLAYERS))
    assert f.team_of_player("Kyle Lowry") == "Toronto Raptors"
    assert f.team_of_player("Jahlil Okafor") == "Philadelphia 76ers"
    shared = GameFacts(
        Table(
            [
                Section(
                    "teams",
                    TEAM_COLS,
                    [["Los Angeles Lakers", 110, 1, 1], ["Los Angeles Clippers", 105, 1, 1]],
                ),
                Section("players", PLAYER_COLS, [["Some Guy", "Los Angeles", 10, 2, 3, 0, 0, 4, 8]]),
            ]
        )
    )
    assert shared.ok and shared.team_of_player("Some Guy") is None
    assert "los angeles" not in shared.aliases  # ambiguous city alias dropped


def test_stats_include_fgm_fga_for_the_sheet():
    f = GameFacts(game(player_rows=PLAYERS))
    assert f.player_rows["Kyle Lowry"]["stats"]["FGM"] == 8
    assert f.player_rows["Kyle Lowry"]["stats"]["FGA"] == 14


def test_suffixed_names_do_not_alias_to_suffix():
    from t2t.data.types import Section, Table
    from t2t.facts import GameFacts
    t = Table([
        Section("teams", ["TEAM", "TEAM-PTS"], [["A Aces", 100], ["B Bears", 90]]),
        Section("players", ["PLAYER_NAME", "TEAM_CITY", "PTS"],
                [["Tim Hardaway Jr.", "A", 20], ["Robinson III", "B", 10]]),
    ])
    f = GameFacts(t)
    assert "jr." not in f.player_aliases and "iii" not in f.player_aliases
    assert f.player_aliases.get("hardaway") == "Tim Hardaway Jr."
