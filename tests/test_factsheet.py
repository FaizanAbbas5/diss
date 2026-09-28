"""Renderer goldens + invariants for the fact-sheet arm.

The golden tests pin exact rendering (the sheet is a versioned experimental
manipulation; accidental drift must fail loudly). The invariant test is
the arm's core guarantee: every number in a rendered sheet passes the
tiered number check, i.e. nothing unsupported can enter the sheet.
"""
import random

import pytest

from t2t.data.types import Section, Table
from t2t.eval.numbers import score_text
from t2t.factsheet import to_factsheet
from t2t.serialise import get_serialiser, to_markdown

TEAM_COLS = [
    "TEAM", "TEAM-PTS", "TEAM-WINS", "TEAM-LOSSES", "TEAM-FG_PCT",
    "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB", "TEAM-AST", "TEAM-TOV",
]
PLAYER_COLS = ["PLAYER_NAME", "TEAM_CITY", "PTS", "REB", "AST", "STL", "BLK", "FGM", "FGA"]

TEAM_ROWS = [
    ["Toronto Raptors", 122, 11, 6, 49, 38, 82, 42, 22, 12],
    ["Philadelphia 76ers", 95, 4, 14, 42, 30, 75, 38, 18, 16],
]
PLAYER_ROWS = [
    ["Kyle Lowry", "Toronto", 24, 4, 8, 2, 0, 8, 14],
    ["DeMar DeRozan", "Toronto", 14, 5, 3, 1, 0, 6, 15],
    ["Norman Powell", "Toronto", 12, 3, 2, 0, 0, 5, 9],
    ["Jonas Valanciunas", "Toronto", 10, 12, 1, 0, 2, 5, 8],
    ["Robert Covington", "Philadelphia", 20, 6, 1, 3, 1, 7, 13],
    ["Jahlil Okafor", "Philadelphia", 15, 8, 1, 0, 1, 7, 12],
    ["Ersan Ilyasova", "Philadelphia", 15, 7, 2, 1, 0, 6, 11],
]


def game(team_rows=TEAM_ROWS, player_rows=PLAYER_ROWS):
    return Table(
        [
            Section("teams", TEAM_COLS, [list(r) for r in team_rows]),
            Section("players", PLAYER_COLS, [list(r) for r in player_rows]),
        ]
    )


GOLDEN = """The Toronto Raptors defeated the Philadelphia 76ers 122 to 95 (margin 27).
The Toronto Raptors are now 11-6.
The Philadelphia 76ers are now 4-14.

Team totals:
Toronto Raptors -- points: 122; field-goal %: 49; three-point %: 38; free-throw %: 82; rebounds: 42; assists: 22; turnovers: 12
Philadelphia 76ers -- points: 95; field-goal %: 42; three-point %: 30; free-throw %: 75; rebounds: 38; assists: 18; turnovers: 16

Team leaders:
Toronto Raptors top scorer: Kyle Lowry with 24 points on 8-of-14 shooting.
Toronto Raptors top rebounder: Jonas Valanciunas with 12 rebounds.
Toronto Raptors top assists: Kyle Lowry with 8 assists.
Philadelphia 76ers top scorer: Robert Covington with 20 points on 7-of-13 shooting.
Philadelphia 76ers top rebounder: Jahlil Okafor with 8 rebounds.
Philadelphia 76ers top assists: Ersan Ilyasova with 2 assists.

Notable performances:
Kyle Lowry had 24 points, 4 rebounds and 8 assists.
Robert Covington had 20 points, 6 rebounds and 1 assist.
Jonas Valanciunas had 10 points, 12 rebounds and 1 assist, a double-double.

Other scorers:
Toronto Raptors: DeMar DeRozan 14 points; Norman Powell 12 points.
Philadelphia 76ers: Ersan Ilyasova 15 points; Jahlil Okafor 15 points."""


def test_rotowire_golden():
    assert to_factsheet(game()) == GOLDEN


def test_e2e_golden():
    t = Table(
        [
            Section(
                "restaurant",
                ["name", "eatType", "food", "priceRange", "familyFriendly"],
                [["The Vaults", "pub", "English", "more than £30", "yes"]],
            )
        ]
    )
    assert to_factsheet(t) == (
        "name: The Vaults\n"
        "eatType: pub\n"
        "food: English\n"
        "priceRange: more than £30\n"
        "familyFriendly: yes"
    )


def test_e2e_none_values_omitted():
    t = Table([Section("restaurant", ["name", "food"], [["The Vaults", None]])])
    assert to_factsheet(t) == "name: The Vaults"


def test_tie_sentence_no_winner_no_margin():
    rows = [list(TEAM_ROWS[0]), list(TEAM_ROWS[1])]
    rows[0][1] = rows[1][1] = 100
    sheet = to_factsheet(game(team_rows=rows))
    assert "finished tied at 100 points apiece" in sheet.splitlines()[0]
    assert "defeated" not in sheet and "margin" not in sheet


def test_every_sheet_number_is_table_supported():
    """The arm's construction guarantee, checked with the eval-suite checker
    itself: nothing unsupported can enter the sheet."""
    tie_rows = [list(TEAM_ROWS[0]), list(TEAM_ROWS[1])]
    tie_rows[0][1] = tie_rows[1][1] = 100
    none_rows = [r[:3] + [None] + r[4:] for r in PLAYER_ROWS]  # REB all None
    for table in (
        game(),
        game(team_rows=tie_rows),
        game(player_rows=none_rows),
        game(player_rows=[]),
    ):
        result = score_text(to_factsheet(table), table)
        assert result["unsupported"] == []
        assert result["support_rate"] == 1.0


def test_sheet_invariant_under_player_row_order():
    base = to_factsheet(game())
    rows = [list(r) for r in PLAYER_ROWS]
    rng = random.Random(7)
    for _ in range(5):
        rng.shuffle(rows)
        assert to_factsheet(game(player_rows=rows)) == base


def test_deterministic():
    assert to_factsheet(game()) == to_factsheet(game())


def test_shared_city_skips_per_team_blocks_but_keeps_notables():
    rows = [list(TEAM_ROWS[0]), list(TEAM_ROWS[1])]
    rows[0][0], rows[1][0] = "Los Angeles Lakers", "Los Angeles Clippers"
    players = [[n, "Los Angeles", *rest] for n, _, *rest in PLAYER_ROWS]
    sheet = to_factsheet(game(team_rows=rows, player_rows=players))
    assert "Team leaders:" not in sheet and "Other scorers:" not in sheet
    assert "Kyle Lowry had 24 points" in sheet  # notables need no team attribution


def test_missing_team_pts_falls_back_to_labelled_lines():
    t = Table(
        [
            Section("teams", ["TEAM", "TEAM-REB"], [["A Ants", 40], ["B Bees", 38]]),
            Section("players", PLAYER_COLS, [list(r) for r in PLAYER_ROWS[:1]]),
        ]
    )
    sheet = to_factsheet(t)
    assert "defeated" not in sheet
    assert "teams:" in sheet and "players:" in sheet
    assert "A Ants -- TEAM-REB: 40" in sheet
    assert to_factsheet(t) == sheet  # still deterministic


def test_registry_dispatch():
    assert get_serialiser() is to_markdown
    assert get_serialiser("markdown") is to_markdown
    assert get_serialiser("factsheet")(game()) == to_factsheet(game())
    with pytest.raises(KeyError):
        get_serialiser("csv")
