from t2t.data.types import Section, Table
from t2t.eval.numbers import score_text

GAME = Table(
    [
        Section("teams", ["TEAM", "TEAM-PTS"], [["Heat", 100], ["Bulls", 88]]),
        Section(
            "players",
            ["PLAYER_NAME", "PTS"],
            [["LeBron James", 27], ["Kevin Love", 14]],
        ),
    ]
)

RESTAURANT = Table(
    [Section("restaurant", ["name", "customer rating"], [["The Vaults", "5 out of 5"]])]
)


def test_copied_number_supported():
    s = score_text("LeBron James scored 27 points.", GAME)
    assert s["unsupported"] == [] and s["support_rate"] == 1.0


def test_fabricated_number_flagged():
    s = score_text("LeBron James scored 29 points.", GAME)
    assert s["unsupported"] == [29.0]


def test_margin_derivation_supported():
    s = score_text("The Heat won 100 - 88, a 12-point victory.", GAME)
    assert s["unsupported"] == [] and s["n_numbers"] == 3
    assert s["n_exact"] == 2 and s["n_derived"] == 1  # 12 only via derivation


def test_combined_sum_supported():
    s = score_text("They combined for 41 points.", GAME)
    assert s["unsupported"] == []


def test_wrong_sum_flagged():
    s = score_text("They combined for 42 points.", GAME)
    assert s["unsupported"] == [42.0]


def test_number_inside_string_cell_supported():
    s = score_text("The Vaults is rated 5 out of 5.", RESTAURANT)
    assert s["unsupported"] == []


def test_spelled_out_numbers_ignored():
    s = score_text("One of the best pubs around.", GAME)
    assert s["n_numbers"] == 0 and s["support_rate"] is None


def test_derivations_can_be_disabled():
    s = score_text("a 12-point victory", GAME, allow_derived=False)
    assert s["unsupported"] == [12.0]
