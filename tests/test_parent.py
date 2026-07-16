from t2t.data.types import Section, Table
from t2t.eval.parent_metric import compute_parent, table_to_parent_format

TABLE = Table(
    [Section("restaurant", ["name", "food"], [["The Vaults", "Italian"]])]
)


def test_table_format_single_row_uses_plain_attributes():
    pairs = table_to_parent_format(TABLE)
    assert (["name"], ["the", "vaults"]) in pairs
    assert (["food"], ["italian"]) in pairs


def test_table_format_multi_row_prefixes_entity():
    t = Table(
        [Section("players", ["PLAYER_NAME", "PTS"], [["LeBron James", 27], ["Kevin Love", 14]])]
    )
    pairs = table_to_parent_format(t)
    assert (["lebron", "james", "pts"], ["27"]) in pairs


def test_faithful_beats_hallucinated():
    faithful = compute_parent(
        ["The Vaults serves Italian food."],
        [["The Vaults is an Italian restaurant."]],
        [TABLE],
    )
    hallucinated = compute_parent(
        ["A cheap family-friendly pub near the river."],
        [["The Vaults is an Italian restaurant."]],
        [TABLE],
    )
    assert faithful["parent_f1"] > hallucinated["parent_f1"]
    assert len(faithful["per_item_f1"]) == 1
