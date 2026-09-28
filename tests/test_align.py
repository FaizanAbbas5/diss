"""Alignment labels and reference filtering on crafted references."""
from t2t.data.types import Section, Table
from t2t.encode.align import alignment_labels, filter_reference

E2E_TABLE = Table(
    [
        Section(
            "restaurant",
            ["name", "area", "priceRange", "customer rating", "food"],
            [["The Vaults", "riverside", "£20-25", "5 out of 5", "Fast food"]],
        )
    ]
)


def test_e2e_string_cells_word_boundary_match():
    labels = alignment_labels(E2E_TABLE, "The Vaults is in the riverside area.")
    assert labels[0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_no_substring_false_positive():
    # 'riverside' in the table must not match the mere word 'river'
    table = Table([Section("restaurant", ["area"], [["river"]])])
    assert alignment_labels(table, "It sits by the riverside.")[0, 0] == 0.0


def test_multiword_and_punctuated_values_match():
    labels = alignment_labels(
        E2E_TABLE, "A Fast food place, rated 5 out of 5, costing £20-25."
    )
    assert labels[0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_matching_is_case_insensitive():
    assert alignment_labels(E2E_TABLE, "THE VAULTS is nice.")[0, 0] == 1.0


ROTO_TABLE = Table(
    [
        Section(
            "teams",
            ["TEAM", "TEAM-PTS"],
            [["Toronto Raptors", 122], ["Philadelphia 76ers", 95]],
        ),
        Section(
            "players",
            ["PLAYER_NAME", "TEAM_CITY", "PTS", "REB", "AST"],
            [
                ["Kyle Lowry", "Toronto", 24, 4, 8],
                ["DeMar DeRozan", "Toronto", 14, 5, 3],
            ],
        ),
    ]
)


def test_numeric_cells_align_via_extracted_numbers():
    labels = alignment_labels(ROTO_TABLE, "Kyle Lowry scored 24 points.")
    assert labels[2, 0] == 1.0  # player name
    assert labels[2, 2] == 1.0  # PTS 24
    assert labels[2, 3] == 0.0  # REB 4 not mentioned
    assert labels[3, 0] == 0.0  # DeRozan not mentioned


def test_entity_aliases_credit_the_key_cell():
    # surname-only and nickname-only mentions, via the t2t.facts alias tables
    labels = alignment_labels(ROTO_TABLE, "Lowry paced the Raptors on Wednesday.")
    assert labels[2, 0] == 1.0  # Kyle Lowry via unique surname
    assert labels[0, 0] == 1.0  # Toronto Raptors via nickname
    assert labels[1, 0] == 0.0  # 76ers not mentioned


def test_none_cells_never_align():
    table = Table([Section("s", ["a", "b"], [["x", None]])])
    labels = alignment_labels(table, "x and None")
    assert labels[0, 1] == 0.0


def test_filter_keeps_grounded_drops_extrinsic_and_relationally_wrong():
    ref = (
        "The Toronto Raptors defeated the Philadelphia 76ers 122 - 95. "
        "The game took place at the Air Canada Centre on Wednesday night. "
        "The 76ers defeated the Raptors in the previous meeting. "
        "Kyle Lowry led the way with 24 points."
    )
    kept = filter_reference(ref, ROTO_TABLE)
    assert "122 - 95" in kept  # grounded + relationally correct
    assert "Air Canada Centre" not in kept  # extrinsic: no aligned fact
    assert "previous meeting" not in kept  # wrong-winner relational claim
    assert "24 points" in kept


def test_filter_returns_empty_when_nothing_grounded():
    assert filter_reference("Nothing to see here.", ROTO_TABLE) == ""


def test_filter_is_identity_ish_on_fully_grounded_text():
    ref = "The Vaults serves Fast food in the riverside area."
    assert filter_reference(ref, E2E_TABLE) == ref
