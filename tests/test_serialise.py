from t2t.data.types import Section, Table
from t2t.serialise import to_markdown


def test_multi_row_section():
    t = Table(
        [
            Section(
                "players",
                ["PLAYER_NAME", "PTS", "REB"],
                [["LeBron James", 27, 8], ["Kevin Love", 14, 11]],
            )
        ]
    )
    md = to_markdown(t)
    assert "### players" in md
    assert "| PLAYER_NAME | PTS | REB |" in md
    assert "| LeBron James | 27 | 8 |" in md


def test_single_row_transposed_to_attribute_value():
    t = Table([Section("restaurant", ["name", "food"], [["The Vaults", "Italian"]])])
    md = to_markdown(t)
    assert "| attribute | value |" in md
    assert "| name | The Vaults |" in md


def test_none_rendered_as_na():
    t = Table([Section("s", ["a"], [[None], [1]])])
    assert "N/A" in to_markdown(t)


def test_deterministic():
    t = Table([Section("s", ["a", "b"], [[1, 2], [3, 4]])])
    assert to_markdown(t) == to_markdown(t)
