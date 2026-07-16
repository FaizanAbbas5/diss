from t2t.data.rotowire import parse_entry

ENTRY = {
    "home_city": "Toronto",
    "home_name": "Raptors",
    "vis_city": "Philadelphia",
    "vis_name": "76ers",
    "home_line": {"TEAM-PTS": "122", "TEAM-CITY": "Toronto", "TEAM-NAME": "Raptors"},
    "vis_line": {"TEAM-PTS": "95", "TEAM-CITY": "Philadelphia", "TEAM-NAME": "76ers"},
    "box_score": {
        "PLAYER_NAME": {"0": "Kyle Lowry", "1": "Bench Guy"},
        "FIRST_NAME": {"0": "Kyle", "1": "Bench"},
        "MIN": {"0": "35", "1": "N/A"},
        "PTS": {"0": "22", "1": "N/A"},
        "EMPTY_STAT": {"0": "N/A", "1": "N/A"},
    },
    "summary": ["The", "Raptors", "won", "."],
}


def _table():
    return parse_entry(ENTRY)


def test_team_section_drops_redundant_columns():
    teams = _table().sections[0]
    assert teams.columns == ["TEAM", "TEAM-PTS"]
    assert teams.rows[0] == ["Toronto Raptors", 122]
    assert teams.rows[1] == ["Philadelphia 76ers", 95]


def test_player_section_drops_dnp_rows():
    players = _table().sections[1]
    names = [row[0] for row in players.rows]
    assert names == ["Kyle Lowry"]


def test_player_section_drops_empty_columns_and_name_parts():
    players = _table().sections[1]
    assert "EMPTY_STAT" not in players.columns
    assert "FIRST_NAME" not in players.columns
    assert set(players.columns) == {"PLAYER_NAME", "MIN", "PTS"}


def test_values_parsed_to_ints():
    players = _table().sections[1]
    row = players.rows[0]
    assert row[players.columns.index("PTS")] == 22
