from t2t.data.types import Section, Table
from t2t.eval.relations import score_text_relations

GAME = Table(
    [
        Section(
            "teams",
            ["TEAM", "TEAM-PTS", "TEAM-WINS", "TEAM-LOSSES"],
            [["Toronto Raptors", 122, 11, 6], ["Philadelphia 76ers", 95, 4, 14]],
        ),
        Section(
            "players",
            ["PLAYER_NAME", "TEAM_CITY", "PTS", "REB", "AST"],
            [
                ["Kyle Lowry", "Toronto", 24, 4, 8],
                ["DeMar DeRozan", "Toronto", 14, 5, 3],
                ["Robert Covington", "Philadelphia", 20, 6, 1],
                ["Jahlil Okafor", "Philadelphia", 15, 8, 1],
                ["Ersan Ilyasova", "Philadelphia", 15, 7, 2],
            ],
        ),
    ]
)


def _wrong(text):
    return score_text_relations(text, GAME)["wrong"]


def test_correct_winner_claim_passes():
    assert _wrong("The Toronto Raptors defeated the Philadelphia 76ers 122-95.") == []


def test_wrong_winner_claim_flagged():
    wrong = _wrong("The 76ers defeated the Raptors 95-122.")
    assert len(wrong) == 1 and wrong[0]["type"] == "winner"


def test_reversed_victory_phrase_flagged():
    wrong = _wrong("The 76ers secured a 95-122 victory.")
    assert len(wrong) == 1 and wrong[0]["type"] == "winner"


def test_fell_to_inversion():
    assert _wrong("The 76ers fell to the Raptors on Monday.") == []
    wrong = _wrong("The Raptors fell to the 76ers on Monday.")
    assert len(wrong) == 1 and wrong[0]["type"] == "winner"


def test_margin_claim():
    assert _wrong("It was a 27-point victory for Toronto.") == []
    wrong = _wrong("It was a 20-point victory for Toronto.")
    assert any(c["type"] == "margin" for c in wrong)


def test_correct_attribution_passes():
    assert _wrong("Kyle Lowry scored 24 points and grabbed 4 rebounds.") == []


def test_wrong_attribution_flagged():
    wrong = _wrong("Kyle Lowry scored 29 points.")
    assert len(wrong) == 1 and wrong[0]["type"] == "attribution"


def test_true_leader_passes():
    assert _wrong("Kyle Lowry led the Raptors with 24 points.") == []


def test_false_leader_flagged():
    # DeRozan's 14 points are correct, but Lowry leads Toronto in scoring
    wrong = _wrong("DeMar DeRozan led the team with 14 points.")
    assert len(wrong) == 1 and wrong[0]["type"] == "leader"


def test_leader_tie_accepts_either_player():
    # Okafor and Ilyasova are tied on 15 points... but Covington has 20;
    # rebounds: Okafor 8 leads Philadelphia
    assert _wrong("Jahlil Okafor led Philadelphia with 8 rebounds.") == []
    wrong = _wrong("Ersan Ilyasova led Philadelphia with 7 rebounds.")
    assert len(wrong) == 1 and wrong[0]["type"] == "leader"


def test_team_misattribution_flagged():
    wrong = _wrong("The Raptors' Robert Covington added 20 points.")
    assert len(wrong) == 1 and wrong[0]["type"] == "team-attribution"


def test_multi_player_clause_skipped():
    # precision-first: no claim extracted, even though 99 is wrong
    assert _wrong("Kyle Lowry and DeMar DeRozan combined for 99 points.") == []


def test_clause_split_keeps_stat_lists_together():
    text = (
        "DeMar DeRozan added 14 points and 5 rebounds, "
        "while Robert Covington contributed 20 points and 6 rebounds."
    )
    result = score_text_relations(text, GAME)
    assert result["n_claims"] >= 4 and result["wrong"] == []


def test_surname_needs_word_boundary():
    # regression: 'Lin' must not match inside 'line'
    t = Table(
        [
            GAME.sections[0],
            Section(
                "players",
                ["PLAYER_NAME", "TEAM_CITY", "PTS", "REB", "AST"],
                [["Jeremy Lin", "Toronto", 10, 2, 5], ["Kyle Lowry", "Toronto", 24, 4, 8]],
            ),
        ]
    )
    wrong = score_text_relations(
        "The 76ers shot 62% from the free throw line.", t
    )["wrong"]
    assert wrong == []


def test_no_teams_section_yields_no_claims():
    t = Table([Section("restaurant", ["name"], [["The Vaults"]])])
    assert score_text_relations("A lovely pub that won awards.", t)["n_claims"] == 0
