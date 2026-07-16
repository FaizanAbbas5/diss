"""Tests for the claim-verification plumbing (splitting, retrieval, scoring
aggregation). The NLI model itself is stubbed so tests stay offline."""
from t2t.data.types import Section, Table
from t2t.eval.claims import (
    build_premise,
    score_run_claims,
    score_text_claims,
    split_sentences,
)

GAME = Table(
    [
        Section("teams", ["TEAM", "TEAM-PTS"], [["Toronto Raptors", 122], ["Philadelphia 76ers", 95]]),
        Section("players", ["PLAYER_NAME", "PTS"], [["Kyle Lowry", 22], ["Joel Embiid", 30]]),
    ]
)

RESTAURANT = Table(
    [Section("restaurant", ["name", "familyFriendly"], [["The Vaults", "no"]])]
)


def stub_scorer(pairs):
    """Entail claims containing 'won'; contradict claims containing 'lost'."""
    out = []
    for _, hyp in pairs:
        if "won" in hyp:
            out.append({"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
        elif "lost" in hyp:
            out.append({"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9})
        else:
            out.append({"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1})
    return out


def test_split_sentences():
    text = "The Raptors won 122 - 95. Kyle Lowry scored 22 points! A rout."
    assert split_sentences(text) == [
        "The Raptors won 122 - 95.",
        "Kyle Lowry scored 22 points!",
        "A rout.",
    ]


def test_premise_retrieves_mentioned_player_only():
    premise = build_premise(GAME, "Kyle Lowry scored 22 points.")
    assert "Kyle Lowry" in premise and "PTS: 22" in premise
    assert "Embiid" not in premise


def test_premise_falls_back_to_first_multirow_section():
    premise = build_premise(GAME, "It was a rout from start to finish.")
    assert "teams" in premise  # team rows included when no entity matches


def test_premise_always_includes_single_row_sections():
    premise = build_premise(RESTAURANT, "A lovely place.")
    assert "familyFriendly: no" in premise


def test_premise_states_derived_game_result():
    premise = build_premise(GAME, "The Raptors defeated the 76ers 122 - 95.")
    assert "Toronto Raptors defeated the Philadelphia 76ers 122 to 95" in premise
    # prepended, so truncation can't drop it
    assert premise.index("defeated") < premise.index("TEAM-PTS")


def test_no_derived_result_for_single_row_tables():
    premise = build_premise(RESTAURANT, "A lovely place.")
    assert "defeated" not in premise


def test_score_text_claims_labels_and_rates():
    result = score_text_claims(
        "The Raptors won the game. The 76ers lost badly. Weather was nice.",
        GAME,
        stub_scorer,
    )
    assert result["n_claims"] == 3
    assert result["n_entailed"] == 1
    assert result["n_contradicted"] == 1
    assert abs(result["entailed_rate"] - 1 / 3) < 1e-9


def test_score_run_claims_batches_and_regroups():
    outputs = {
        "a": "The Raptors won.",
        "b": "The 76ers lost. Something neutral happened.",
    }
    tables = {"a": GAME, "b": GAME}
    result = score_run_claims(outputs, tables, scorer=stub_scorer)
    assert result["per_item"]["a"]["n_entailed"] == 1
    assert result["per_item"]["b"]["n_contradicted"] == 1
    assert result["aggregate"]["total_claims"] == 3
    assert result["aggregate"]["pct_items_with_contradiction"] == 0.5


def test_empty_output_handled():
    result = score_text_claims("", GAME, stub_scorer)
    assert result["n_claims"] == 0 and result["entailed_rate"] is None
