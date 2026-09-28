"""Degenerate sentence fragments must not reach the NLI scorer."""
from t2t.data.types import Section, Table
from t2t.eval.claims import score_run_claims


def _table():
    return Table(sections=[Section(
        name="teams",
        columns=["TEAM", "TEAM-PTS"],
        rows=[["Jazz", 84], ["Bucks", 81]],
        key_column=0,
    )])


def test_fragments_skipped():
    seen = []

    def fake_scorer(pairs):
        seen.extend(s for _p, s in pairs)
        return [{"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
                for _ in pairs]

    out = {"x": "The Jazz beat the Bucks 84 to 81. 23). Final."}
    res = score_run_claims(out, {"x": _table()}, scorer=fake_scorer)
    assert seen == ["The Jazz beat the Bucks 84 to 81."]
    assert res["per_item"]["x"]["n_claims"] == 1


def test_compound_sentence_split_into_clause_units():
    seen = []

    def fake_scorer(pairs):
        seen.extend(s for _p, s in pairs)
        return [{"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
                for _ in pairs]

    out = {"x": ("The Jazz beat the Bucks 84 to 81, while Khris Middleton "
                 "led the Bucks with 18 points; Giannis Antetokounmpo "
                 "added 14 points, and Malcolm Brogdon impressed.")}
    res = score_run_claims(out, {"x": _table()}, scorer=fake_scorer)
    assert seen == [
        "The Jazz beat the Bucks 84 to 81",
        "Khris Middleton led the Bucks with 18 points",
        "Giannis Antetokounmpo added 14 points",
        "Malcolm Brogdon impressed.",
    ]
    assert res["per_item"]["x"]["n_claims"] == 4


def test_stat_lists_not_split():
    seen = []

    def fake_scorer(pairs):
        seen.extend(s for _p, s in pairs)
        return [{"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}
                for _ in pairs]

    out = {"x": "Middleton finished with 18 points, 6 rebounds, and 9 assists."}
    score_run_claims(out, {"x": _table()}, scorer=fake_scorer)
    assert seen == ["Middleton finished with 18 points, 6 rebounds, and 9 assists."]
