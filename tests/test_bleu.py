import math

from t2t.eval.bleu import bleu_from_counts, compute_bleu, item_counts, tokenise


def test_tokenise_splits_punctuation_both_sides():
    # pre-tokenised reference and raw output must yield the same tokens
    assert tokenise("76ers , 122 - 95 .") == tokenise("76ers, 122-95.")


def test_identical_text_scores_100():
    ref = "The Raptors defeated the 76ers 122 - 95 on Monday night ."
    res = compute_bleu([ref], [[ref]])
    assert res["bleu"] == 100.0


def test_disjoint_text_scores_zero():
    res = compute_bleu(["completely different words entirely here now"],
                       [["The Raptors defeated the 76ers on Monday"]])
    assert res["bleu"] == 0.0


def test_partial_overlap_between_zero_and_100():
    res = compute_bleu(["The Raptors defeated the 76ers on Tuesday night ."],
                       [["The Raptors defeated the 76ers on Monday night ."]])
    assert 0 < res["bleu"] < 100


def test_brevity_penalty_punishes_short_output():
    long_ref = " ".join(["the Raptors defeated the 76ers"] * 6)
    short = compute_bleu(["the Raptors defeated the 76ers"], [[long_ref]])
    full = compute_bleu([long_ref], [[long_ref]])
    assert short["brevity_penalty"] < 1.0
    assert short["bleu"] < full["bleu"]


def test_corpus_bleu_equals_bleu_from_summed_item_counts():
    # the property the bootstrap relies on: corpus BLEU is a function of
    # summed per-item counts
    preds = ["The Raptors won 122 - 95 .", "Lowry scored 24 points ."]
    refs = [["The Raptors won the game 122 - 95 ."], ["Lowry added 24 points ."]]
    res = compute_bleu(preds, refs)
    items = res["per_item"]
    agg = {k: sum(d[k] for d in items) for k in items[0]}
    manual = bleu_from_counts(
        [agg[f"bleu_match_{n}"] for n in range(1, 5)],
        [agg[f"bleu_total_{n}"] for n in range(1, 5)],
        agg["bleu_hyp_len"], agg["bleu_ref_len"],
    )
    assert math.isclose(manual, res["bleu"], abs_tol=0.01)


def test_multi_reference_takes_best_match():
    pred = "The Raptors won the game ."
    one = compute_bleu([pred], [["The Raptors lost the game ."]])["bleu"]
    two = compute_bleu([pred], [["The Raptors lost the game .",
                                 "The Raptors won the game ."]])["bleu"]
    assert two > one
