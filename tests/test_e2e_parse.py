import pytest

from t2t.data.e2e import parse_mr


def test_parse_mr_basic():
    pairs = parse_mr("name[The Vaults], eatType[pub], customer rating[5 out of 5]")
    assert pairs[0] == ("name", "The Vaults")
    assert dict(pairs)["customer rating"] == "5 out of 5"
    assert len(pairs) == 3


def test_parse_mr_rejects_garbage():
    with pytest.raises(ValueError):
        parse_mr("not a meaning representation")
