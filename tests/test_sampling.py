"""Seeded subsampling in data.load — the final runs depend on this being
deterministic and unbiased (first-N on RotoWire is temporally ordered)."""
from unittest.mock import patch

from t2t import data
from t2t.data.types import Example, Section, Table


def _fake_examples(n=50):
    return [
        Example(id=f"x-{i:03d}", table=Table([Section("s", ["a"], [[i]])]))
        for i in range(n)
    ]


def _load(**kwargs):
    with patch("t2t.data.e2e.load", return_value=_fake_examples()):
        return data.load("e2e_cleaned", "validation", **kwargs)


def test_no_seed_takes_first_n():
    ids = [e.id for e in _load(limit=5)]
    assert ids == ["x-000", "x-001", "x-002", "x-003", "x-004"]


def test_seeded_sample_is_deterministic():
    a = [e.id for e in _load(limit=10, sample_seed=7)]
    b = [e.id for e in _load(limit=10, sample_seed=7)]
    assert a == b and len(a) == 10


def test_seeded_sample_differs_from_first_n_and_across_seeds():
    s7 = [e.id for e in _load(limit=10, sample_seed=7)]
    s8 = [e.id for e in _load(limit=10, sample_seed=8)]
    first = [e.id for e in _load(limit=10)]
    assert s7 != first and s7 != s8


def test_sample_preserves_original_order():
    ids = [e.id for e in _load(limit=10, sample_seed=7)]
    assert ids == sorted(ids)


def test_limit_geq_population_returns_all():
    assert len(_load(limit=200, sample_seed=7)) == 50
