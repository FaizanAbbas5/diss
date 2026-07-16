from t2t.config import config_hash


def test_hash_key_order_insensitive():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_hash_sensitive_to_values():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_hash_is_short_hex():
    h = config_hash({"a": 1})
    assert len(h) == 10
    int(h, 16)
