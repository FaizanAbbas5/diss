"""Sentinel prompt splitting: pure id-level logic plus, when the cached
Qwen tokenizer is available, the real-tokenizer property that the sentinel
never merges with its neighbours in any registered prompt."""
import pytest

from t2t.generate.soft import SENTINEL, find_subsequence, split_ids, split_prompt_ids
from t2t.prompts import PROMPTS


def test_find_subsequence():
    assert find_subsequence([1, 2, 9, 9, 3], [9, 9]) == 2
    assert find_subsequence([9, 1, 9, 9], [9, 9]) == 2
    assert find_subsequence([1, 2, 3], [4]) is None
    assert find_subsequence([1], [1]) == 0


def test_split_ids_removes_sentinel_and_keeps_order():
    prefix, suffix = split_ids([1, 2, 9, 9, 3, 4], [9, 9])
    assert prefix == [1, 2] and suffix == [3, 4]


def test_split_ids_raises_when_sentinel_missing():
    with pytest.raises(ValueError, match="Sentinel"):
        split_ids([1, 2, 3], [9, 9])


@pytest.fixture(scope="module")
def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # no cache and offline
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


@pytest.mark.parametrize("prompt_name", sorted(PROMPTS))
def test_real_tokenizer_split_every_prompt(qwen_tokenizer, prompt_name):
    prefix, suffix = split_prompt_ids(qwen_tokenizer, prompt_name)
    assert prefix and suffix
    sentinel_ids = qwen_tokenizer.encode(SENTINEL, add_special_tokens=False)
    assert find_subsequence(prefix, sentinel_ids) is None
    assert find_subsequence(suffix, sentinel_ids) is None
    # the assistant header must sit at the end of the suffix, ready for
    # reference tokens (training) or generation
    tail = qwen_tokenizer.decode(suffix[-10:])
    assert "assistant" in tail


def test_real_tokenizer_split_is_clean_reassembly(qwen_tokenizer):
    """prefix + sentinel + suffix must reproduce the full prompt ids exactly
    — i.e. the sentinel did not merge with neighbouring tokens."""
    from t2t.prompts import get_messages

    messages = get_messages("e2e/faithful_v1", SENTINEL)
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full = qwen_tokenizer.encode(text, add_special_tokens=False)
    sentinel_ids = qwen_tokenizer.encode(SENTINEL, add_special_tokens=False)
    prefix, suffix = split_prompt_ids(qwen_tokenizer, "e2e/faithful_v1")
    assert prefix + sentinel_ids + suffix == full
