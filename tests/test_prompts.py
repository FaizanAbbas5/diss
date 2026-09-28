import pytest

from t2t.prompts import PROMPTS, extract_final, get_messages, get_spec


def test_all_prompts_have_table_slot():
    for name, spec in PROMPTS.items():
        assert "{table}" in spec["user"], name
        assert spec["system"], name


def test_get_messages_formats_table():
    msgs = get_messages("e2e/faithful_v1", "| name | X |")
    assert msgs[0]["role"] == "system"
    assert "| name | X |" in msgs[1]["content"]


def test_unknown_prompt_raises():
    with pytest.raises(KeyError):
        get_spec("e2e/nonexistent")


def test_extract_final_strips_scratch_work():
    spec = get_spec("e2e/structured_v1")
    text = "- name: X\n- food: Italian\nSUMMARY: X serves Italian food."
    assert extract_final(text, spec) == "X serves Italian food."


def test_extract_final_uses_last_marker():
    spec = {"extract_marker": "SUMMARY:"}
    text = "SUMMARY: draft\nrevised facts\nSUMMARY: final text"
    assert extract_final(text, spec) == "final text"


def test_extract_final_tolerates_markdown_and_case():
    spec = get_spec("e2e/structured_v1")
    text = "- **Family Friendly:** No\n- **Summary:** Alimentum is in the city centre."
    assert extract_final(text, spec) == "Alimentum is in the city centre."


def test_extract_final_noop_without_marker():
    assert extract_final("  plain text ", get_spec("e2e/plain_v1")) == "plain text"
    assert extract_final("no marker here", get_spec("e2e/structured_v1")) == "no marker here"
