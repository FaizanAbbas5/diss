"""Groq backend tests with the HTTP layer stubbed out (no network, no key)."""
import json

import pytest

from t2t.data.types import Example, Section, Table
from t2t.generate import groq


class FakeResponse:
    def __init__(self, status_code, content="", headers=None):
        self.status_code = status_code
        self.text = content
        self.headers = headers or {}
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _example(i="x-0"):
    table = Table([Section("restaurant", ["name"], [["The Vaults"]])])
    return Example(id=i, table=table)


CFG = {
    "name": "groq-test",
    "backend": "groq",
    "model": "llama-3.1-8b-instant",
    "prompt": "e2e/faithful_v1",
    "max_new_tokens": 64,
    "request_interval": 0,
    "seed": 13,
}


def test_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # stub the loader so the developer's real .env can't satisfy the check
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        groq.run_generation_groq(CFG, [_example()], tmp_path / "out")


def test_payload_and_output(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(json)
        return FakeResponse(200, "A pub called The Vaults.")

    monkeypatch.setattr(groq.requests, "post", fake_post)
    path = groq.run_generation_groq(CFG, [_example()], tmp_path)

    payload = seen[0]
    assert payload["temperature"] == 0
    assert payload["model"] == "llama-3.1-8b-instant"
    assert "The Vaults" in payload["messages"][1]["content"]

    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec == {"id": "x-0", "output": "A pub called The Vaults.", "seq_logprob": None}


def test_factsheet_serialisation_reaches_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(json)
        return FakeResponse(200, "ok")

    monkeypatch.setattr(groq.requests, "post", fake_post)
    cfg = {**CFG, "serialisation": "factsheet"}
    groq.run_generation_groq(cfg, [_example()], tmp_path)

    user_msg = seen[0]["messages"][1]["content"]
    assert "name: The Vaults" in user_msg  # fact-sheet lines, not markdown
    assert "| attribute | value |" not in user_msg


def test_retries_on_429_then_succeeds(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq.time, "sleep", lambda s: None)
    responses = [FakeResponse(429, headers={"retry-after": "1"}), FakeResponse(200, "ok text")]

    monkeypatch.setattr(
        groq.requests, "post", lambda *a, **k: responses.pop(0)
    )
    path = groq.run_generation_groq(CFG, [_example()], tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["output"] == "ok text"


def test_resumes_without_calling_api(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    (tmp_path / "generations.jsonl").write_text(
        json.dumps({"id": "x-0", "output": "cached", "seq_logprob": None}) + "\n",
        encoding="utf-8",
    )

    def explode(*a, **k):
        raise AssertionError("API called despite cache")

    monkeypatch.setattr(groq.requests, "post", explode)
    groq.run_generation_groq(CFG, [_example()], tmp_path)


def test_non_retryable_error_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        groq.requests, "post", lambda *a, **k: FakeResponse(404, "model not found")
    )
    with pytest.raises(RuntimeError, match="404"):
        groq.run_generation_groq(CFG, [_example()], tmp_path)
