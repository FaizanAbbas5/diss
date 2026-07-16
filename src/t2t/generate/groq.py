"""Groq API backend (OpenAI-compatible chat completions) for pilot runs.

Groq's free tier serves open models (e.g. llama-3.1-8b-instant) at high
speed, useful for metric calibration, annotation-protocol pilots, and
cross-family robustness checks.

NOT valid for the dissertation's arm comparison: the served stack differs
from our local frozen LLM, there is no embedding access for the adapter
arm, and no logprobs are returned (seq_logprob is null in these runs).

Requires GROQ_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from ..config import config_hash
from ..data.types import Example
from ..prompts import extract_final, get_messages, get_spec
from ..serialise import get_serialiser
from .hf import _existing_ids

_URL = "https://api.groq.com/openai/v1/chat/completions"
_MAX_ATTEMPTS = 6
_RETRYABLE = {429, 500, 502, 503}


def _complete(payload: dict, api_key: str, timeout: int = 120) -> str:
    for attempt in range(_MAX_ATTEMPTS):
        resp = requests.post(
            _URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code in _RETRYABLE and attempt < _MAX_ATTEMPTS - 1:
            wait = min(float(resp.headers.get("retry-after") or 2**attempt), 60.0)
            print(f"  HTTP {resp.status_code}, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("Groq API: retries exhausted")


def run_generation_groq(
    cfg: dict, examples: list[Example], out_dir: str | Path
) -> Path:
    from dotenv import load_dotenv

    load_dotenv()  # picks up GROQ_API_KEY from a repo-root .env if present
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not set; export it (or set it in the Colab cell) "
            "to use backend: groq"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps({"config": cfg, "config_hash": config_hash(cfg)}, indent=2),
        encoding="utf-8",
    )

    gen_path = out_dir / "generations.jsonl"
    done = _existing_ids(gen_path)
    todo = [ex for ex in examples if ex.id not in done]
    if not todo:
        print(f"All {len(examples)} generations already cached in {gen_path}")
        return gen_path
    if done:
        print(f"Resuming: {len(done)} cached, {len(todo)} to generate")

    spec = get_spec(cfg["prompt"])
    serialise = get_serialiser(cfg.get("serialisation", "markdown"))
    interval = float(cfg.get("request_interval", 2.0))

    with open(gen_path, "a", encoding="utf-8") as out:
        for n, ex in enumerate(todo, 1):
            payload = {
                "model": cfg["model"],
                "messages": get_messages(cfg["prompt"], serialise(ex.table)),
                "temperature": 0,
                "max_tokens": int(cfg.get("max_new_tokens", 128)),
                "seed": int(cfg.get("seed", 13)),
            }
            t0 = time.time()
            raw = _complete(payload, api_key).strip()
            final = extract_final(raw, spec)
            record = {"id": ex.id, "output": final, "seq_logprob": None}
            if final != raw:
                record["raw_output"] = raw
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n}/{len(todo)}] {time.time() - t0:.1f}s", flush=True)
            if n < len(todo) and interval > 0:
                time.sleep(interval)
    return gen_path
