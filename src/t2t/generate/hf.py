"""Hugging Face generation harness.

Deterministic (greedy) decoding, identical across arms; resumable JSONL
output so interrupted Colab sessions pick up where they left off.

Every generation also records its length-normalised sequence log-probability
(uncertainty-based faithfulness signal; Xiao & Wang 2021 showed predictive
uncertainty correlates with hallucination in data-to-text). Captured at
generation time so no run has to be repeated to get it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import config_hash
from ..data.types import Example
from ..prompts import extract_final, get_messages, get_spec
from ..serialise import get_serialiser, to_markdown


def load_model_and_tokenizer(model_name: str, quantization: str = "none"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict = {}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif quantization != "none":
        raise ValueError(f"Unknown quantization: {quantization!r}")
    elif torch.cuda.is_available():
        kwargs["dtype"] = "auto"
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def _seq_logprob(token_ids, scores_row, eos_id: int) -> float | None:
    """Mean per-token log-probability up to and including the first EOS
    (positions after EOS are padding in batched generation)."""
    lps: list[float] = []
    for tid, lp in zip(token_ids.tolist(), scores_row.tolist()):
        lps.append(lp)
        if tid == eos_id:
            break
    return sum(lps) / len(lps) if lps else None


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def build_prompt(tokenizer, prompt_name: str, example: Example, serialise=to_markdown) -> str:
    messages = get_messages(prompt_name, serialise(example.table))
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_generation(cfg: dict, examples: list[Example], out_dir: str | Path) -> Path:
    import torch

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

    model, tokenizer = load_model_and_tokenizer(
        cfg["model"], cfg.get("quantization", "none")
    )
    serialise = get_serialiser(cfg.get("serialisation", "markdown"))
    device = next(model.parameters()).device
    batch_size = int(cfg.get("batch_size", 1))
    max_new = int(cfg.get("max_new_tokens", 128))

    with open(gen_path, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            prompts = [build_prompt(tokenizer, cfg["prompt"], ex, serialise) for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            new_tokens = gen.sequences[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            scores = model.compute_transition_scores(
                gen.sequences, gen.scores, normalize_logits=True
            )
            logprobs = [
                _seq_logprob(toks, row, tokenizer.eos_token_id)
                for toks, row in zip(new_tokens, scores)
            ]
            spec = get_spec(cfg["prompt"])
            for ex, text, lp in zip(batch, texts, logprobs):
                raw = text.strip()
                final = extract_final(raw, spec)
                record = {
                    "id": ex.id,
                    "output": final,
                    "seq_logprob": round(lp, 4) if lp is not None else None,
                }
                if final != raw:
                    record["raw_output"] = raw
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{min(start + batch_size, len(todo))}/{len(todo)}] {time.time() - t0:.1f}s",
                flush=True,
            )
    return gen_path
