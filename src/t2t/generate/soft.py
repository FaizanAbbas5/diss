"""Generation with soft tokens spliced into the frozen LLM's embeddings.

The chat prompt is rendered with a sentinel string in the {table} slot,
tokenised once, and split at the sentinel's token ids. Sequences are then
assembled as inputs_embeds = [embed(prefix), soft (or soft+table embeds),
embed(suffix)] and passed to model.generate. Known gotchas handled here:
soft vectors are cast to the embedding dtype (fp16 under 4-bit); batches
are left-padded with the pad-token embedding and the attention mask covers
the soft positions; decoding is greedy with the same generation config as
Arm 1; output records use the exact hf.py schema (id / output /
seq_logprob) so eval + dashboard run unchanged.

Variants: 'replacement' (soft tokens only in the slot), 'augmentation'
(soft tokens prepended to the markdown table), 'random' (k random-but-fixed
vectors, the extra-tokens control for augmentation gains).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import config_hash, load_config, run_dir
from ..data.types import Example
from ..encode import ColumnVocab, SoftTableModel, featurise_table
from ..prompts import extract_final, get_messages, get_spec
from ..serialise import to_markdown
from .hf import _existing_ids, _seq_logprob, load_model_and_tokenizer

# Rare mathematical brackets: byte-level BPE keeps them out of any merge
# with the surrounding newlines, so the id subsequence is findable.
SENTINEL = "⟦TABLE⟧"

VARIANTS = ("replacement", "augmentation", "random")


def find_subsequence(ids: list[int], sub: list[int]) -> int | None:
    for i in range(len(ids) - len(sub) + 1):
        if ids[i : i + len(sub)] == sub:
            return i
    return None


def split_ids(ids: list[int], sentinel_ids: list[int]) -> tuple[list[int], list[int]]:
    pos = find_subsequence(ids, sentinel_ids)
    if pos is None:
        raise ValueError(
            "Sentinel token ids not found in the rendered prompt — the "
            "tokenizer merged the sentinel with its neighbours. Pick a rarer "
            "SENTINEL for this tokenizer."
        )
    return ids[:pos], ids[pos + len(sentinel_ids) :]


def split_prompt_ids(
    tokenizer, prompt_name: str, add_generation_prompt: bool = True
) -> tuple[list[int], list[int]]:
    """(prefix_ids, suffix_ids) around the {table} slot of the chat prompt.

    Constant per (tokenizer, prompt): the table content itself enters as
    embeddings, never as prompt text.
    """
    messages = get_messages(prompt_name, SENTINEL)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    sentinel_ids = tokenizer.encode(SENTINEL, add_special_tokens=False)
    return split_ids(ids, sentinel_ids)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    import torch

    # Checkpoints carry config dicts and vocab mappings, not just tensors.
    return torch.load(path, map_location=map_location, weights_only=False)


def resolve_checkpoint(cfg: dict, results_dir: str | Path = "results") -> Path:
    """Generation configs point at training runs via train_config (path to
    the training yaml, resolved through the same run_dir hashing so no
    hash is hardcoded) or an explicit checkpoint path."""
    if cfg.get("checkpoint"):
        return Path(cfg["checkpoint"])
    if cfg.get("train_config"):
        train_cfg = load_config(cfg["train_config"])
        return run_dir(train_cfg, results_dir) / "checkpoint.pt"
    raise ValueError("Soft generation config needs 'checkpoint' or 'train_config'")


def build_soft_model(ckpt: dict) -> tuple["SoftTableModel", ColumnVocab, dict]:
    vocab = ColumnVocab.from_dict(ckpt["vocab"])
    net = SoftTableModel.from_config(
        ckpt["config"], vocab_size=len(vocab), hidden_size=ckpt["hidden_size"]
    )
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, vocab, ckpt["stats"]


def random_soft_vectors(k: int, hidden_size: int, std: float, seed: int):
    """The 'random' control: k fixed vectors, scaled like real embeddings."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.randn(k, hidden_size, generator=gen) * std


def _batch_features(examples: list[Example], vocab, stats):
    import torch

    feats = [featurise_table(ex.table, vocab, stats) for ex in examples]
    r_max = max(f["col_id"].shape[0] for f in feats)
    c_max = max(f["col_id"].shape[1] for f in feats)
    batch: dict = {}
    for key in ("col_id", "kind", "num_feats", "str_bucket", "is_key", "cell_mask"):
        parts = []
        for f in feats:
            t = f[key]
            pad = [0, c_max - t.shape[1], 0, r_max - t.shape[0]]
            if t.dim() == 3:  # num_feats [R, C, F]
                pad = [0, 0] + pad
            parts.append(torch.nn.functional.pad(t, pad))
        batch[key] = torch.stack(parts)
    return batch


def run_generation_soft(cfg: dict, examples: list[Example], out_dir: str | Path) -> Path:
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(cfg, cfg.get("results_dir", "results"))
    ckpt = load_checkpoint(ckpt_path)
    train_cfg = ckpt["config"]
    variant = cfg.get("variant", train_cfg.get("variant", "replacement"))
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    prompt_name = cfg.get("prompt", train_cfg["prompt"])
    if prompt_name != train_cfg["prompt"]:
        print(
            f"Warning: generating with prompt {prompt_name!r} but the "
            f"checkpoint was trained with {train_cfg['prompt']!r}"
        )

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "config_hash": config_hash(cfg),
                "checkpoint": str(ckpt_path),
                "checkpoint_step": ckpt.get("step"),
                "variant": variant,
            },
            indent=2,
        ),
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
    if model.config.hidden_size != ckpt["hidden_size"]:
        raise ValueError(
            f"Checkpoint was trained against hidden_size={ckpt['hidden_size']} "
            f"but {cfg['model']} has {model.config.hidden_size}"
        )
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    emb_dtype = embed_layer.weight.dtype

    net, vocab, stats = build_soft_model(ckpt)
    net.to(device)
    k = net.n_queries

    if variant == "random":
        fixed = random_soft_vectors(
            k,
            ckpt["hidden_size"],
            std=embed_layer.weight.std().item(),
            seed=int(cfg.get("seed", 13)),
        ).to(device)

    prefix_ids, suffix_ids = split_prompt_ids(tokenizer, prompt_name)
    prefix_t = torch.tensor(prefix_ids, device=device)
    suffix_t = torch.tensor(suffix_ids, device=device)
    pad_embed = embed_layer(torch.tensor([tokenizer.pad_token_id], device=device))[0]

    spec = get_spec(prompt_name)
    batch_size = int(cfg.get("batch_size", 1))
    max_new = int(cfg.get("max_new_tokens", 128))
    # table text must match what the checkpoint was TRAINED with; default to
    # the training config's serialisation so a mismatch cannot happen
    from ..serialise import get_serialiser

    serialise = get_serialiser(
        cfg.get("serialisation", ckpt["config"].get("serialisation", "markdown"))
    )

    with open(gen_path, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            t0 = time.time()
            with torch.no_grad():
                if variant == "random":
                    soft = fixed.unsqueeze(0).expand(len(batch), -1, -1)
                else:
                    feats = {
                        key: t.to(device)
                        for key, t in _batch_features(batch, vocab, stats).items()
                    }
                    soft, _ = net(feats)
                soft = soft.to(emb_dtype)

                seqs = []
                for i, ex in enumerate(batch):
                    parts = [embed_layer(prefix_t), soft[i]]
                    if variant in ("augmentation", "random"):
                        table_ids = tokenizer.encode(
                            serialise(ex.table), add_special_tokens=False
                        )
                        parts.append(embed_layer(torch.tensor(table_ids, device=device)))
                    parts.append(embed_layer(suffix_t))
                    seqs.append(torch.cat(parts))

                t_max = max(s.shape[0] for s in seqs)
                inputs_embeds = pad_embed.expand(len(seqs), t_max, -1).clone()
                attention_mask = torch.zeros(
                    len(seqs), t_max, dtype=torch.long, device=device
                )
                for i, s in enumerate(seqs):  # left padding for generation
                    inputs_embeds[i, t_max - s.shape[0] :] = s
                    attention_mask[i, t_max - s.shape[0] :] = 1

                gen = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            # With inputs_embeds and no input_ids, sequences hold only the
            # generated tokens; slice defensively in case a future
            # transformers version prepends anything.
            new_tokens = gen.sequences[:, -len(gen.scores) :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            scores = model.compute_transition_scores(
                gen.sequences, gen.scores, normalize_logits=True
            )
            logprobs = [
                _seq_logprob(toks, row, tokenizer.eos_token_id)
                for toks, row in zip(new_tokens, scores)
            ]
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
