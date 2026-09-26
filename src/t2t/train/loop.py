"""Arm-2 training loop: encoder + projector against the frozen LLM.

Loss = LM cross-entropy on reference tokens through the frozen LLM
(+ aux_weight * BCE on per-cell "mentioned" labels). AdamW on the trainable
module only, cosine schedule with warmup, gradient accumulation to the
configured effective batch. Checkpoints every checkpoint_every optimizer
steps to the hashed run dir and resumes automatically, since sessions may
be interrupted at any step. Resume restarts the (seeded) dataloader shuffle at an
epoch boundary; step count and optimizer/scheduler state are exact.

CPU smoke uses fp32 0.5B; GPU runs use 4-bit NF4 (Linear4bit backprops
gradients w.r.t. inputs, which is all a frozen LLM needs) with gradient
checkpointing and use_cache off.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from ..config import config_hash, run_dir, set_seed
from .. import data
from ..encode import ColumnVocab, SoftTableModel, compute_column_stats
from .collate import Arm2Collator
from .dataset import Arm2Dataset

_ENCODER_KEYS = ("col_id", "kind", "num_feats", "str_bucket", "is_key", "cell_mask")


def _save_checkpoint(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # write-then-rename: never leave a torn file
    torch.save(payload, tmp)
    tmp.replace(path)


def train(cfg: dict, results_dir: str | Path = "results") -> Path:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from ..generate.hf import load_model_and_tokenizer

    out = run_dir(cfg, results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps({"config": cfg, "config_hash": config_hash(cfg)}, indent=2),
        encoding="utf-8",
    )
    seed = int(cfg.get("seed", 13))
    set_seed(seed)

    examples = data.load(cfg["dataset"], cfg.get("split", "train"), limit=cfg.get("limit"))
    vocab = ColumnVocab.from_examples(examples)
    stats = compute_column_stats(examples)
    print(f"{len(examples)} examples, column vocab {len(vocab)}")

    model, tokenizer = load_model_and_tokenizer(cfg["model"], cfg.get("quantization", "none"))
    model.requires_grad_(False)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # transformers only applies checkpointing when self.training is
        # True, and load_model_and_tokenizer calls model.eval(); without
        # this line the flag is set but inert and full activations are
        # kept (OOM on 20GB at augmentation sequence lengths). Qwen2 has
        # no dropout, so train() changes nothing else.
        model.train()
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    hidden_size = model.config.hidden_size

    from ..serialise import get_serialiser

    dataset = Arm2Dataset(
        examples,
        tokenizer,
        cfg["prompt"],
        vocab,
        stats,
        variant=cfg.get("variant", "replacement"),
        filter_refs=bool(cfg.get("filter_references", False)),
        max_refs_per_example=int(cfg.get("max_refs_per_example", 1)),
        max_ref_tokens=int(cfg.get("max_ref_tokens", 512)),
        serialise=get_serialiser(cfg.get("serialisation", "markdown")),
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after filtering")
    collator = Arm2Collator(k=int(cfg.get("k", 16)), pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )

    net = SoftTableModel.from_config(cfg, vocab_size=len(vocab), hidden_size=hidden_size)
    net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Trainable params: {n_params / 1e6:.2f}M (LLM frozen)")

    max_steps = int(cfg["max_steps"])
    grad_accum = int(cfg.get("grad_accum", 1))
    aux_weight = float(cfg.get("aux_weight", 0.3))
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(float(cfg.get("warmup_ratio", 0.05)) * max_steps)),
        num_training_steps=max_steps,
    )

    ckpt_path = out / "checkpoint.pt"
    log_path = out / "train_log.jsonl"
    step = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        step = ckpt["step"]
        print(f"Resumed from {ckpt_path} at step {step}")
        if log_path.exists():
            # Drop log entries past the checkpoint (steps the killed run got
            # through but that will now be replayed) so curves stay clean.
            with open(log_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            kept = [r for r in rows if r["step"] <= step]
            if len(kept) < len(rows):
                log_path.write_text(
                    "".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8"
                )
    if step >= max_steps:
        print(f"Training already complete ({step}/{max_steps})")
        return out

    def save(step: int) -> None:
        _save_checkpoint(
            ckpt_path,
            {
                "step": step,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": cfg,
                "vocab": vocab.to_dict(),
                "stats": stats,
                "hidden_size": hidden_size,
                "n_params": n_params,
            },
        )

    checkpoint_every = int(cfg.get("checkpoint_every", 100))
    bce = torch.nn.BCEWithLogitsLoss()
    net.train()

    def batches():
        epoch = step * grad_accum // max(1, len(loader))
        while True:
            # Re-seed the shuffle per epoch so resume at an epoch boundary
            # replays the same order an uninterrupted run would have seen.
            loader.generator.manual_seed(seed + epoch)
            yield from loader
            epoch += 1

    batch_iter = batches()
    # The run dir was created at startup, but model loading takes minutes;
    # recreate it in case an external sync or cleanup removed it.
    out.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        while step < max_steps:
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            lm_sum = aux_sum = 0.0
            for _ in range(grad_accum):
                batch = {key: t.to(device) for key, t in next(batch_iter).items()}
                soft, aux_logits = net({key: batch[key] for key in _ENCODER_KEYS})

                embeds = embed_layer(batch["input_ids"])
                b_idx = torch.arange(soft.shape[0], device=device).unsqueeze(1)
                embeds[b_idx, batch["soft_pos"]] = soft.to(embeds.dtype)

                # Base model + LM head on loss positions only. Passing
                # labels= to the causal-LM head materialises full-sequence
                # logits in fp32 (~2k tokens x 152k vocab), an OOM at
                # augmentation sequence lengths. The loss is masked to
                # reference tokens anyway; computing logits just there is
                # mathematically identical (same mean CE over non-ignored
                # positions as the HF labels= path).
                hidden = model.model(
                    inputs_embeds=embeds,
                    attention_mask=batch["attention_mask"],
                ).last_hidden_state
                shift_labels = batch["labels"][:, 1:]
                sel = shift_labels != -100
                logits = model.lm_head(hidden[:, :-1][sel]).float()
                lm_loss = torch.nn.functional.cross_entropy(
                    logits, shift_labels[sel]
                )
                mask = batch["cell_mask"]
                aux_loss = bce(aux_logits[mask], batch["aux_labels"][mask])
                loss = lm_loss + aux_weight * aux_loss
                (loss / grad_accum).backward()
                lm_sum += lm_loss.item()
                aux_sum += aux_loss.item()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            entry = {
                "step": step,
                "loss": round(lm_sum / grad_accum + aux_weight * aux_sum / grad_accum, 4),
                "lm_loss": round(lm_sum / grad_accum, 4),
                "aux_loss": round(aux_sum / grad_accum, 4),
                "lr": scheduler.get_last_lr()[0],
                "sec": round(time.time() - t0, 2),
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % int(cfg.get("log_every", 1)) == 0 or step == max_steps:
                print(
                    f"step {step}/{max_steps} lm {entry['lm_loss']:.4f} "
                    f"aux {entry['aux_loss']:.4f} ({entry['sec']}s)",
                    flush=True,
                )
            if step % checkpoint_every == 0 or step == max_steps:
                save(step)

    if math.isnan(lm_sum):
        raise RuntimeError("LM loss is NaN: check dtype/lr")
    print(f"Done: {step} steps -> {ckpt_path}")
    return out
