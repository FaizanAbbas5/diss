"""Batch collation for Arm-2 training.

Produces right-padded batches (fine for teacher forcing; generation uses
left padding in t2t.generate.soft). Soft-token positions carry the pad id
in input_ids as a placeholder — the training loop overwrites their
embeddings with the encoder output via soft_pos — and are covered by the
attention mask. Labels are -100 everywhere except reference tokens, so
prompt, soft, table, and padding positions never contribute to the LM loss.
"""
from __future__ import annotations

import torch


class Arm2Collator:
    def __init__(self, k: int, pad_token_id: int):
        self.k = k
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        batch = self._encoder_batch(items)

        plans = []
        for it in items:
            table_ids = it["table_ids"] or []
            ids = (
                list(it["prefix_ids"])
                + [self.pad_token_id] * self.k
                + list(table_ids)
                + list(it["suffix_ids"])
                + list(it["ref_ids"])
            )
            labels = [-100] * (len(ids) - len(it["ref_ids"])) + list(it["ref_ids"])
            plans.append((ids, labels, len(it["prefix_ids"])))

        t_max = max(len(ids) for ids, _, _ in plans)
        n = len(plans)
        input_ids = torch.full((n, t_max), self.pad_token_id, dtype=torch.long)
        labels = torch.full((n, t_max), -100, dtype=torch.long)
        attention_mask = torch.zeros(n, t_max, dtype=torch.long)
        soft_pos = torch.zeros(n, self.k, dtype=torch.long)
        for i, (ids, labs, soft_start) in enumerate(plans):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, : len(labs)] = torch.tensor(labs)
            attention_mask[i, : len(ids)] = 1
            soft_pos[i] = torch.arange(soft_start, soft_start + self.k)

        batch.update(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            soft_pos=soft_pos,
        )
        return batch

    def _encoder_batch(self, items: list[dict]) -> dict[str, torch.Tensor]:
        r_max = max(it["col_id"].shape[0] for it in items)
        c_max = max(it["col_id"].shape[1] for it in items)
        batch: dict[str, torch.Tensor] = {}
        for key in (
            "col_id",
            "kind",
            "num_feats",
            "str_bucket",
            "is_key",
            "cell_mask",
            "aux_labels",
        ):
            parts = []
            for it in items:
                t = it[key]
                pad = [0, c_max - t.shape[1], 0, r_max - t.shape[0]]
                if t.dim() == 3:  # num_feats [R, C, F]
                    pad = [0, 0] + pad
                parts.append(torch.nn.functional.pad(t, pad))
            batch[key] = torch.stack(parts)
        return batch
