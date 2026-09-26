"""Permutation-invariant table encoder -> k soft vectors -> projector.

Architecture:
cell = column-ID embedding + value encoding (numeric MLP / string hash
buckets / learned null) -> row = masked mean of cells + key-cell value
encoding -> 1-2 transformer layers over rows (no positional encoding, so
row order cannot leak in) -> k=16 learned queries cross-attend over rows
(mini Q-Former) -> linear/MLP projector into the frozen LLM's embedding
space. The projector output dimension is read from model.config.hidden_size
at runtime, never hardcoded.

An auxiliary head predicts per-cell "mentioned in reference" (BCE), giving
the encoder a direct content-selection signal alongside the LM loss.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from ..data.types import Example, Table
from .numeric import N_FEATURES, column_key, numeric_features

# Cell kinds; padding shares kind 0 with None and is separated by cell_mask.
KIND_NONE, KIND_NUMERIC, KIND_STRING = 0, 1, 2

N_STRING_BUCKETS = 2048


class ColumnVocab:
    """Per-dataset vocabulary of (section, column) -> embedding id.

    Id 0 is reserved for unknown/padding; real columns start at 1. Built on
    the train split and saved with the checkpoint.
    """

    UNK = 0

    def __init__(self, keys: list[str]):
        self.index = {k: i + 1 for i, k in enumerate(sorted(set(keys)))}

    @classmethod
    def from_examples(cls, examples: list[Example]) -> "ColumnVocab":
        keys = [
            column_key(sec.name, col)
            for ex in examples
            for sec in ex.table.sections
            for col in sec.columns
        ]
        return cls(keys)

    def id(self, section_name: str, column_name: str) -> int:
        return self.index.get(column_key(section_name, column_name), self.UNK)

    def __len__(self) -> int:
        return len(self.index) + 1  # +1 for UNK/pad slot 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.index)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> "ColumnVocab":
        vocab = cls([])
        vocab.index = dict(d)
        return vocab


def string_bucket(value: str, n_buckets: int = N_STRING_BUCKETS) -> int:
    """Stable hash bucket (md5, not Python hash(), which is salted per run)."""
    digest = hashlib.md5(value.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def featurise_table(
    table: Table,
    vocab: ColumnVocab,
    stats: dict[str, list[float]],
    n_buckets: int = N_STRING_BUCKETS,
) -> dict[str, torch.Tensor]:
    """Table -> unbatched tensors [R, C(, F)].

    Rows are concatenated across sections (section identity travels in the
    section-scoped column ids); C is the widest row of this table. Returns
    col_id, kind, num_feats, str_bucket, is_key, cell_mask.
    """
    n_rows = sum(len(sec.rows) for sec in table.sections)
    n_cols = max(len(sec.columns) for sec in table.sections)
    col_id = torch.zeros(n_rows, n_cols, dtype=torch.long)
    kind = torch.zeros(n_rows, n_cols, dtype=torch.long)
    num_feats = torch.zeros(n_rows, n_cols, N_FEATURES)
    str_bucket = torch.zeros(n_rows, n_cols, dtype=torch.long)
    is_key = torch.zeros(n_rows, n_cols)
    cell_mask = torch.zeros(n_rows, n_cols, dtype=torch.bool)

    r = 0
    for sec in table.sections:
        for row in sec.rows:
            for c, (col, cell) in enumerate(zip(sec.columns, row)):
                col_id[r, c] = vocab.id(sec.name, col)
                cell_mask[r, c] = True
                if c == sec.key_column:
                    is_key[r, c] = 1.0
                if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    kind[r, c] = KIND_NUMERIC
                    mean, std = stats.get(column_key(sec.name, col), (0.0, 0.0))
                    num_feats[r, c] = torch.tensor(numeric_features(cell, mean, std))
                elif isinstance(cell, str):
                    kind[r, c] = KIND_STRING
                    str_bucket[r, c] = string_bucket(cell, n_buckets)
                # None stays KIND_NONE -> learned null vector
            r += 1
    return {
        "col_id": col_id,
        "kind": kind,
        "num_feats": num_feats,
        "str_bucket": str_bucket,
        "is_key": is_key,
        "cell_mask": cell_mask,
    }


class _QFormerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln_q = nn.LayerNorm(d)
        self.ln_x = nn.LayerNorm(d)
        self.ln_f = nn.LayerNorm(d)

    def forward(
        self, queries: torch.Tensor, rows: torch.Tensor, row_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.ln_q(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.ln_x(queries)
        queries = (
            queries
            + self.cross_attn(
                q, rows, rows, key_padding_mask=~row_mask, need_weights=False
            )[0]
        )
        return queries + self.ffn(self.ln_f(queries))


class TableEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_queries: int = 16,
        qformer_layers: int = 1,
        n_buckets: int = N_STRING_BUCKETS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.n_queries = n_queries
        self.col_emb = nn.Embedding(vocab_size, d)
        self.num_mlp = nn.Sequential(nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.str_emb = nn.Embedding(n_buckets, d)
        self.null_vec = nn.Parameter(torch.zeros(d))
        self.key_proj = nn.Linear(d, d)
        layer = nn.TransformerEncoderLayer(
            d,
            n_heads,
            dim_feedforward=2 * d,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.row_transformer = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False
        )
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.qformer = nn.ModuleList(
            _QFormerLayer(d, n_heads, dropout) for _ in range(qformer_layers)
        )
        self.aux_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """batch tensors are [B, R, C(, F)]; returns (soft [B, k, d],
        aux_logits [B, R, C])."""
        cell_mask = batch["cell_mask"]
        kind = batch["kind"]

        value = self.num_mlp(batch["num_feats"])
        value = torch.where((kind == KIND_NUMERIC).unsqueeze(-1), value, self.null_vec)
        value = torch.where(
            (kind == KIND_STRING).unsqueeze(-1), self.str_emb(batch["str_bucket"]), value
        )
        cell = self.col_emb(batch["col_id"]) + value
        cell = cell * cell_mask.unsqueeze(-1)

        counts = cell_mask.sum(-1, keepdim=True).clamp(min=1)
        rows = cell.sum(-2) / counts
        key_value = (value * batch["is_key"].unsqueeze(-1) * cell_mask.unsqueeze(-1)).sum(-2)
        rows = rows + self.key_proj(key_value)

        row_mask = cell_mask.any(-1)
        rows = self.row_transformer(rows, src_key_padding_mask=~row_mask)
        # Padded rows can carry garbage out of the transformer; zero them so
        # they cannot leak through the (masked) cross-attention as 0 * inf.
        rows = rows * row_mask.unsqueeze(-1)

        soft = self.queries.unsqueeze(0).expand(rows.shape[0], -1, -1)
        for layer in self.qformer:
            soft = layer(soft, rows, row_mask)

        row_ctx = rows.unsqueeze(2).expand(-1, -1, cell.shape[2], -1)
        aux_logits = self.aux_head(torch.cat([cell, row_ctx], dim=-1)).squeeze(-1)
        return soft, aux_logits


class Projector(nn.Module):
    """d -> LLM hidden size. 2-layer MLP by default; linear via mlp=False."""

    def __init__(self, d: int, hidden_size: int, mlp: bool = True):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, hidden_size))
        else:
            self.net = nn.Linear(d, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoftTableModel(nn.Module):
    """Encoder + projector: the full trainable module (LLM stays frozen)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        d: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_queries: int = 16,
        qformer_layers: int = 1,
        projector_mlp: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.encoder = TableEncoder(
            vocab_size,
            d=d,
            n_heads=n_heads,
            n_layers=n_layers,
            n_queries=n_queries,
            qformer_layers=qformer_layers,
            dropout=dropout,
        )
        self.projector = Projector(d, hidden_size, mlp=projector_mlp)

    @classmethod
    def from_config(cls, arch: dict, vocab_size: int, hidden_size: int) -> "SoftTableModel":
        return cls(
            vocab_size,
            hidden_size,
            d=int(arch.get("d", 256)),
            n_heads=int(arch.get("n_heads", 4)),
            n_layers=int(arch.get("n_layers", 2)),
            n_queries=int(arch.get("k", 16)),
            qformer_layers=int(arch.get("qformer_layers", 1)),
            projector_mlp=bool(arch.get("projector_mlp", True)),
            dropout=float(arch.get("dropout", 0.0)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        soft, aux_logits = self.encoder(batch)
        return self.projector(soft), aux_logits
