"""Table encoder: shapes, masks, and the permutation-invariance property
the architecture claims (shuffling player rows must not change the soft
tokens; no positional encoding over rows anywhere)."""
import random

import pytest
import torch

from t2t.data.types import Example, Section, Table
from t2t.encode import (
    ColumnVocab,
    SoftTableModel,
    compute_column_stats,
    featurise_table,
)
from t2t.encode.table_encoder import KIND_NONE, KIND_NUMERIC, KIND_STRING

PLAYER_COLS = ["PLAYER_NAME", "TEAM_CITY", "PTS", "REB", "AST"]
PLAYERS = [
    ["Kyle Lowry", "Toronto", 24, 4, 8],
    ["DeMar DeRozan", "Toronto", 14, 5, 3],
    ["Robert Covington", "Philadelphia", 20, 6, 1],
    ["Jahlil Okafor", "Philadelphia", None, 8, 1],
]


def game(players=PLAYERS):
    return Table(
        [
            Section(
                "teams",
                ["TEAM", "TEAM-PTS"],
                [["Toronto Raptors", 122], ["Philadelphia 76ers", 95]],
            ),
            Section("players", PLAYER_COLS, [list(p) for p in players]),
        ]
    )


@pytest.fixture(scope="module")
def setup():
    examples = [Example(id="g0", table=game())]
    vocab = ColumnVocab.from_examples(examples)
    stats = compute_column_stats(examples)
    torch.manual_seed(0)
    net = SoftTableModel(vocab_size=len(vocab), hidden_size=96, d=64, n_queries=8)
    net.eval()
    return vocab, stats, net


def _forward(net, table, vocab, stats):
    feats = featurise_table(table, vocab, stats)
    batch = {k: t.unsqueeze(0) for k, t in feats.items()}
    with torch.no_grad():
        return net(batch)


def test_featurise_shapes_kinds_and_masks(setup):
    vocab, stats, _ = setup
    feats = featurise_table(game(), vocab, stats)
    assert feats["col_id"].shape == (6, 5)  # 2 team + 4 player rows, widest row 5
    assert feats["cell_mask"][0].tolist() == [True, True, False, False, False]
    assert feats["kind"][0, 0] == KIND_STRING and feats["kind"][0, 1] == KIND_NUMERIC
    assert feats["kind"][5, 2] == KIND_NONE  # Okafor's None PTS
    assert feats["is_key"][:, 0].sum() == 6  # key column marked in every row
    assert (feats["col_id"][2:, :] == feats["col_id"][2, 0]).sum() == 4  # names share id


def test_soft_and_aux_shapes(setup):
    vocab, stats, net = setup
    soft, aux = _forward(net, game(), vocab, stats)
    assert soft.shape == (1, 8, 96)
    assert aux.shape == (1, 6, 5)
    assert torch.isfinite(soft).all() and torch.isfinite(aux).all()


def test_row_permutation_invariance(setup):
    vocab, stats, net = setup
    soft_a, _ = _forward(net, game(), vocab, stats)
    shuffled = list(PLAYERS)
    random.Random(7).shuffle(shuffled)
    assert shuffled != PLAYERS
    soft_b, _ = _forward(net, game(shuffled), vocab, stats)
    assert torch.allclose(soft_a, soft_b, atol=1e-5)


def test_batch_padding_does_not_change_output(setup):
    vocab, stats, net = setup
    big = featurise_table(game(), vocab, stats)
    small = featurise_table(
        Table([Section("teams", ["TEAM", "TEAM-PTS"], [["A Ants", 1], ["B Bees", 2]])]),
        vocab,
        stats,
    )
    r = big["col_id"].shape[0] - small["col_id"].shape[0]
    c = big["col_id"].shape[1] - small["col_id"].shape[1]
    batch = {}
    for key in big:
        pad = [0, 0, 0, c, 0, r] if small[key].dim() == 3 else [0, c, 0, r]
        batch[key] = torch.stack(
            [big[key], torch.nn.functional.pad(small[key], pad)]
        )
    with torch.no_grad():
        soft_batched, _ = net(batch)
    soft_alone, _ = _forward(net, game(), vocab, stats)
    assert torch.allclose(soft_batched[0:1], soft_alone, atol=1e-5)
    assert torch.isfinite(soft_batched[1]).all()  # padded rows must not NaN


def test_gradients_reach_all_embedding_tables(setup):
    vocab, stats, net = setup
    net.train()
    feats = featurise_table(game(), vocab, stats)
    soft, aux = net({k: t.unsqueeze(0) for k, t in feats.items()})
    (soft.sum() + aux.sum()).backward()
    enc = net.encoder
    assert enc.col_emb.weight.grad is not None and enc.col_emb.weight.grad.abs().sum() > 0
    assert enc.str_emb.weight.grad is not None
    assert enc.num_mlp[0].weight.grad is not None
    assert enc.queries.grad is not None
    net.zero_grad(set_to_none=True)
    net.eval()


def test_unknown_columns_map_to_unk_id(setup):
    vocab, _, _ = setup
    assert vocab.id("nonexistent", "col") == ColumnVocab.UNK
    assert vocab.id("teams", "TEAM") != ColumnVocab.UNK


def test_vocab_roundtrip(setup):
    vocab, _, _ = setup
    again = ColumnVocab.from_dict(vocab.to_dict())
    assert again.index == vocab.index and len(again) == len(vocab)
