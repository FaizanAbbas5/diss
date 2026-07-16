"""Collate masks: labels only on reference tokens, attention over all real
positions including the soft slots, padding inert everywhere."""
import torch

from t2t.data.types import Section, Table
from t2t.encode import ColumnVocab, alignment_labels, featurise_table
from t2t.train.collate import Arm2Collator

PAD = 0
K = 4


def _item(prefix, suffix, ref, table_ids=None, n_cells=3):
    table = Table([Section("s", [f"c{i}" for i in range(n_cells)], [["v"] * n_cells])])
    vocab = ColumnVocab.from_examples([])
    feats = featurise_table(table, vocab, {})
    return {
        **feats,
        "aux_labels": alignment_labels(table, ""),
        "prefix_ids": prefix,
        "table_ids": table_ids,
        "suffix_ids": suffix,
        "ref_ids": ref,
        "id": "x",
    }


def test_layout_labels_and_masks_replacement():
    item = _item(prefix=[11, 12], suffix=[31], ref=[41, 42, 43])
    batch = Arm2Collator(k=K, pad_token_id=PAD)([item])
    total = 2 + K + 1 + 3
    assert batch["input_ids"].shape == (1, total)
    assert batch["soft_pos"][0].tolist() == [2, 3, 4, 5]
    assert (batch["input_ids"][0, batch["soft_pos"][0]] == PAD).all()
    assert batch["attention_mask"][0].tolist() == [1] * total
    # labels: -100 on prompt+soft+suffix, ref ids on ref positions
    assert batch["labels"][0].tolist() == [-100] * (total - 3) + [41, 42, 43]


def test_augmentation_places_table_tokens_after_soft():
    item = _item(prefix=[11], suffix=[31], ref=[41], table_ids=[21, 22])
    batch = Arm2Collator(k=K, pad_token_id=PAD)([item])
    ids = batch["input_ids"][0].tolist()
    assert ids == [11] + [PAD] * K + [21, 22, 31, 41]
    assert batch["labels"][0].tolist() == [-100] * (1 + K + 3) + [41]


def test_batch_padding_is_inert():
    long = _item(prefix=[11, 12], suffix=[31], ref=[41, 42, 43])
    short = _item(prefix=[13], suffix=[32], ref=[44], n_cells=2)
    batch = Arm2Collator(k=K, pad_token_id=PAD)([long, short])
    t = batch["input_ids"].shape[1]
    assert t == 2 + K + 1 + 3
    short_len = 1 + K + 1 + 1
    assert batch["attention_mask"][1].tolist() == [1] * short_len + [0] * (t - short_len)
    assert (batch["input_ids"][1, short_len:] == PAD).all()
    assert (batch["labels"][1, short_len:] == -100).all()
    # encoder tensors padded to widest table, mask says which cells are real
    assert batch["cell_mask"].shape == (2, 1, 3)
    assert batch["cell_mask"][1].tolist() == [[True, True, False]]
    assert (batch["aux_labels"][1, :, 2:] == 0).all()


def test_soft_positions_respect_prefix_length_per_item():
    a = _item(prefix=[1, 2, 3], suffix=[9], ref=[5])
    b = _item(prefix=[1], suffix=[9], ref=[5])
    batch = Arm2Collator(k=K, pad_token_id=PAD)([a, b])
    assert batch["soft_pos"][0, 0].item() == 3
    assert batch["soft_pos"][1, 0].item() == 1
