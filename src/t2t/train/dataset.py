"""Training items for Arm 2: featurised table + token plan + aux labels.

One item per (example, reference) pair, capped by max_refs_per_example
(default 1: the first reference, deterministic). With filter_references on,
the reference is reduced to its grounded sentences first (t2t.encode.align);
items whose filtered reference is empty are dropped and counted.

Token layout per item (assembled by the collator / training loop):
    prefix | SOFT(k) | [table tokens, augmentation only] | suffix | reference
The prompt is split around the {table} slot once per dataset via the
sentinel machinery in t2t.generate.soft, so training and generation use the
identical split.
"""
from __future__ import annotations

from torch.utils.data import Dataset

from ..data.types import Example
from ..encode import ColumnVocab, alignment_labels, featurise_table, filter_reference
from ..facts import GameFacts
from ..generate.soft import split_prompt_ids
from ..serialise import to_markdown


class Arm2Dataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer,
        prompt_name: str,
        vocab: ColumnVocab,
        stats: dict[str, list[float]],
        variant: str = "replacement",
        filter_refs: bool = False,
        max_refs_per_example: int = 1,
        max_ref_tokens: int = 512,
        serialise=to_markdown,
    ):
        if variant not in ("replacement", "augmentation"):
            raise ValueError(f"Cannot train variant {variant!r}")
        self.prefix_ids, self.suffix_ids = split_prompt_ids(
            tokenizer, prompt_name, add_generation_prompt=True
        )
        self.items: list[dict] = []
        self.n_dropped = 0
        eos = tokenizer.eos_token_id
        for ex in examples:
            facts = GameFacts(ex.table)
            features = featurise_table(ex.table, vocab, stats)
            table_ids = (
                tokenizer.encode(serialise(ex.table), add_special_tokens=False)
                if variant == "augmentation"
                else None
            )
            for ref in ex.references[:max_refs_per_example]:
                if filter_refs:
                    ref = filter_reference(ref, ex.table, facts)
                    if not ref:
                        self.n_dropped += 1
                        continue
                ref_ids = tokenizer.encode(ref, add_special_tokens=False)
                if len(ref_ids) > max_ref_tokens:
                    # Truncated references get no EOS: don't teach the model
                    # to stop mid-summary.
                    ref_ids = ref_ids[:max_ref_tokens]
                else:
                    ref_ids = ref_ids + [eos]
                self.items.append(
                    {
                        **features,
                        "aux_labels": alignment_labels(ex.table, ref, facts),
                        "prefix_ids": self.prefix_ids,
                        "table_ids": table_ids,
                        "suffix_ids": self.suffix_ids,
                        "ref_ids": ref_ids,
                        "id": ex.id,
                    }
                )
        if self.n_dropped:
            print(
                f"filter_references: dropped {self.n_dropped} items with no "
                f"grounded sentence ({len(self.items)} remain)"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]
