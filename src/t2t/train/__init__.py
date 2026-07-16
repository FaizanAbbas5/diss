"""Training for the Arm-2 encoder + projector (frozen LLM)."""
from __future__ import annotations

from .collate import Arm2Collator
from .dataset import Arm2Dataset

__all__ = ["Arm2Collator", "Arm2Dataset", "train"]


def train(cfg: dict, results_dir="results"):
    from .loop import train as _train

    return _train(cfg, results_dir)
