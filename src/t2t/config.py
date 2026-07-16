"""Config loading, hashing, and reproducibility utilities."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return cfg


def config_hash(cfg: dict) -> str:
    """Short stable hash of the full resolved config; keys order-insensitive."""
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:10]


def run_dir(cfg: dict, results_root: str | Path = "results") -> Path:
    return Path(results_root) / f"{cfg.get('name', 'run')}-{config_hash(cfg)}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
