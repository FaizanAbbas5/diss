"""E2E cleaned (Dusek et al. 2019): restaurant meaning representations.

Source CSVs from https://github.com/tuetschek/e2e-cleaning (cleaned-data/).
Each CSV row is (mr, ref); multiple rows share an MR, so we group the
references by MR and emit one Example per unique MR.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import requests

from .types import Example, Section, Table

_BASE = "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/"
_FILES = {
    "train": "train-fixed.no-ol.csv",
    "validation": "devel-fixed.no-ol.csv",
    "test": "test-fixed.csv",
}

# MR attributes look like: name[The Vaults], customer rating[5 out of 5]
_MR_RE = re.compile(r"([A-Za-z ]+?)\[([^\]]*)\]")


def download(split: str, data_dir: str | Path = "data/e2e") -> Path:
    dest = Path(data_dir) / _FILES[split]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_BASE + _FILES[split], timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def parse_mr(mr: str) -> list[tuple[str, str]]:
    pairs = [(k.strip(), v.strip()) for k, v in _MR_RE.findall(mr)]
    if not pairs:
        raise ValueError(f"Unparseable MR: {mr!r}")
    return pairs


def load(split: str = "validation", data_dir: str | Path = "data/e2e") -> list[Example]:
    path = download(split, data_dir)
    grouped: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        if "mr" not in cols or "ref" not in cols:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            mr = (row[cols["mr"]] or "").strip()
            ref = (row[cols["ref"]] or "").strip()
            if mr:
                grouped.setdefault(mr, []).append(ref)

    examples = []
    for i, (mr, refs) in enumerate(grouped.items()):
        pairs = parse_mr(mr)
        section = Section(
            name="restaurant",
            columns=[k for k, _ in pairs],
            rows=[[v for _, v in pairs]],
        )
        examples.append(
            Example(
                id=f"e2e-{split}-{i:04d}",
                table=Table([section]),
                references=[r for r in refs if r],
                meta={"mr": mr},
            )
        )
    return examples
