"""RotoWire NBA box scores (Wiseman et al. 2017).

Downloads rotowire.tar.bz2 from harvardnlp/boxscore-data and parses each
game into a canonical Table with 'teams' and 'players' sections.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import requests

from .types import Cell, Example, Section, Table

_URL = "https://github.com/harvardnlp/boxscore-data/raw/master/rotowire.tar.bz2"
_SPLITS = {"train": "train.json", "validation": "valid.json", "test": "test.json"}

# Preferred column order; any remaining box-score keys are appended sorted.
_PLAYER_COLS = [
    "PLAYER_NAME", "TEAM_CITY", "START_POSITION", "MIN", "PTS",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
    "STL", "BLK", "TO", "PF",
]
_SKIP_PLAYER_KEYS = {"FIRST_NAME", "SECOND_NAME"}


def _to_cell(value: str) -> Cell:
    if value in ("N/A", ""):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def download(data_dir: str | Path = "data/rotowire") -> Path:
    data_dir = Path(data_dir)
    if not all((data_dir / f).exists() for f in _SPLITS.values()):
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / "rotowire.tar.bz2"
        if not archive.exists():
            resp = requests.get(_URL, timeout=300)
            resp.raise_for_status()
            archive.write_bytes(resp.content)
        with tarfile.open(archive, "r:bz2") as tar:
            for member in tar.getmembers():
                fname = Path(member.name).name
                if member.isfile() and fname in _SPLITS.values():
                    member.name = fname  # flatten the 'rotowire/' prefix
                    tar.extract(member, data_dir)
    return data_dir


# Redundant with the composite TEAM column built from city + name.
_SKIP_TEAM_KEYS = {"TEAM-CITY", "TEAM-NAME"}


def _team_section(entry: dict) -> Section:
    keys = sorted((set(entry["home_line"]) | set(entry["vis_line"])) - _SKIP_TEAM_KEYS)
    columns = ["TEAM"] + keys
    rows = []
    for prefix in ("home", "vis"):
        line = entry[f"{prefix}_line"]
        name = f"{entry[f'{prefix}_city']} {entry[f'{prefix}_name']}"
        rows.append([name] + [_to_cell(str(line.get(k, "N/A"))) for k in keys])
    return Section(name="teams", columns=columns, rows=rows)


# Identity columns that don't count as stats when deciding if a player played.
_PLAYER_ID_COLS = {"PLAYER_NAME", "TEAM_CITY", "START_POSITION"}


def _player_section(entry: dict) -> Section:
    box = entry["box_score"]
    present = [k for k in box if k not in _SKIP_PLAYER_KEYS]
    columns = [c for c in _PLAYER_COLS if c in present]
    columns += sorted(k for k in present if k not in columns)
    player_ids = sorted(box["PLAYER_NAME"], key=int)
    rows = [
        [_to_cell(str(box[c].get(pid, "N/A"))) for c in columns]
        for pid in player_ids
    ]
    # Drop players with no stats at all (DNP rows are pure N/A noise in the
    # prompt) and columns that are empty for every remaining player.
    stat_idx = [i for i, c in enumerate(columns) if c not in _PLAYER_ID_COLS]
    rows = [r for r in rows if any(r[i] is not None for i in stat_idx)]
    keep = [
        i
        for i, c in enumerate(columns)
        if c == "PLAYER_NAME" or any(r[i] is not None for r in rows)
    ]
    columns = [columns[i] for i in keep]
    rows = [[r[i] for i in keep] for r in rows]
    return Section(name="players", columns=columns, rows=rows)


def parse_entry(entry: dict) -> Table:
    """Parse one raw RotoWire game dict into a canonical Table."""
    return Table([_team_section(entry), _player_section(entry)])


def load(split: str = "validation", data_dir: str | Path = "data/rotowire") -> list[Example]:
    data_dir = download(data_dir)
    with open(data_dir / _SPLITS[split], encoding="utf-8") as f:
        entries = json.load(f)

    examples = []
    for i, entry in enumerate(entries):
        table = parse_entry(entry)
        summary = " ".join(entry["summary"]).strip()
        examples.append(
            Example(
                id=f"rotowire-{split}-{i:04d}",
                table=table,
                references=[summary] if summary else [],
                meta={"home": entry["home_name"], "vis": entry["vis_name"]},
            )
        )
    return examples
