"""Load the experiment config and expand it into cells."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

PRESSURES = ("none", "no_speaker", "peers")


@dataclass(frozen=True)
class Cell:
    role: str
    visibility: str
    pressure: str

    @property
    def id(self) -> str:
        return f"{self.role}-{self.visibility}-{self.pressure}"


def load_config(path: str | Path) -> dict:
    with open(REPO_ROOT / path) as f:
        cfg = yaml.safe_load(f)
    if len(cfg["seeds"]) < cfg["runs_per_cell"]:
        raise ValueError("need at least one seed per run index")
    unknown = set(cfg["factors"]["pressure"]) - set(PRESSURES)
    if unknown:
        raise ValueError(f"unknown pressure levels: {sorted(unknown)}")
    return cfg


def model_id(cfg: dict) -> str:
    m = cfg["model"]
    return os.environ.get(m["env_var"]) or m["default"]


def cells(cfg: dict) -> list[Cell]:
    f = cfg["factors"]
    return [Cell(r, v, p) for r, v, p in itertools.product(f["role"], f["visibility"], f["pressure"])]
