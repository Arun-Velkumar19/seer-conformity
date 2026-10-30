"""Load the experiment config and model registry, and expand cells."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

PRESSURES = ("none", "no_speaker", "peers")
MODEL_FIELDS = ("provider", "model", "effort", "max_output_tokens")


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
    with open(REPO_ROOT / cfg["model_registry"]) as f:
        registry = yaml.safe_load(f)["models"]
    cfg["model_specs"] = {}
    for key in cfg["models"]:
        if key not in registry:
            raise ValueError(f"model {key!r} is not in {cfg['model_registry']}")
        missing = [k for k in MODEL_FIELDS if registry[key].get(k) is None]
        if missing:
            raise ValueError(f"model {key!r} is missing {missing} (effort must be explicit)")
        cfg["model_specs"][key] = {"key": key, **registry[key]}
    return cfg


def cells(cfg: dict) -> list[Cell]:
    f = cfg["factors"]
    return [Cell(r, v, p) for r, v, p in itertools.product(f["role"], f["visibility"], f["pressure"])]
