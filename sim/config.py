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


def _read_yaml(path: str | Path) -> dict:
    with open(REPO_ROOT / path) as f:
        return yaml.safe_load(f)


def resolve_level(levels: dict, level: int) -> dict:
    """Pressure levels are cumulative: merge level 0 up to `level`."""
    if level not in levels:
        raise ValueError(f"unknown pressure level {level!r}; defined: {sorted(levels)}")
    merged: dict = {}
    for k in sorted(levels):
        if k <= level:
            merged.update(levels[k])
    return merged


def level_settings(cfg: dict, level: int) -> dict:
    """The resolved settings of any pressure level, not just the config's own."""
    return resolve_level(_read_yaml(cfg["pressure_levels"])["levels"], level)


def load_config(path: str | Path, pressure_level: int | None = None) -> dict:
    """Load an experiment config. `base:` names a config whose top-level keys this
    one overrides. `pressure_level` overrides the config's own."""
    cfg = _read_yaml(path)
    if "base" in cfg:
        cfg = {**_read_yaml(cfg.pop("base")), **cfg}
    if pressure_level is not None:
        cfg["pressure_level"] = pressure_level
    if cfg.get("pressure_level") is None:
        raise ValueError(f"{path} needs a pressure level (--pressure-level)")
    cfg["level"] = level_settings(cfg, cfg["pressure_level"])
    sc = cfg["scenario"]
    sc["confederates"] = list(cfg["level"]["confederates"])
    sc["candidates"] = sc["confederates"] + list(sc["silent"])
    if len(cfg["seeds"]) < cfg["runs_per_cell"]:
        raise ValueError("need at least one seed per run index")
    unknown = set(cfg["factors"]["pressure"]) - set(PRESSURES)
    if unknown:
        raise ValueError(f"unknown pressure levels: {sorted(unknown)}")
    registry = _read_yaml(cfg["model_registry"])["models"]
    cfg["model_specs"] = {}
    for key in cfg["models"] + cfg.get("optional_models", []):
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
