"""Render the subject's system prompt and per-step moderator messages for a cell.

All text comes from prompts/ and confederates/; nothing experimental is
hard-coded here. Rendering is deterministic in (cell, seed).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import yaml

from .config import REPO_ROOT, Cell

_COMMENT = re.compile(r"<!--.*?-->\s*", re.S)
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def _yaml(path: str) -> dict:
    return yaml.safe_load(_read(path))


def fill(template: str, **values: str) -> str:
    """Replace {key} placeholders; fail loudly on any left unfilled."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value.strip())
    leftover = _PLACEHOLDER.findall(template)
    if leftover:
        raise ValueError(f"unfilled placeholders: {leftover}")
    return template.strip()


@dataclass(frozen=True)
class Step:
    name: str
    text: str
    tool: str | None  # tool the subject must call at this step, if any


class Scenario:
    def __init__(self, cfg: dict):
        p = cfg["paths"]
        sc = cfg["scenario"]
        self.subject = sc["subject"]
        self.confederates = sc["confederates"]
        self.candidates = sc["candidates"]
        self.no_speaker_prefix = sc["attribution"]["no_speaker"]

        self.rules = _COMMENT.sub("", _read(p["rules"]))
        self.roles = {k: _read(v) for k, v in p["roles"].items()}
        self.visibility = {k: _read(v) for k, v in p["visibility"].items()}
        self.turns = _yaml(p["turns"])

        bank = _yaml(p["arguments"])
        self.target = bank["target"]
        self.suggested_target = bank["suggested_target"]
        self.arguments = {a["id"]: a for a in bank["arguments"]}
        self.filler = {f["pairs_with"]: f["text"] for f in _yaml(p["filler"])["filler"]}

        self._validate()

    def _validate(self) -> None:
        if set(self.filler) != set(self.arguments):
            raise ValueError("every argument needs exactly one paired filler")
        for t in (self.target, self.suggested_target):
            if t not in self.candidates:
                raise ValueError(f"{t} is not a vote candidate")
        for r in (1, 2):
            if len(self.round_ids(r)) != len(self.confederates):
                raise ValueError(f"round {r} needs one argument per confederate slot")

    def round_ids(self, round_no: int) -> list[str]:
        return [i for i, a in self.arguments.items() if a["round"] == round_no]

    def slot_order(self, seed: int) -> dict[int, list[str]]:
        """Argument ids per slot, per round. Depends only on the seed, never the cell."""
        rng = random.Random(seed)
        order = {}
        for r in (1, 2):
            ids = sorted(self.round_ids(r))
            rng.shuffle(ids)
            order[r] = ids
        return order

    def system_prompt(self, cell: Cell) -> str:
        return fill(self.rules, role_block=self.roles[cell.role])

    def round_messages(self, cell: Cell, ids: list[str]) -> str:
        lines = []
        for speaker, arg_id in zip(self.confederates, ids):
            if cell.pressure == "none":
                lines.append(f"{speaker}: {self.filler[arg_id].strip()}")
            elif cell.pressure == "no_speaker":
                lines.append(f"{self.no_speaker_prefix}{self.arguments[arg_id][cell.role].strip()}")
            else:
                lines.append(f"{speaker}: {self.arguments[arg_id][cell.role].strip()}")
        return "\n\n".join(lines)

    def steps(self, cell: Cell, seed: int) -> list[Step]:
        order = self.slot_order(seed)
        values = {
            "visibility_block": self.visibility[cell.visibility],
            "opening_requirement": self.turns["opening_requirement"][cell.role],
        }
        rounds = {"pressure_round_1": 1, "pressure_round_2_and_final_vote": 2}
        out = []
        for s in self.turns["steps"]:
            kv = dict(values)
            if s["step"] in rounds:
                kv["round_messages"] = self.round_messages(cell, order[rounds[s["step"]]])
            needed = {k: v for k, v in kv.items() if "{" + k + "}" in s["text"]}
            out.append(Step(s["step"], fill(s["text"], **needed), s.get("tool")))
        return out

    def diagnostic_step(self) -> Step:
        s = self.turns["diagnostic"][0]
        return Step(s["step"], fill(s["text"]), s.get("tool"))
