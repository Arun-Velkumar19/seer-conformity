"""Render the subject's system prompt and per-step moderator messages for a cell.

All text comes from prompts/ and confederates/ (calibration) or from a
scenario file in scenarios/ (Study 1); nothing experimental is hard-coded
here. Rendering is deterministic in (scenario, pressure level, cell, seed).
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


def and_list(names: list[str]) -> str:
    """"A, B, and C" (two names: "A and B")."""
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


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
    def __init__(self, cfg: dict, scenario_id: str | None = None):
        """`scenario_id` picks one of a study config's scenario files; calibration
        configs have a single built-in scenario (scenario_id None)."""
        p = cfg["paths"]
        self.id = scenario_id
        if "scenarios" in cfg and scenario_id not in cfg["scenarios"]:
            raise ValueError(f"unknown scenario {scenario_id!r}; defined: {sorted(cfg['scenarios'])}")
        self.pressure_level = cfg["pressure_level"]
        self.level = cfg["level"]
        self.n_rounds = 3 if self.level["isolation_round"] else 2

        self.rules = _COMMENT.sub("", _read(p["rules"]))
        self.minion_rule = _read(p["minion_rule"])
        self.roles = {k: _read(v) for k, v in p["roles"].items()}
        self.visibility = {k: _read(v) for k, v in p["visibility"].items()}
        self.turns = _yaml(p["turns"])
        self.no_speaker_prefix = self.turns["no_speaker_prefix"]

        if scenario_id is None:
            sc = cfg["scenario"]
            self.subject = sc["subject"]
            self.confederates = sc["confederates"]
            self.silent = sc["silent"]
            self.names: dict[str, str] = {}   # calibration templates have the names written in
            bank = _yaml(p["arguments"])
            self.target = bank["target"]
            self.suggested_target = bank["suggested_target"]
            self.bank = {a["id"]: a for a in bank["arguments"]}          # every level
            self.declarations = {d["id"]: d["text"] for d in bank["vote_declarations"]}
            self.filler = {f["pairs_with"]: f["text"] for f in _yaml(p["filler"])["filler"]}
        else:
            self._load_scenario_file(cfg["scenarios"][scenario_id])
        self.candidates = self.confederates + self.silent
        self.arguments = {i: a for i, a in self.bank.items()            # this level
                          if a.get("level", 0) <= self.pressure_level}
        claims = [i for i, a in self.arguments.items() if a.get("counterclaim") == "claim"]
        self.claim_id = claims[0] if self.level["counterclaim"] and claims else None

        self._validate()

    def _load_scenario_file(self, path: str) -> None:
        """A Study 1 scenario (scenarios/README.md): cast, evening chat, final
        messages, and level-4 texts with their fillers inline."""
        data = _yaml(path)
        if data["id"] != self.id:
            raise ValueError(f"{path} has id {data['id']!r}, expected {self.id!r}")
        cast = data["cast"]
        self.subject = cast["subject"]
        self.confederates = list(cast["confederates"])
        self.silent = [cast["werewolf"], cast["alternative"]]
        self.target, self.suggested_target = cast["werewolf"], cast["alternative"]
        self.cast = cast
        self.names = {"subject": cast["subject"], "claimant": cast["claimant"], "werewolf": cast["werewolf"],
                      "alternative": cast["alternative"], "evening": data["evening"]}
        self.final_messages = data["final_message"]
        self.bank, self.filler = {}, {}
        for a in data["arguments"]:
            a = dict(a)
            if "both" in a:
                a["seer"] = a["villager"] = a.pop("both")
            self.filler[a["id"]] = a.pop("filler")
            self.bank[a["id"]] = a
        self.declarations = {d["id"]: d["text"] for d in data["vote_declarations"]}
        self.filler.update({d["id"]: d["filler"] for d in data["vote_declarations"]})
        if len(self.confederates) != len(self.level["confederates"]):
            raise ValueError(f"{path}: pressure level {self.pressure_level} needs "
                             f"{len(self.level['confederates'])} confederates")

    def _validate(self) -> None:
        if set(self.filler) != set(self.bank) | set(self.declarations):
            raise ValueError("every argument and vote declaration needs exactly one paired filler")
        for t in (self.target, self.suggested_target):
            if t not in self.candidates:
                raise ValueError(f"{t} is not a vote candidate")
        for r in range(1, self.n_rounds + 1):
            if r > 1 and self.level["counterclaim"] and not self.round_ids(r):
                continue  # filled entirely by follow-ups and support lines (Study 1 scenario files)
            if len(self.round_ids(r)) != len(self.confederates):
                raise ValueError(f"round {r} needs one argument per confederate slot at level {self.pressure_level}")
        if self.level["vote_declarations"] and len(self.declarations) < len(self.confederates):
            raise ValueError("need one vote declaration per confederate slot")
        if self.level["counterclaim"]:
            if self.claim_id is None:
                raise ValueError("counterclaim level needs a `counterclaim: claim` entry")
            for r in range(2, self.n_rounds + 1):
                if len(self.support_ids(r)) != len(self.confederates) - 1:
                    raise ValueError(f"round {r} needs one counterclaim support line per other confederate")
                if len(self._cc_ids(r, "followup")) != 1:
                    raise ValueError(f"round {r} needs one counterclaimer follow-up")
            for i, a in self.arguments.items():
                if a.get("counterclaim") == "replaces" and a["replaces"] not in self.round_ids(a["round"]):
                    raise ValueError(f"{i} replaces {a['replaces']!r}, which is not a round-{a['round']} argument")

    def round_ids(self, round_no: int) -> list[str]:
        """Regular arguments for a round (counterclaim entries are placed separately)."""
        return [i for i, a in self.arguments.items() if a["round"] == round_no and "counterclaim" not in a]

    def _cc_ids(self, round_no: int, kind: str) -> list[str]:
        return [i for i, a in self.arguments.items() if a["round"] == round_no and a.get("counterclaim") == kind]

    def support_ids(self, round_no: int) -> list[str]:
        return self._cc_ids(round_no, "support")

    def slot_order(self, seed: int) -> dict:
        """Argument ids per slot, per round, plus vote-declaration ids per slot
        ("votes", empty below level 1) and the counterclaimer's name ("counterclaimer",
        level 4 only). Depends only on the seed and the pressure level, never the cell.
        Draws happen in level order, so each level keeps the lower levels' slots: at
        level 4 the counterclaimer's slot takes the claim in round 1 and its
        follow-up in rounds 2-3, while the other slots get support lines. Arguments that
        undercut the counterclaim are swapped for their `replaces` entry."""
        rng = random.Random(seed)
        order: dict = {}
        for r in range(1, self.n_rounds + 1):
            ids = sorted(self.round_ids(r))
            rng.shuffle(ids)
            order[r] = ids
        order["votes"] = []
        if self.level["vote_declarations"]:
            ids = sorted(self.declarations)
            rng.shuffle(ids)
            order["votes"] = ids[:len(self.confederates)]
        if self.level["counterclaim"]:
            k = rng.randrange(len(self.confederates))
            order["counterclaimer"] = self.confederates[k]
            order[1][k] = self.claim_id
            swaps = {a["replaces"]: i for i, a in self.arguments.items() if a.get("counterclaim") == "replaces"}
            order[1] = [swaps.get(x, x) for x in order[1]]
            for r in range(2, self.n_rounds + 1):
                support = sorted(self.support_ids(r))
                rng.shuffle(support)
                rest = iter(support)
                followup = self._cc_ids(r, "followup")[0]
                order[r] = [followup if j == k else next(rest) for j in range(len(self.confederates))]
        return order

    def system_prompt(self, cell: Cell) -> str:
        players = self.confederates + self.silent
        rules = self.rules if self.level["minion_rule"] else self.rules.replace("{minion_rule}\n", "")
        role_block = self.roles[cell.role]
        if self.names:
            role_block = fill(role_block, final_message=self.final_messages[cell.role], **self.names)
        return fill(
            rules,
            **self.names,
            minion_rule=self.minion_rule,
            role_block=role_block,
            player_count=str(len(players) + 2),  # + Quinn and Avery
            players=", ".join(players),
            living=and_list(players),
            greeters=and_list(self.confederates),
        )

    def _line(self, cell: Cell, speaker: str, pressure_text: str, filler_text: str) -> str:
        if cell.pressure == "none":
            return f"{speaker}: {filler_text.strip()}"
        if cell.pressure == "no_speaker":
            return f"{self.no_speaker_prefix}{pressure_text.strip()}"
        return f"{speaker}: {pressure_text.strip()}"

    def round_messages(self, cell: Cell, ids: list[str], vote_ids: list[str] = ()) -> str:
        lines = [self._line(cell, s, self.arguments[i][cell.role], self.filler[i])
                 for s, i in zip(self.confederates, ids)]
        lines += [self._line(cell, s, self.declarations[i], self.filler[i])
                  for s, i in zip(self.confederates, vote_ids)]
        return "\n\n".join(lines)

    def steps(self, cell: Cell, seed: int) -> list[Step]:
        order = self.slot_order(seed)
        values = {
            **self.names,
            "visibility_block": self.visibility[cell.visibility],
            "opening_requirement": fill(self.turns["opening_requirement"][cell.role], **self.names),
        }
        out = []
        for s in self.turns["steps" if self.n_rounds == 2 else "steps_three_rounds"]:
            kv = dict(values)
            if "round" in s:
                last = s["round"] == self.n_rounds
                kv["round_messages"] = self.round_messages(cell, order[s["round"]], order["votes"] if last else ())
            needed = {k: v for k, v in kv.items() if "{" + k + "}" in s["text"]}
            out.append(Step(s["step"], fill(s["text"], **needed), s.get("tool")))
        return out

    def diagnostic_step(self) -> Step:
        s = self.turns["diagnostic"][0]
        return Step(s["step"], fill(s["text"]), s.get("tool"))
