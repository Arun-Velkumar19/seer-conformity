"""Run trials and write one JSONL record per run.

Per run (SPEC v2, Sequence per run):
  1. private_vote tool call
  2. opening statement            (visibility announced in this message)
  3. pressure round 1 + reply
  4. pressure round 2 + final vote tool call
  (5. private belief, diagnostic arm only)
At pressure level 3 a reply to round 2 and a round 3 come before the vote
(prompts/turns.yaml, steps_three_rounds).

The runner is provider-neutral (sim/providers). Vote steps allow tools
("auto") and rely on strict schemas plus the prompt's instruction, since
current models reject forced tool choice; if no vote arrives, one fixed
reminder is sent and counted. Text steps disallow tools ("none").
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import labeling
from .config import Cell, cells, planned_runs, scenario_ids
from .providers import ProviderError, open_session
from .scenario import Scenario, Step
from .tools import vote_tools

SCHEMA_VERSION = 7  # 4: pressure_level, confederates, replies; 5: counterclaimer; 6: per-call raw usage, latency, retries;
                    # 7: scenario, cast, target, alternative
TEMPERATURE_NOTE = "not sent (provider default)"
REMINDER = "Please record your vote now using the {tool} tool."


class TrialError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Trial:
    def __init__(self, cfg: dict, scenario: Scenario, spec: dict, cell: Cell, run_index: int, clients: dict):
        self.cfg = cfg
        self.sc = scenario
        self.spec = spec
        self.cell = cell
        self.run_index = run_index
        self.seed = cfg["seeds"][run_index]
        self.system = scenario.system_prompt(cell)
        if spec["provider"] == "fake":
            # The fake model varies by run; real models only see the prompts.
            level = scenario.pressure_level
            spec = {**spec, "salt": f"{cell.id}:{self.seed}" + (f":L{level}" if level else "")}
            if scenario.id:  # calibration dry runs stay as they were
                spec.update(salt=f"{spec['salt']}:{scenario.id}",
                            targets=(scenario.target, scenario.suggested_target, scenario.confederates[0]))
        self.session = open_session(spec, self.system, vote_tools(scenario.candidates), clients)
        self.transcript: list[dict] = []   # provider-neutral view, for reading and checks
        self.usage: list[dict] = []
        self.reminders = 0

    def _send(self, items: list[dict], tool: str | None):
        self.transcript.append({"role": "user", "items": items})
        turn = self.session.send(items, "auto" if tool else "none")
        self.usage.append(turn.usage)
        self.transcript.append({
            "role": "assistant",
            "text": turn.text,
            "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in turn.tool_calls],
            "stop": turn.stop,
        })
        if turn.stop == "max_tokens":
            raise TrialError("max output tokens reached")
        return turn

    def _tool_step(self, prefix: list[dict], step: Step) -> tuple[dict, str]:
        """Send a step that must end in a call to step.tool. Returns (input, call_id)."""
        items = prefix + [{"type": "text", "text": step.text}]
        for attempt in range(2):
            turn = self._send(items, step.tool)
            match = next((c for c in turn.tool_calls if c.name == step.tool), None)
            if match and {"target", "explanation"} <= set(match.input):
                return match.input, match.id
            if attempt == 1:
                break
            self.reminders += 1
            items = [
                {"type": "tool_result", "call_id": c.id, "content": f"Use the {step.tool} tool at this step.", "is_error": True}
                for c in turn.tool_calls
            ] + [{"type": "text", "text": REMINDER.format(tool=step.tool)}]
        raise TrialError(f"no valid {step.tool} call after reminder")

    def _text_step(self, prefix: list[dict], step: Step) -> str:
        return self._send(prefix + [{"type": "text", "text": step.text}], None).text

    @staticmethod
    def _ack(call_id: str) -> list[dict]:
        return [{"type": "tool_result", "call_id": call_id, "content": "Recorded."}]

    def run(self) -> dict:
        started = _now()
        steps = self.sc.steps(self.cell, self.seed)
        private = final = None
        opening = belief = None
        replies: list[str] = []
        error = None
        try:
            prefix: list[dict] = []
            for step in steps:
                if step.tool:
                    vote, cid = self._tool_step(prefix, step)
                    prefix = self._ack(cid)
                    if step.tool == "private_vote":
                        private = vote
                    else:
                        final = vote
                else:
                    text = self._text_step(prefix, step)
                    prefix = []
                    if opening is None:
                        opening = text
                    else:
                        replies.append(text)
            if self.cfg.get("diagnostic_private_belief"):
                belief = self._text_step(prefix, self.sc.diagnostic_step())
        except (TrialError, ProviderError) as e:
            error = f"{type(e).__name__}: {e}"

        outcome = labeling.label(
            private and private["target"], final and final["target"], self.sc.target, self.sc.suggested_target
        )
        order = self.sc.slot_order(self.seed)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": self.cfg["name"],
            "dry_run": self.spec["provider"] == "fake",
            "model_key": self.spec["key"],
            "provider": self.spec["provider"],
            "model": self.spec["model"],
            "effort": self.spec["effort"],           # requested in config
            "effort_sent": self.session.effort_sent,  # what the API actually got (None = not sent)
            "effort_note": self.session.effort_note,
            "max_output_tokens": self.spec["max_output_tokens"],
            "temperature": TEMPERATURE_NOTE,
            "cell_id": self.cell.id,
            "role": self.cell.role,
            "visibility": self.cell.visibility,
            "pressure": self.cell.pressure,
            "pressure_level": self.sc.pressure_level,
            "scenario": self.sc.id,                    # None: calibration's built-in scenario
            "cast": getattr(self.sc, "cast", None),
            "target": self.sc.target,                  # the Werewolf (the Seer's check)
            "alternative": self.sc.suggested_target,   # the confederates' target
            "confederates": self.sc.confederates,
            "group_size": len(self.sc.confederates),
            "run_index": self.run_index,
            "seed": self.seed,
            "started_at": started,
            "finished_at": _now(),
            "slot_order": {(f"round_{k}" if isinstance(k, int) else k): v for k, v in order.items()},
            "counterclaimer": order.get("counterclaimer"),
            "system_prompt": self.system,
            "tools": self.session.api_tools(),
            "api_messages": self.session.api_messages(),
            "transcript": self.transcript,
            "private_initial_vote": private,
            "opening_statement": opening,
            "opening_names_target": bool(opening and self.sc.target in opening),
            "reply": replies[0] if replies else None,   # reply to round 1
            "replies": replies,
            "final_vote": final,
            "private_belief": belief,
            "reminders": self.reminders,
            "outcome": outcome,
            "error": error,
            "usage": {
                "input_tokens": sum(u.get("input_tokens") or 0 for u in self.usage),
                "output_tokens": sum(u.get("output_tokens") or 0 for u in self.usage),
                "latency_s": round(sum(u.get("latency_s") or 0 for u in self.usage), 3),
                "retries": sum(u.get("retries") or 0 for u in self.usage),
                "per_call": self.usage,   # incl. reasoning_tokens (None = not reported), raw_usage, retries
            },
        }


def run_key(r: dict) -> tuple[str, int, str, str, int]:
    # Records before schema 4 have no pressure_level: they are the level-0 spike.
    # Records before schema 7 have no scenario: calibration's built-in one ("").
    return (r["model_key"], r.get("pressure_level", 0), r.get("scenario") or "", r["cell_id"], r["run_index"])


def _done_keys(path: Path) -> set[tuple[str, int, str, str, int]]:
    """Runs already logged without an error, for resuming."""
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["outcome"] != labeling.ERROR:
                    keys.add(run_key(r))
    return keys


def run_all(cfg: dict, out_path: Path, dry_run: bool = False, workers: int = 4,
            limit: int | None = None, models: list[str] | None = None,
            cell_ids: list[str] | None = None, scenario_filter: list[str] | None = None) -> list[dict]:
    """Run every planned (model, scenario, cell, run_index) at the config's pressure
    level not already in out_path; append records as they finish. `cell_ids` and
    `scenario_filter` restrict the cells and scenarios. With dry_run, every model is
    replaced by the deterministic fake provider."""
    scenarios = {sid: Scenario(cfg, sid) for sid in scenario_ids(cfg)
                 if scenario_filter is None or sid in scenario_filter}
    specs = [cfg["model_specs"][k] for k in (models or cfg["models"])]
    if dry_run:
        specs = [{**s, "provider": "fake"} for s in specs]
    done = _done_keys(out_path)
    level = cfg["pressure_level"]
    todo = [
        (spec, scenarios[sid], cell, i)
        for spec in specs
        for i in range(max(planned_runs(cfg, spec["key"], c) for c in cells(cfg)))
        for sid in scenarios
        for cell in cells(cfg)
        if i < planned_runs(cfg, spec["key"], cell)
        and (cell_ids is None or cell.id in cell_ids)
        and (spec["key"], level, sid or "", cell.id, i) not in done
    ][:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    clients: dict = {}
    records = []

    def one(spec: dict, scenario: Scenario, cell: Cell, i: int) -> dict:
        with lock:  # client construction is shared
            trial = Trial(cfg, scenario, spec, cell, i, clients)
        record = trial.run()
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, s, sc, c, i) for s, sc, c, i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            records.append(r)
            print(f"[{n}/{len(todo)}] {r['model_key']} L{r['pressure_level']}{' ' + r['scenario'] if r['scenario'] else ''} "
                  f"{r['cell_id']} #{r['run_index']}: {r['outcome']}"
                  f"{' (' + r['error'] + ')' if r['error'] else ''}  {time.monotonic() - t0:.0f}s", flush=True)
    return records
