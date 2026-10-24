"""Run trials against the Claude API and write one JSONL record per run.

Per run (SPEC v2, Sequence per run):
  1. private_vote tool call
  2. opening statement            (visibility announced in this message)
  3. pressure round 1 + reply
  4. pressure round 2 + final vote tool call
  (5. private belief, diagnostic arm only)

The conversation is append-only: each response's content goes back unchanged,
so thinking blocks stay valid. Forced tool_choice is not available on current
models, so vote steps use tool_choice "auto" + strict schemas + the prompt's
instruction; if no vote arrives, one fixed reminder is sent and counted.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from . import labeling
from .config import Cell, cells, model_id
from .scenario import Scenario, Step
from .tools import vote_tools

SCHEMA_VERSION = 1
TEMPERATURE_NOTE = "api_default (sampling params not accepted on API version 2026-10-21)"
REMINDER = "Please record your vote now using the {tool} tool."


class TrialError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(content) -> list | str:
    """JSON-safe copy of message content (SDK blocks or plain dicts)."""
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        out.append(block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else block)
    return out


def _text(content) -> str:
    return "\n".join(b.text for b in content if b.type == "text").strip()


class Trial:
    def __init__(self, client, cfg: dict, scenario: Scenario, cell: Cell, run_index: int):
        self.client = client
        self.cfg = cfg
        self.sc = scenario
        self.cell = cell
        self.run_index = run_index
        self.seed = cfg["seeds"][run_index]
        self.model = model_id(cfg)
        self.system = scenario.system_prompt(cell)
        self.tools = vote_tools(scenario.candidates)
        self.history: list[dict] = []
        self.usage: list[dict] = []
        self.reminders = 0

    def _call(self, user_content: list[dict], tool: str | None):
        self.history.append({"role": "user", "content": user_content})
        m = self.cfg["model"]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=m["max_tokens"],
            system=self.system,
            messages=self.history,
            tools=self.tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True} if tool else {"type": "none"},
            output_config={"effort": m["effort"]},
        )
        self.history.append({"role": "assistant", "content": resp.content})
        u = resp.usage
        self.usage.append(
            {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                "stop_reason": resp.stop_reason,
            }
        )
        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise TrialError(f"refusal (category={category})")
        if resp.stop_reason == "max_tokens":
            raise TrialError("max_tokens reached")
        return resp

    def _tool_step(self, prefix: list[dict], step: Step) -> tuple[dict, str]:
        """Send a step that must end in a call to step.tool. Returns (input, tool_use_id)."""
        content = prefix + [{"type": "text", "text": step.text}]
        for attempt in range(2):
            resp = self._call(content, step.tool)
            uses = [b for b in resp.content if b.type == "tool_use"]
            match = next((b for b in uses if b.name == step.tool), None)
            if match:
                return dict(match.input), match.id
            if attempt == 1:
                break
            self.reminders += 1
            content = [
                {"type": "tool_result", "tool_use_id": b.id, "content": f"Use the {step.tool} tool at this step.", "is_error": True}
                for b in uses
            ] + [{"type": "text", "text": REMINDER.format(tool=step.tool)}]
        raise TrialError(f"no {step.tool} call after reminder")

    def _text_step(self, prefix: list[dict], step: Step) -> str:
        resp = self._call(prefix + [{"type": "text", "text": step.text}], None)
        return _text(resp.content)

    @staticmethod
    def _ack(tool_use_id: str) -> list[dict]:
        return [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "Recorded."}]

    def run(self) -> dict:
        started = _now()
        steps = self.sc.steps(self.cell, self.seed)
        private = final = None
        opening = reply = belief = None
        error = None
        try:
            private, tid = self._tool_step([], steps[0])
            opening = self._text_step(self._ack(tid), steps[1])
            reply = self._text_step([], steps[2])
            final, tid = self._tool_step([], steps[3])
            if self.cfg.get("diagnostic_private_belief"):
                belief = self._text_step(self._ack(tid), self.sc.diagnostic_step())
        except (TrialError, anthropic.APIError) as e:
            error = f"{type(e).__name__}: {e}"

        outcome = labeling.label(
            private and private["target"], final and final["target"], self.sc.target, self.sc.suggested_target
        )
        order = self.sc.slot_order(self.seed)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": self.cfg["name"],
            "cell_id": self.cell.id,
            "role": self.cell.role,
            "visibility": self.cell.visibility,
            "pressure": self.cell.pressure,
            "group_size": len(self.sc.confederates),
            "run_index": self.run_index,
            "seed": self.seed,
            "model": self.model,
            "effort": self.cfg["model"]["effort"],
            "max_tokens": self.cfg["model"]["max_tokens"],
            "temperature": TEMPERATURE_NOTE,
            "sdk_version": anthropic.__version__,
            "started_at": started,
            "finished_at": _now(),
            "slot_order": {"round_1": order[1], "round_2": order[2]},
            "system_prompt": self.system,
            "tools": self.tools,
            "messages": [{"role": m["role"], "content": _dump(m["content"])} for m in self.history],
            "private_initial_vote": private,
            "opening_statement": opening,
            "opening_names_target": bool(opening and self.sc.target in opening),
            "reply": reply,
            "final_vote": final,
            "private_belief": belief,
            "reminders": self.reminders,
            "outcome": outcome,
            "error": error,
            "usage": {
                "input_tokens": sum(u["input_tokens"] for u in self.usage),
                "output_tokens": sum(u["output_tokens"] for u in self.usage),
                "per_call": self.usage,
            },
        }


def _done_keys(path: Path) -> set[tuple[str, int]]:
    """(cell_id, run_index) pairs already logged without an error, for resuming."""
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["outcome"] != labeling.ERROR:
                    keys.add((r["cell_id"], r["run_index"]))
    return keys


def run_all(cfg: dict, out_path: Path, client=None, workers: int = 4, limit: int | None = None) -> list[dict]:
    """Run every (cell, run_index) not already in out_path; append records as they finish."""
    client = client or anthropic.Anthropic(max_retries=5)
    scenario = Scenario(cfg)
    done = _done_keys(out_path)
    todo = [
        (cell, i)
        for i in range(cfg["runs_per_cell"])
        for cell in cells(cfg)
        if (cell.id, i) not in done
    ][:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    records = []

    def one(cell: Cell, i: int) -> dict:
        record = Trial(client, cfg, scenario, cell, i).run()
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, c, i) for c, i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            records.append(r)
            print(f"[{n}/{len(todo)}] {r['cell_id']} #{r['run_index']}: {r['outcome']}"
                  f"{' (' + r['error'] + ')' if r['error'] else ''}  {time.monotonic() - t0:.0f}s", flush=True)
    return records
