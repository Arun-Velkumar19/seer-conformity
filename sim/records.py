"""Read run records back from JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from . import labeling
from .runner import REMINDER, run_key

_REMINDER_START = REMINDER.split("{")[0]


def load_records(path: str | Path) -> list[dict]:
    """One record per (model, cell, run_index): the latest non-error attempt if
    any, else the latest error. Errored attempts are retried by resuming, so a
    file can hold several attempts for the same run."""
    best: dict[tuple, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prev = best.get(run_key(r))
            if prev is None or r["outcome"] != labeling.ERROR or prev["outcome"] == labeling.ERROR:
                best[run_key(r)] = r
    return sorted(best.values(), key=run_key)


def user_texts(record: dict) -> list[str]:
    """Text the subject was shown, in order, excluding vote reminders."""
    return [
        i["text"]
        for m in record["transcript"] if m["role"] == "user"
        for i in m["items"] if i["type"] == "text" and not i["text"].startswith(_REMINDER_START)
    ]
