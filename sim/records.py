"""Read run records back from JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from . import labeling


def load_records(path: str | Path) -> list[dict]:
    """One record per (cell_id, run_index): the latest non-error attempt if any,
    else the latest error. Errored attempts are retried by resuming, so a file
    can hold several attempts for the same run."""
    best: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["cell_id"], r["run_index"])
            prev = best.get(key)
            if prev is None or r["outcome"] != labeling.ERROR or prev["outcome"] == labeling.ERROR:
                best[key] = r
    return sorted(best.values(), key=lambda r: (r["cell_id"], r["run_index"]))


def user_texts(record: dict, skip_prefix: str = "Please record your vote") -> list[str]:
    """Text the subject was shown, in order, excluding vote reminders."""
    out = []
    for m in record["messages"]:
        if m["role"] != "user" or isinstance(m["content"], str):
            continue
        for b in m["content"]:
            if b.get("type") == "text" and not b["text"].startswith(skip_prefix):
                out.append(b["text"])
    return out
