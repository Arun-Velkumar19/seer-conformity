"""Deterministic fake model for --dry-run and tests. No SDK, no keys, no network.

Responses depend only on what the session has been shown (system prompt and
user turns), so the same run always produces the same record. The mix of
behaviours exercises every branch of the pipeline:

- private vote: the target, except ~1 in 10 sessions vote the suggested target
  (-> excluded_initially_incorrect)
- final vote: cycles held / caved / other by a hash of the transcript
- ~1 in 15 sessions skip the first tool call (-> one vote reminder)
"""

from __future__ import annotations

import hashlib
import re

from . import ToolCall, Turn

TARGET, SUGGESTED, OTHER = "Morgan", "Drew", "Riley"


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


class FakeSession:
    def __init__(self, spec: dict, system: str, tools: list[dict]):
        self.spec = spec
        self.system = system
        self.tools = tools
        self.history: list[dict] = []
        self.effort_sent: str | None = spec["effort"]
        self.effort_note: str | None = None
        self.calls = 0
        self.skipped = False

    def _seen(self) -> str:
        return self.spec.get("salt", "") + self.system + "".join(
            i.get("text", "") for m in self.history if m["role"] == "user" for i in m["content"]
        )

    def send(self, items: list[dict], tool_mode: str) -> Turn:
        self.history.append({"role": "user", "content": items})
        last = next((i["text"] for i in reversed(items) if i["type"] == "text"), "")
        seen = self._seen()
        h = _h(seen)
        self.calls += 1
        usage = {"input_tokens": len(seen.split()), "output_tokens": 0}

        if tool_mode == "none":
            role_claim = "I'm the Seer: I checked Morgan and Morgan is the Werewolf." if "You are the **Seer**" in self.system \
                else "Avery's final message named Morgan, so I plan to vote Morgan."
            turn = Turn(text=f"[fake] {role_claim}", tool_calls=[], stop="end")
        else:
            tool = "private_vote" if "private_vote" in last else "vote"
            if not self.skipped and tool == "private_vote" and _h("skip" + seen) % 15 == 0:
                self.skipped = True
                turn = Turn(text="[fake] Let me think about that.", tool_calls=[], stop="end")
            else:
                if tool == "private_vote":
                    target = SUGGESTED if h % 10 == 0 else TARGET
                else:
                    target = (TARGET, TARGET, SUGGESTED, OTHER)[h % 4]
                call = ToolCall(f"fake_{self.calls}", tool, {"target": target, "explanation": "[fake] deterministic"})
                turn = Turn(text="", tool_calls=[call], stop="tool")

        turn.usage = {**usage, "output_tokens": len(re.findall(r"\w+", turn.text)) + 10 * len(turn.tool_calls),
                      "latency_s": 0.0, "retries": 0, "request_id": None, "raw_usage": None}
        self.history.append({
            "role": "assistant",
            "content": ([{"type": "text", "text": turn.text}] if turn.text else [])
            + [{"type": "tool_call", "id": c.id, "name": c.name, "input": c.input} for c in turn.tool_calls],
        })
        return turn

    def api_messages(self) -> list[dict]:
        return self.history

    def api_tools(self) -> list[dict]:
        return self.tools
