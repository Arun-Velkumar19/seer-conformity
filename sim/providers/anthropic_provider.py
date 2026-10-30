"""Anthropic Messages API session.

Effort goes in output_config.effort. Forced tool_choice is rejected by current
models, so vote steps use "auto" with strict tool schemas; text steps use
"none". Temperature is not sent (rejected on API version 2026-10-21).
"""

from __future__ import annotations

import anthropic

from . import ProviderError, ToolCall, Turn, call_raw, dump_usage


def make_client():
    # Reads ANTHROPIC_API_KEY from the environment.
    return anthropic.Anthropic(max_retries=5)


def _dump(content):
    if isinstance(content, str):
        return content
    return [b.model_dump(mode="json", exclude_none=True) if hasattr(b, "model_dump") else b for b in content]


class AnthropicSession:
    def __init__(self, client, spec: dict, system: str, tools: list[dict]):
        self.client = client
        self.spec = spec
        self.system = system
        self.tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["schema"], "strict": True}
            for t in tools
        ]
        self.history: list[dict] = []
        self.effort_sent: str | None = spec["effort"]
        self.effort_note: str | None = None

    def send(self, items: list[dict], tool_mode: str) -> Turn:
        content = []
        for it in items:
            if it["type"] == "text":
                content.append({"type": "text", "text": it["text"]})
            else:
                block = {"type": "tool_result", "tool_use_id": it["call_id"], "content": it["content"]}
                if it.get("is_error"):
                    block["is_error"] = True
                content.append(block)
        self.history.append({"role": "user", "content": content})
        try:
            resp, call = call_raw(
                self.client.messages.with_raw_response.create,
                model=self.spec["model"],
                max_tokens=self.spec["max_output_tokens"],
                system=self.system,
                messages=self.history,
                tools=self.tools,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True} if tool_mode == "auto" else {"type": "none"},
                output_config={"effort": self.spec["effort"]},
            )
        except anthropic.APIError as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e
        self.history.append({"role": "assistant", "content": resp.content})

        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,   # includes any thinking tokens
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
            "stop_reason": resp.stop_reason,
            **call,
            "raw_usage": dump_usage(u),
        }
        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise ProviderError(f"refusal (category={category})")
        stop = {"end_turn": "end", "tool_use": "tool", "max_tokens": "max_tokens"}.get(resp.stop_reason, resp.stop_reason)
        return Turn(
            text="\n".join(b.text for b in resp.content if b.type == "text").strip(),
            tool_calls=[ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"],
            stop=stop,
            usage=usage,
        )

    def api_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": _dump(m["content"])} for m in self.history]

    def api_tools(self) -> list[dict]:
        return self.tools
