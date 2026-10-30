"""Provider-neutral chat sessions.

The runner talks to every model through the same small interface:

    session = open_session(spec, system, tools, clients)
    turn = session.send(items, tool_mode)      # tool_mode: "auto" or "none"

`items` are what the subject is shown in one user turn:
    {"type": "text", "text": ...}
    {"type": "tool_result", "call_id": ..., "content": ..., "is_error": bool}

Each session keeps its provider-native history (sent back unchanged, so
reasoning/thinking items stay valid) and exposes it via api_messages() for
logging. SDKs are imported lazily, so the fake provider needs neither SDK nor
keys. Each SDK reads its own API key from the environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class ProviderError(Exception):
    """An API failure or refusal that ends the trial."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop: str                      # "end", "tool", "refusal", "max_tokens", or the raw reason
    usage: dict = field(default_factory=dict)   # input_tokens, output_tokens, plus provider extras


def call_raw(create, **kwargs):
    """Call an SDK `with_raw_response.create` and return (parsed response, call log).
    The log has the SDK's own retry count (it retries rate limits and server errors
    silently), the request id, and wall-clock latency including those retries."""
    t0 = time.monotonic()
    raw = create(**kwargs)
    latency = time.monotonic() - t0
    return raw.parse(), {
        "latency_s": round(latency, 3),
        "retries": getattr(raw, "retries_taken", None),
        "request_id": getattr(raw, "request_id", None),
    }


def dump_usage(u) -> dict | None:
    """The usage object exactly as the API returned it, for the run log."""
    if u is None:
        return None
    return u.model_dump(mode="json") if hasattr(u, "model_dump") else dict(vars(u))


def open_session(spec: dict, system: str, tools: list[dict], clients: dict):
    provider = spec["provider"]
    if provider == "fake":
        from .fake import FakeSession
        return FakeSession(spec, system, tools)
    if provider == "anthropic":
        from .anthropic_provider import AnthropicSession, make_client
    elif provider == "openai":
        from .openai_provider import OpenAISession, make_client
    else:
        raise ValueError(f"unknown provider {provider!r}")
    if provider not in clients:
        clients[provider] = make_client()
    session_cls = AnthropicSession if provider == "anthropic" else OpenAISession
    return session_cls(clients[provider], spec, system, tools)
