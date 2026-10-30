"""OpenAI Responses API session.

Effort goes in reasoning.effort. Nothing is stored server-side (store=False);
reasoning items come back encrypted and are sent back unchanged with the rest
of the output, so the conversation is append-only. Temperature is not sent.

If the model spec sets omit_effort_if_rejected and the API rejects the
reasoning settings, the request is retried once without them, and the session
keeps sending none; effort_sent becomes None and effort_note keeps the API's
message for the run log.
"""

from __future__ import annotations

import json

import openai

from . import ProviderError, ToolCall, Turn


def make_client():
    # Reads OPENAI_API_KEY from the environment.
    return openai.OpenAI(max_retries=5)


def _rejects_reasoning(e) -> bool:
    """True if a 400 is about the reasoning settings (reasoning.effort or the
    encrypted-reasoning include), not some other problem."""
    param = (getattr(e, "param", None) or "").lower()
    msg = (getattr(e, "message", None) or str(e)).lower()
    return param.startswith("reasoning") or param == "include" or "reasoning" in msg


class OpenAISession:
    def __init__(self, client, spec: dict, system: str, tools: list[dict]):
        self.client = client
        self.spec = spec
        self.system = system
        self.tools = [
            {"type": "function", "name": t["name"], "description": t["description"],
             "parameters": t["schema"], "strict": True}
            for t in tools
        ]
        self.items: list[dict] = []   # full input list, native format
        self.effort_sent: str | None = spec["effort"]
        self.effort_note: str | None = None

    def _create(self, tool_mode: str):
        kwargs = dict(
            model=self.spec["model"],
            instructions=self.system,
            input=self.items,
            tools=self.tools,
            tool_choice="auto" if tool_mode == "auto" else "none",
            parallel_tool_calls=False,
            max_output_tokens=self.spec["max_output_tokens"],
            store=False,
        )
        if self.effort_sent is not None:
            kwargs["reasoning"] = {"effort": self.effort_sent}
            kwargs["include"] = ["reasoning.encrypted_content"]
        return self.client.responses.create(**kwargs)

    def send(self, items: list[dict], tool_mode: str) -> Turn:
        user_text = []
        for it in items:
            if it["type"] == "tool_result":
                self.items.append({"type": "function_call_output", "call_id": it["call_id"], "output": it["content"]})
            else:
                user_text.append({"type": "input_text", "text": it["text"]})
        if user_text:
            self.items.append({"role": "user", "content": user_text})
        try:
            try:
                resp = self._create(tool_mode)
            except openai.BadRequestError as e:
                if not (self.spec.get("omit_effort_if_rejected") and self.effort_sent is not None
                        and _rejects_reasoning(e)):
                    raise
                self.effort_sent = None
                self.effort_note = f"effort {self.spec['effort']!r} rejected by API, sent none: {e.message}"[:500]
                resp = self._create(tool_mode)
        except openai.APIError as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e
        self.items.extend(o.model_dump(mode="json", exclude_none=True) for o in resp.output)

        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens if u else 0,
            "output_tokens": u.output_tokens if u else 0,
            "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) if u else 0,
            "status": resp.status,
        }

        texts, calls = [], []
        for o in resp.output:
            if o.type == "message":
                for c in o.content:
                    if c.type == "refusal":
                        raise ProviderError(f"refusal: {c.refusal}")
                    if c.type == "output_text":
                        texts.append(c.text)
            elif o.type == "function_call":
                try:
                    args = json.loads(o.arguments)
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(o.call_id, o.name, args))

        if resp.status == "incomplete":
            reason = getattr(resp.incomplete_details, "reason", None)
            stop = "max_tokens" if reason == "max_output_tokens" else f"incomplete:{reason}"
        else:
            stop = "tool" if calls else "end"
        return Turn(text="\n".join(texts).strip(), tool_calls=calls, stop=stop, usage=usage)

    def api_messages(self) -> list[dict]:
        # Native input items as sent: user/assistant messages carry a role;
        # reasoning, function_call and function_call_output items carry a type.
        # The system prompt is sent as `instructions` and logged separately.
        return list(self.items)

    def api_tools(self) -> list[dict]:
        return self.tools
