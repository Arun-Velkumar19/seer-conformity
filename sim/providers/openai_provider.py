"""OpenAI Responses API session.

Effort goes in reasoning.effort. Nothing is stored server-side (store=False);
reasoning items come back encrypted and are sent back unchanged with the rest
of the output, so the conversation is append-only. Temperature is not sent.

If the model spec sets omit_effort_if_rejected and the API rejects the
reasoning settings, the request is retried once without them, and the session
keeps sending none; effort_sent becomes None and effort_note keeps the API's
message for the run log. Only an error that names the reasoning parameter
itself counts as a rejection (a mention inside a docs URL does not).

A usage-policy flag ("Invalid prompt: ... flagged ...") is retried as the same
request, effort unchanged, up to the client's retry limit; if it keeps
failing, the trial ends with an error so a resume runs it again. A policy flag
never drops effort.
"""

from __future__ import annotations

import json
import re
import time

import openai

from . import ProviderError, ToolCall, Turn, call_raw, dump_usage


def make_client():
    # Reads OPENAI_API_KEY from the environment.
    return openai.OpenAI(max_retries=5)


_URL = re.compile(r"https?://\S+")
# Parameter names as an error message states them, e.g. "'reasoning.effort' is not supported".
_REASONING_PARAM_TEXT = re.compile(r"""['"`]?\b(reasoning(\.\w+)*|reasoning_effort|include)\b['"`]?\s+(is|are)\s+not\s+supported"""
                                   r"""|unsupported\s+(parameter|value)\s*:\s*['"`]?(reasoning|include)""", re.I)


def _error_fields(e) -> tuple[str, str, str]:
    body = getattr(e, "body", None)
    body = body.get("error", body) if isinstance(body, dict) else {}
    param = (getattr(e, "param", None) or body.get("param") or "").lower()
    code = (getattr(e, "code", None) or body.get("code") or "").lower()
    msg = getattr(e, "message", None) or str(e)
    return param, code, msg


def _is_policy_flag(e) -> bool:
    """True if a 400 says the prompt was flagged under the usage policy."""
    _, code, msg = _error_fields(e)
    text = _URL.sub("", msg).lower()
    return code == "invalid_prompt" or "usage policy" in text or "flagged as potentially violating" in text


def _rejects_reasoning(e) -> bool:
    """True if a 400 is about the reasoning settings themselves (reasoning.effort
    or the encrypted-reasoning include), judged from the error's param or code
    field or explicit parameter text; never from a URL in the message."""
    if _is_policy_flag(e):
        return False
    param, code, msg = _error_fields(e)
    if param.startswith("reasoning") or param == "include":
        return True
    if "reasoning" in code or "effort" in code:
        return True
    return bool(_REASONING_PARAM_TEXT.search(_URL.sub("", msg)))


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
        self.sleep = time.sleep       # between policy-flag retries; tests replace it

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
        return call_raw(self.client.responses.with_raw_response.create, **kwargs)

    def _create_with_retries(self, tool_mode: str):
        limit = getattr(self.client, "max_retries", 5)
        policy_retries, effort_retry = 0, False
        while True:
            try:
                resp, call = self._create(tool_mode)
                break
            except openai.BadRequestError as e:
                if _is_policy_flag(e) and policy_retries < limit:
                    policy_retries += 1            # same request again, effort unchanged
                    self.sleep(min(2 ** policy_retries, 30))
                    continue
                if not (self.spec.get("omit_effort_if_rejected") and self.effort_sent is not None
                        and not effort_retry and _rejects_reasoning(e)):
                    raise
                self.effort_sent = None
                self.effort_note = f"effort {self.spec['effort']!r} rejected by API, sent none: {e.message}"[:500]
                effort_retry = True
        if effort_retry:
            call["effort_retry"] = True        # an attempt was refused for its reasoning settings
        if policy_retries:
            call["policy_retries"] = policy_retries   # attempts refused as usage-policy flags
        return resp, call

    def send(self, items: list[dict], tool_mode: str) -> Turn:
        user_text = []
        for it in items:
            if it["type"] == "tool_result":
                self.items.append({"type": "function_call_output", "call_id": it["call_id"], "output": it["content"]})
            else:
                user_text.append({"type": "input_text", "text": it["text"]})
        if user_text:
            self.items.append({"role": "user", "content": user_text})
        t0 = time.monotonic()
        try:
            resp, call = self._create_with_retries(tool_mode)
        except openai.APIError as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e
        self.items.extend(o.model_dump(mode="json", exclude_none=True) for o in resp.output)

        u = resp.usage
        details = getattr(u, "output_tokens_details", None)
        usage = {
            # None, not 0, when the API omits a field: "missing" must not read as "zero"
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "status": resp.status,
            **call,
            "latency_s": round(time.monotonic() - t0, 3),   # includes a refused-effort attempt, if any
            "raw_usage": dump_usage(u),
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
