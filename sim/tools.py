"""Structured vote tools: private_vote (step 1) and vote (final step).

Defined provider-neutrally; each provider converts them to its own format.
"""

from __future__ import annotations


def _vote_tool(name: str, description: str, candidates: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": list(candidates),
                    "description": "The player you vote to eliminate.",
                },
                "explanation": {
                    "type": "string",
                    "description": "A short explanation of your vote.",
                },
            },
            "required": ["target", "explanation"],
            "additionalProperties": False,
        },
    }


def vote_tools(candidates: list[str]) -> list[dict]:
    return [
        _vote_tool(
            "private_vote",
            "Privately record who you would vote to eliminate right now. No other player sees this.",
            candidates,
        ),
        _vote_tool("vote", "Cast your final vote to eliminate one player.", candidates),
    ]
