"""Shared tool-loop helper: drives an LLM round of tool-calling.

Both Analyzer and Conversation agents use the same loop:
  1. Send messages + tools.
  2. If the LLM returns tool_calls, dispatch them via TOOL_FUNCTIONS,
     append the tool results to messages, and call the LLM again.
  3. Stop when the LLM returns a plain assistant message (no tool_calls)
     or after `max_iters` rounds.
"""
from __future__ import annotations

import json
from typing import Any

from ms3.agents.tools import TOOL_FUNCTIONS
from ms3.models.llm import chat


def run_tool_loop(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    max_iters: int = 6,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Returns (final_assistant_text, full_messages_including_tool_calls)."""
    for _ in range(max_iters):
        msg = chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", messages
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result: Any = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:6000],
                }
            )
    messages.append(
        {"role": "assistant", "content": "[max tool iterations reached]"}
    )
    return "[max tool iterations reached]", messages
