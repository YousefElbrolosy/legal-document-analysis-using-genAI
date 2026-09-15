"""Shared tool-loop helper for the Ollama-Cloud-backed agents.

Ollama's response/message shape differs from OpenAI's:
  - `response.message.tool_calls[i].function.arguments` is already a Python
    dict (no JSON-decoding step needed),
  - assistant messages with tool calls are echoed back as
    `{"role": "assistant", "tool_calls": [{"function": {"name", "arguments"}}]}`
    (no `id`/`type` fields),
  - tool results are `{"role": "tool", "name": ..., "content": ...}` —
    there is no `tool_call_id` field.

The loop:
  1. Send messages + tools.
  2. If the response message has `tool_calls`, dispatch each via
     `TOOL_FUNCTIONS` and append the result as a tool message.
  3. Stop when the model returns a plain assistant message (no tool_calls)
     or after `max_iters` rounds.
"""
from __future__ import annotations

import json
from typing import Any

from ms3.agents.tools import TOOL_FUNCTIONS
from ms3.models.llm import chat


def _arguments_to_dict(arguments: Any) -> dict[str, Any]:
    """Normalise `function.arguments` across SDK versions (dict | str | mapping)."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return {}
    try:
        return dict(arguments)
    except Exception:
        return {}


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
        content = getattr(msg, "content", "") or ""
        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            return content, messages

        # Echo the assistant turn back in Ollama-native shape so the next
        # call can interpret the tool exchange correctly.
        echoed_calls: list[dict[str, Any]] = []
        dispatch: list[tuple[str, dict[str, Any]]] = []
        for tc in tool_calls:
            name = tc.function.name
            args = _arguments_to_dict(tc.function.arguments)
            echoed_calls.append({"function": {"name": name, "arguments": args}})
            dispatch.append((name, args))
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": echoed_calls,
            }
        )

        for name, args in dispatch:
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
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str)[:6000],
                }
            )

    messages.append(
        {"role": "assistant", "content": "[max tool iterations reached]"}
    )
    return "[max tool iterations reached]", messages
