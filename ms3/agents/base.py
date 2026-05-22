"""Shared tool-loop helper for the multi-agent system.

Supports two response/message dialects, switched on `ms3.config.PROVIDER`:

  * OpenAI-compatible (`nvidia`, `openrouter`)
      - `tool_calls[i].function.arguments` is a JSON string,
      - echoed assistant turns include `id` + `type="function"` per call,
      - tool result messages use `tool_call_id`.

  * Ollama Cloud (`ollama`)
      - `tool_calls[i].function.arguments` is already a Python dict,
      - echoed assistant turns omit `id`/`type`,
      - tool result messages use `name` (no `tool_call_id`).

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
from ms3.config import PROVIDER
from ms3.models.llm import chat

_OPENAI_COMPAT = {"openai", "nvidia", "openrouter"}


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


def _arguments_to_str(arguments: Any) -> str:
    """OpenAI-compat echo: arguments must be a JSON string."""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(_arguments_to_dict(arguments), ensure_ascii=False)


def run_tool_loop(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    max_iters: int = 6,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Returns (final_assistant_text, full_messages_including_tool_calls)."""
    openai_compat = PROVIDER in _OPENAI_COMPAT

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

        # Build the echoed assistant turn in the provider's native shape.
        echoed_calls: list[dict[str, Any]] = []
        # (call_id, name, parsed_args) for dispatch + tool-result formatting.
        dispatch: list[tuple[str | None, str, dict[str, Any]]] = []
        for i, tc in enumerate(tool_calls):
            name = tc.function.name
            args_parsed = _arguments_to_dict(tc.function.arguments)
            if openai_compat:
                # OpenAI requires id; synthesise one if the provider omits it.
                call_id = getattr(tc, "id", None) or f"call_{i}"
                echoed_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _arguments_to_str(tc.function.arguments),
                        },
                    }
                )
                dispatch.append((call_id, name, args_parsed))
            else:
                echoed_calls.append(
                    {"function": {"name": name, "arguments": args_parsed}}
                )
                dispatch.append((None, name, args_parsed))

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": echoed_calls,
            }
        )

        for call_id, name, args in dispatch:
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result: Any = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            payload = json.dumps(result, ensure_ascii=False, default=str)[:6000]
            if openai_compat:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": payload,
                    }
                )
            else:
                messages.append(
                    {"role": "tool", "name": name, "content": payload}
                )

    messages.append(
        {"role": "assistant", "content": "[max tool iterations reached]"}
    )
    return "[max tool iterations reached]", messages
