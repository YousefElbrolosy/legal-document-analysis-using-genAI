"""OpenRouter-backed Qwen2.5-72B-Instruct client.

OpenRouter exposes an OpenAI-compatible API, so the OpenAI Python SDK works
unchanged with a `base_url` override. Tool-calling uses the standard
`tools=[...]` / `tool_choice="auto"` format.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from ms3.config import (
    MAX_TOKENS,
    MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
    TOP_P,
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy ms3/.env.example to ms3/.env "
            "and fill in your key."
        )
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/contractnli-ms3",
            "X-Title": "ContractNLI MS3 multi-agent",
        },
    )


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    """Single completion call. Returns the raw OpenAI ChatCompletion message."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format is not None:
        kwargs["response_format"] = response_format
    completion = get_client().chat.completions.create(**kwargs)
    return completion.choices[0].message
