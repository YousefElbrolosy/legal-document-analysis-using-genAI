"""LLM client(s) for the MS3 multi-agent system.

Two providers are wired up:

  * `get_client()`        — OpenRouter / OpenAI-compatible client. Preserved
                            for fallback / future swap, but NOT currently
                            used by `chat()`.
  * `get_ollama_client()` — Ollama Cloud client (the official `ollama`
                            Python SDK pointed at https://ollama.com with a
                            Bearer token). This is what `chat()` uses today.

Ollama's OpenAI-compat semantics differ in two ways that the rest of the
codebase has to handle:
  - `response.message.tool_calls[i].function.arguments` is already a Python
    dict (no `json.loads` step needed).
  - Tool result messages are `{"role": "tool", "name": ..., "content": ...}`
    — no `tool_call_id` field. (See `ms3/agents/base.py`.)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI
from ollama import Client as OllamaClient

from ms3.config import (
    MAX_TOKENS,
    MODEL,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
    TOP_P,
)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """OpenRouter / OpenAI-compatible client. Kept for fallback — chat()
    does not use this today."""
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


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    """Ollama Cloud client, authenticated with a Bearer token from .env."""
    if not OLLAMA_API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Add it to ms3/.env "
            "(see ms3/.env.example)."
        )
    return OllamaClient(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    """Single non-streaming chat call against Ollama Cloud.

    Returns the response's `.message` object, which exposes:
      - `.content`   (str)
      - `.tool_calls` (list | None) — each item has `.function.name` and
        `.function.arguments` (already a dict).
    """
    options: dict[str, Any] = {
        "temperature": TEMPERATURE if temperature is None else temperature,
        "top_p": TOP_P,
        "num_predict": MAX_TOKENS if max_tokens is None else max_tokens,
    }
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        kwargs["tools"] = tools
    # Ollama's analogue of OpenAI's response_format={"type":"json_object"}.
    if response_format is not None and response_format.get("type") == "json_object":
        kwargs["format"] = "json"

    response = get_ollama_client().chat(**kwargs)
    return response.message
