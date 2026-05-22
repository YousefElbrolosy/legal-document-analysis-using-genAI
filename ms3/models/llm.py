"""LLM client(s) for the MS3 multi-agent system.

Four providers are wired up, selected by `MS3_PROVIDER`:

  * `openai`     — OpenAI (api.openai.com). Default.
  * `nvidia`     — NVIDIA Build / NIM (OpenAI-compatible, free tier).
  * `openrouter` — OpenRouter (OpenAI-compatible).
  * `ollama`     — Ollama Cloud (uses the official `ollama` Python SDK).

`chat()` returns an object that exposes `.content` (str) and `.tool_calls`
(list | None). The shape of each tool_call differs slightly between the
OpenAI SDK and the Ollama SDK; `ms3/agents/base.py` handles both via
`ms3.config.PROVIDER`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from ms3.config import (
    MAX_TOKENS,
    MODEL,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PROVIDER,
    TEMPERATURE,
    TOP_P,
)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to ms3/.env "
            "(see ms3/.env.example)."
        )
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


@lru_cache(maxsize=1)
def get_nvidia_client() -> OpenAI:
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Copy ms3/.env.example to ms3/.env "
            "and fill in your key from https://build.nvidia.com."
        )
    return OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
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
def get_ollama_client():
    # Lazy import — only required when MS3_PROVIDER=ollama, so users on
    # OpenAI / NVIDIA / OpenRouter don't need the `ollama` package installed.
    from ollama import Client as OllamaClient

    if not OLLAMA_API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Add it to ms3/.env "
            "(see ms3/.env.example)."
        )
    return OllamaClient(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )


def _chat_openai(
    client: OpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        kwargs["tools"] = tools
    if response_format is not None:
        kwargs["response_format"] = response_format
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message


def _chat_ollama(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> Any:
    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": TOP_P,
        "num_predict": max_tokens,
    }
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        kwargs["tools"] = tools
    if response_format is not None and response_format.get("type") == "json_object":
        kwargs["format"] = "json"
    response = get_ollama_client().chat(**kwargs)
    return response.message


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    """Single non-streaming chat call against the configured provider."""
    temp = TEMPERATURE if temperature is None else temperature
    tokens = MAX_TOKENS if max_tokens is None else max_tokens

    if PROVIDER == "openai":
        return _chat_openai(get_openai_client(), messages, tools, temp, tokens, response_format)
    if PROVIDER == "nvidia":
        return _chat_openai(get_nvidia_client(), messages, tools, temp, tokens, response_format)
    if PROVIDER == "openrouter":
        return _chat_openai(get_openrouter_client(), messages, tools, temp, tokens, response_format)
    if PROVIDER == "ollama":
        return _chat_ollama(messages, tools, temp, tokens, response_format)
    raise RuntimeError(
        f"Unknown MS3_PROVIDER={PROVIDER!r}. Expected one of: openai, nvidia, openrouter, ollama."
    )
