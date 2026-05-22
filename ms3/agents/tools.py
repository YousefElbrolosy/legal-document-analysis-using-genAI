"""Tool registry + `traced_tool` decorator.

Each tool here is exposed to the LLM via LangChain's `@tool`-style schema.
The `traced_tool` wrapper records `{tool_name, agent, args, output, latency_ms}`
on the active Runtrace held in a contextvar so any agent can call any tool
without threading the trace through every signature.
"""
from __future__ import annotations

import contextvars
import time
from typing import Any, Callable

from ms3.config import (
    DEFAULT_GRAPH_TOP_K_RULES,
    DEFAULT_VECTOR_TOP_K,
)
from ms3.playbook import apply as _apply
from ms3.playbook.loader import (
    DATASET_CHOICE_TO_LABEL,
    HID_TO_NDA,
    NDA_TO_HID,
    hypothesis_text,
)
from ms3.rag.graph_retriever import get_graph_retriever
from ms3.rag.vector_retriever import get_vector_retriever
from ms3.runtrace.builder import Runtrace, now

_current_runtrace: contextvars.ContextVar[Runtrace | None] = contextvars.ContextVar(
    "ms3_runtrace", default=None
)
_current_agent: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ms3_agent", default="orchestrator"
)
_current_hypothesis: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ms3_hypothesis", default=None
)


def set_runtrace(rt: Runtrace | None) -> contextvars.Token:
    return _current_runtrace.set(rt)


def set_agent(name: str) -> contextvars.Token:
    return _current_agent.set(name)


def set_hypothesis(hid: str | None) -> contextvars.Token:
    return _current_hypothesis.set(hid)


def traced_tool(name: str) -> Callable:
    """Decorator that records every invocation into the active runtrace."""

    def deco(fn: Callable) -> Callable:
        def wrapper(**kwargs):
            start = time.perf_counter()
            started_at = now()
            error: str | None = None
            output: Any = None
            try:
                output = fn(**kwargs)
                return output
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                ended_at = now()
                latency_ms = (time.perf_counter() - start) * 1000.0
                rt = _current_runtrace.get()
                if rt is not None:
                    rt.add_tool_call(
                        tool_name=name,
                        agent=_current_agent.get(),
                        args={k: _safe(v) for k, v in kwargs.items()},
                        output=_safe(output),
                        started_at=started_at,
                        ended_at=ended_at,
                        latency_ms=latency_ms,
                        error=error,
                        related_hypothesis_id=_current_hypothesis.get(),
                    )

        wrapper.__name__ = name
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


def _safe(v: Any) -> Any:
    """Clamp tool outputs to JSON-safe shapes; truncate huge strings."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v if len(v) <= 4000 else v[:4000] + " …[truncated]"
    if isinstance(v, list):
        return [_safe(x) for x in v[:20]]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in list(v.items())[:30]}
    return str(v)[:4000]


# --- contract registry (so tools can look up the active contract chunks) -----
_contract_registry: dict[str, dict[str, Any]] = {}


def register_contract(contract: dict[str, Any]) -> None:
    _contract_registry[contract["contract_id"]] = contract


def _get_chunks(contract_id: str) -> list[dict[str, Any]]:
    c = _contract_registry.get(contract_id)
    if c is None:
        raise KeyError(f"contract '{contract_id}' not registered")
    return c["chunks"]


# --- the tools ---------------------------------------------------------------

@traced_tool("vector_rag")
def vector_rag(query: str, top_k: int = DEFAULT_VECTOR_TOP_K) -> list[dict[str, Any]]:
    """Semantic-similarity retrieval of historical clause/hypothesis precedents."""
    return get_vector_retriever().retrieve(query, top_k=top_k)


@traced_tool("graph_rag")
def graph_rag(query: str, top_k_rules: int = DEFAULT_GRAPH_TOP_K_RULES) -> list[dict[str, Any]]:
    """Graph-traversal retrieval (rule node entry -> connected clauses)."""
    return get_graph_retriever().retrieve(query, top_k_rules=top_k_rules)


@traced_tool("playbook_lookup")
def playbook_lookup(hypothesis_id: str, label: str) -> dict[str, Any]:
    """Deterministically apply the MS1 playbook for (hypothesis_id, label).

    Returns severity, recommended_action, criticality, playbook_rule_ids.
    """
    return _apply.apply(hypothesis_id, label)


@traced_tool("validate_quote")
def validate_quote(contract_id: str, quote: str) -> dict[str, Any]:
    """Quote-integrity check: does `quote` appear (case-insensitive, whitespace
    tolerant) inside any chunk of `contract_id`? Returns chunk_id + span.
    """
    chunks = _get_chunks(contract_id)
    needle = " ".join(quote.lower().split())
    for ch in chunks:
        hay = " ".join(ch["text"].lower().split())
        if needle and needle in hay:
            return {
                "ok": True,
                "chunk_id": ch["chunk_id"],
                "span": ch["span"],
            }
    return {"ok": False, "reason": "quote not found in any contract chunk"}


@traced_tool("get_contract_chunk")
def get_contract_chunk(contract_id: str, chunk_id: str) -> dict[str, Any]:
    """Return one named chunk's text + span."""
    for ch in _get_chunks(contract_id):
        if ch["chunk_id"] == chunk_id:
            return {"chunk_id": chunk_id, "text": ch["text"], "span": ch["span"]}
    return {"error": f"chunk {chunk_id} not found"}


@traced_tool("web_search")
def web_search(query: str, k: int = 3) -> list[dict[str, Any]]:
    """DuckDuckGo web search via `ddgs` (no API key needed). Conv mode only."""
    try:
        from ddgs import DDGS
    except Exception:
        return [{"error": "ddgs library not installed"}]
    out: list[dict[str, Any]] = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=k):
            out.append(
                {
                    "title": r.get("title"),
                    "href": r.get("href") or r.get("url"),
                    "snippet": r.get("body") or r.get("snippet"),
                }
            )
    return out


# --- JSON-Schema descriptors for the LLM tool-calling interface --------------

TOOL_SCHEMAS_ANALYZER = [
    {
        "type": "function",
        "function": {
            "name": "vector_rag",
            "description": "Retrieve historical precedents (clause + hypothesis + outcome) from the ContractNLI training split by semantic similarity. Use this to recall how similar NDA clauses have been interpreted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": DEFAULT_VECTOR_TOP_K},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_rag",
            "description": "Graph-based retrieval: enter at the closest hypothesis/rule node and return its connected clauses with relations (Entailment/Contradiction).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k_rules": {"type": "integer", "default": DEFAULT_GRAPH_TOP_K_RULES},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contract_chunk",
            "description": "Fetch the full text + char-span of a contract chunk by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                },
                "required": ["contract_id", "chunk_id"],
            },
        },
    },
]

TOOL_SCHEMAS_VALIDATOR = [
    {
        "type": "function",
        "function": {
            "name": "validate_quote",
            "description": "Check whether a quote string appears verbatim (whitespace-insensitive) inside the contract's text. Returns ok + chunk_id if matched, else ok=false.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["contract_id", "quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "playbook_lookup",
            "description": "Look up the playbook's expected severity/action for a (hypothesis_id, label) pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["hypothesis_id", "label"],
            },
        },
    },
]

TOOL_SCHEMAS_CONVERSATION = [
    {
        "type": "function",
        "function": {
            "name": "vector_rag",
            "description": "Semantic search over the historical ContractNLI corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": DEFAULT_VECTOR_TOP_K},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_rag",
            "description": "Graph-based retrieval over the NetworkX rule/clause graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k_rules": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contract_chunk",
            "description": "Fetch a chunk of the active contract by chunk_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                },
                "required": ["contract_id", "chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web (DuckDuckGo). Use only when the user explicitly asks for outside information or current law.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]


TOOL_FUNCTIONS: dict[str, Callable] = {
    "vector_rag": vector_rag,
    "graph_rag": graph_rag,
    "playbook_lookup": playbook_lookup,
    "validate_quote": validate_quote,
    "get_contract_chunk": get_contract_chunk,
    "web_search": web_search,
}


__all__ = [
    "set_runtrace",
    "set_agent",
    "set_hypothesis",
    "register_contract",
    "vector_rag",
    "graph_rag",
    "playbook_lookup",
    "validate_quote",
    "get_contract_chunk",
    "web_search",
    "TOOL_SCHEMAS_ANALYZER",
    "TOOL_SCHEMAS_VALIDATOR",
    "TOOL_SCHEMAS_CONVERSATION",
    "TOOL_FUNCTIONS",
    "NDA_TO_HID",
    "HID_TO_NDA",
    "DATASET_CHOICE_TO_LABEL",
    "hypothesis_text",
]
