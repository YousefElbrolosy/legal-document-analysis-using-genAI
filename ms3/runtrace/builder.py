"""Incremental runtrace builder.

Agents (and the @traced_tool wrapper) push events to a single `Runtrace`
instance held in the LangGraph state. At write time `finalize()` computes
the summaries and metrics and `validate.py` checks the result against
`assets/runtrace_ms3.schema.json`.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ms3.config import (
    EMBED_MODEL,
    MAX_TOKENS,
    MODEL,
    SEED,
    TEMPERATURE,
    TOP_P,
    VECTOR_DB_PATH,
    GRAPH_DB_PATH,
)
from ms3.playbook.loader import load_playbook, ruleset_hash


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Runtrace:
    """Mutable accumulator. Use `to_dict()` to get the schema-shaped object."""

    def __init__(
        self,
        mode: str,
        retriever_mode: str,
        contract: dict[str, Any],
        run_id: str | None = None,
    ):
        assert mode in ("hypothesis_analysis", "conversation")
        assert retriever_mode in ("vector", "graph")
        self.run_id = run_id or f"ms3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        self.started_at = now()
        self.ended_at: str | None = None
        self.mode = mode
        self.retriever_mode = retriever_mode
        self.contract = contract
        self.conversation_history: list[dict[str, Any]] = []
        self.hypothesis_traces: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.run_validations: list[dict[str, Any]] = []
        pb = load_playbook()
        self.playbook_meta = {
            "playbook_id": pb["playbook_id"],
            "schema_version": "3.0-ms3",
            "schema_file": "assets/runtrace_ms3.schema.json",
            "version": str(pb["version"]),
            "ruleset_hash": ruleset_hash(),
        }

    # --- event accumulators ---------------------------------------------
    def add_tool_call(
        self,
        *,
        tool_name: str,
        agent: str,
        args: dict[str, Any],
        output: Any,
        started_at: str,
        ended_at: str,
        latency_ms: float,
        error: str | None = None,
        related_hypothesis_id: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "agent": agent,
            "args": args,
            "output": output,
            "started_at": started_at,
            "ended_at": ended_at,
            "latency_ms": latency_ms,
        }
        if error is not None:
            entry["error"] = error
        if related_hypothesis_id is not None:
            entry["related_hypothesis_id"] = related_hypothesis_id
        self.tool_calls.append(entry)

    def add_turn(
        self,
        role: str,
        content: str,
        retrieval_mode: str | None = None,
        retrieved_context: list[str] | None = None,
    ) -> None:
        turn: dict[str, Any] = {
            "turn_id": len(self.conversation_history) + 1,
            "role": role,
            "content": content,
        }
        if retrieval_mode:
            turn["retrieval_mode"] = retrieval_mode
        if retrieved_context is not None:
            turn["retrieved_context"] = retrieved_context
        self.conversation_history.append(turn)

    def add_hypothesis_trace(self, trace: dict[str, Any]) -> None:
        self.hypothesis_traces.append(trace)

    def add_run_validation(self, v: dict[str, Any]) -> None:
        self.run_validations.append(v)

    # --- finalize -------------------------------------------------------
    def _label_counts(self) -> dict[str, int]:
        counts = {"ENTAILED": 0, "CONTRADICTED": 0, "NOT_MENTIONED": 0}
        for h in self.hypothesis_traces:
            counts[h["decision"]["label"]] += 1
        return counts

    def _metrics(self) -> dict[str, Any]:
        n = len(self.hypothesis_traces)
        if n == 0:
            return {
                "hypothesis_count": 0,
                "label_counts": {"ENTAILED": 0, "CONTRADICTED": 0, "NOT_MENTIONED": 0},
            }
        correct = sum(1 for h in self.hypothesis_traces if h["decision"]["label"] == h["gold_label"])
        compliant = sum(
            1
            for h in self.hypothesis_traces
            if (not h["compliant_evidence_required"])
            or len(h["decision"]["evidence"]["supporting"]) >= 1
        )
        qi = sum(1 for h in self.hypothesis_traces if h["quote_integrity_pass"])
        total_latency = sum(h["latency_ms"] for h in self.hypothesis_traces)
        return {
            "hypothesis_count": n,
            "correct_count": correct,
            "compliant_count": compliant,
            "quote_integrity_count": qi,
            "contract_accuracy": round(correct / n, 6),
            "groundedness_rate": round(compliant / n, 6),
            "quote_integrity_rate": round(qi / n, 6),
            "contract_latency_ms": round(total_latency, 3),
            "label_counts": self._label_counts(),
        }

    def _summaries(self) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        total: dict[str, int] = {}
        by_agent: dict[str, dict[str, int]] = {}
        for c in self.tool_calls:
            total[c["tool_name"]] = total.get(c["tool_name"], 0) + 1
            by_agent.setdefault(c["agent"], {})
            by_agent[c["agent"]][c["tool_name"]] = (
                by_agent[c["agent"]].get(c["tool_name"], 0) + 1
            )
        return total, by_agent

    def to_dict(self) -> dict[str, Any]:
        if self.ended_at is None:
            self.ended_at = now()
        total, by_agent = self._summaries()
        return {
            "schema_version": "3.0-ms3",
            "run": {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "framework": "multi_agent_system",
                "mode": self.mode,
                "parameters": {
                    "base_model": MODEL,
                    "adapter_method": "none",
                    "quantization": "none",
                    "embedding_model": EMBED_MODEL,
                    "seed": SEED,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_seq_length": MAX_TOKENS,
                    "provider": "openrouter",
                },
            },
            "retrieval_strategy": {
                "mode": self.retriever_mode,
                "knowledge_base_source": "ContractNLI-train",
                "selector": f"CLI flag --retriever={self.retriever_mode}",
                "exactly_one_branch_per_run": True,
                "external_context_role": "historical_precedent_only",
                "index_reference": {
                    "index_type": "chroma_vector_db" if self.retriever_mode == "vector" else "networkx_graph",
                    "path": str(VECTOR_DB_PATH if self.retriever_mode == "vector" else GRAPH_DB_PATH),
                },
            },
            "conversation_history": self.conversation_history,
            "contract": self.contract,
            "playbook": self.playbook_meta,
            "hypothesis_traces": self.hypothesis_traces,
            "tool_calls": self.tool_calls,
            "tool_call_summary": total,
            "tool_call_summary_by_agent": by_agent,
            "metrics": self._metrics(),
            "run_validations": self.run_validations,
        }


def write_runtrace(rt: Runtrace, path: Path) -> Path:
    from ms3.runtrace.validate import validate_runtrace

    obj = rt.to_dict()
    validate_runtrace(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
