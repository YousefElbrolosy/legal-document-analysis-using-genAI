"""MS1-parity metrics. Names match `evaluation_metrics.csv` exactly:
`label_accuracy`, `groundedness`, `quote_integrity_pass_rate`, `avg_latency_ms`.

Per-contract wall-clock latency is derived from `run.ended_at - run.started_at`
(the existing `metrics.contract_latency_ms` field is zero — see note in
ms3/agents/orchestrator.py's `_node_commit`).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_iso_ms(ts: str) -> float:
    """Parse the runtrace ISO timestamp (`...Z`) and return epoch milliseconds."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).timestamp() * 1000.0


def _contract_wallclock_ms(rt: dict[str, Any]) -> float:
    run = rt.get("run") or {}
    started = run.get("started_at")
    ended = run.get("ended_at")
    if not started or not ended:
        return 0.0
    try:
        return max(0.0, _parse_iso_ms(ended) - _parse_iso_ms(started))
    except Exception:
        return 0.0


def aggregate(runtraces: list[dict[str, Any]]) -> dict[str, float]:
    if not runtraces:
        return {
            "label_accuracy": 0.0,
            "groundedness": 0.0,
            "quote_integrity_pass_rate": 0.0,
            "avg_latency_ms": 0.0,
        }
    total = correct = compliant = qi = 0
    latencies: list[float] = []
    for rt in runtraces:
        hypothesis_traces = rt.get("hypothesis_traces", [])
        # Skip conversation-mode runtraces — they have no hypothesis_traces.
        if not hypothesis_traces:
            continue
        for h in hypothesis_traces:
            total += 1
            if h["decision"]["label"] == h["gold_label"]:
                correct += 1
            if (not h.get("compliant_evidence_required")) or len(
                h["decision"]["evidence"]["supporting"]
            ) >= 1:
                compliant += 1
            if h.get("quote_integrity_pass"):
                qi += 1
        latencies.append(_contract_wallclock_ms(rt))
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    return {
        "label_accuracy": round(correct / total, 6) if total else 0.0,
        "groundedness": round(compliant / total, 6) if total else 0.0,
        "quote_integrity_pass_rate": round(qi / total, 6) if total else 0.0,
        "avg_latency_ms": avg_latency,
    }
