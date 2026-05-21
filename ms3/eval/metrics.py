"""MS1-parity metrics. Names match `evaluation_metrics.csv` exactly:
`label_accuracy`, `groundedness`, `quote_integrity_pass_rate`, `avg_latency_ms`.
"""
from __future__ import annotations

from typing import Any


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
        for h in rt.get("hypothesis_traces", []):
            total += 1
            if h["decision"]["label"] == h["gold_label"]:
                correct += 1
            if (not h.get("compliant_evidence_required")) or len(
                h["decision"]["evidence"]["supporting"]
            ) >= 1:
                compliant += 1
            if h.get("quote_integrity_pass"):
                qi += 1
        latencies.append(float(rt.get("metrics", {}).get("contract_latency_ms", 0.0)))
    return {
        "label_accuracy": round(correct / total, 6) if total else 0.0,
        "groundedness": round(compliant / total, 6) if total else 0.0,
        "quote_integrity_pass_rate": round(qi / total, 6) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
    }
