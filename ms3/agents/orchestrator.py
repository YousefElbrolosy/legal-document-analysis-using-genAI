"""Orchestrator: LangGraph state machine that drives the Analyzer over the
17 hypotheses and owns the Runtrace lifecycle.

Hypothesis-analysis graph (simplified — no validator retry loop):

    orchestrator -> analyzer -> commit -> orchestrator -> ... -> END

`validate_draft()` is still called (pure code, zero LLM tokens) inside
`commit` so the runtrace carries quote_integrity_pass and validations[],
but its verdict no longer triggers a re-analyze.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ms3.agents.analyzer import analyze_hypothesis
from ms3.agents.tools import register_contract, set_agent, set_runtrace
from ms3.agents.validator import validate_draft
from ms3.playbook.loader import HID_TO_NDA, criticality, hypothesis_text
from ms3.runtrace.builder import Runtrace, now


class AnalysisState(TypedDict, total=False):
    contract_id: str
    contract_chunks: list[dict[str, Any]]
    gold_labels: dict[str, str]
    hypotheses_queue: list[str]
    current_hid: str | None
    draft: dict[str, Any] | None
    prior_decisions: list[dict[str, Any]]
    runtrace: Runtrace


def _node_orchestrator(state: AnalysisState) -> AnalysisState:
    set_agent("orchestrator")
    queue = state.get("hypotheses_queue") or []
    if not queue:
        state["current_hid"] = None
        return state
    state["current_hid"] = queue[0]
    state["hypotheses_queue"] = queue[1:]
    state["draft"] = None
    return state


def _node_analyzer(state: AnalysisState) -> AnalysisState:
    hid = state["current_hid"]
    state["draft"] = analyze_hypothesis(
        contract_id=state["contract_id"],
        contract_chunks=state["contract_chunks"],
        hid=hid,
        prior_decisions=state.get("prior_decisions", []),
    )
    return state


def _node_commit(state: AnalysisState) -> AnalysisState:
    hid = state["current_hid"]
    draft = state["draft"] or {}

    # Pure-code post-check (no LLM call) — feeds the runtrace's
    # validations[] / quote_integrity_pass fields. Never loops back.
    verdict = validate_draft(
        contract_id=state["contract_id"], hid=hid, draft=draft
    )
    expected = verdict.get("playbook_expected") or {}

    # Playbook is truth — override the LLM's risk fields with the
    # deterministic mapping.
    risk = {
        "severity": expected.get("severity") or draft.get("risk", {}).get("severity", "LOW"),
        "recommended_action": expected.get("recommended_action")
        or draft.get("risk", {}).get("recommended_action", "ACCEPT"),
        "playbook_rule_ids": [hid],
        "criticality": criticality(hid),
    }

    gold = state["gold_labels"].get(hid, "NOT_MENTIONED")
    label = draft.get("label", "NOT_MENTIONED")
    if label not in ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED"):
        label = "NOT_MENTIONED"

    supporting = [
        {
            "chunk_id": ev.get("chunk_id", "chunk_0"),
            "quote": ev.get("quote", "")[:1000],
            "relevance_score": float(ev.get("relevance_score", 0.8)),
        }
        for ev in (draft.get("evidence") or {}).get("supporting") or []
    ]
    counter = [
        {
            "chunk_id": ev.get("chunk_id", "chunk_0"),
            "quote": ev.get("quote", "")[:1000],
            "relevance_score": float(ev.get("relevance_score", 0.5)),
        }
        for ev in (draft.get("evidence") or {}).get("counter") or []
    ]

    justification = draft.get("justification") or {}
    decision = {
        "label": label,
        "confidence": float(draft.get("confidence", 0.7)),
        "evidence": {"supporting": supporting, "counter": counter},
        "justification": {
            "claim": justification.get("claim", "")[:600] or "(no claim provided)",
            "inference": justification.get("inference", "")[:600] or "(no inference provided)",
            "limitations": justification.get("limitations", "")[:600] or "(no limitations provided)",
        },
        "risk": risk,
    }

    qi_pass = all(
        v.get("status") in ("PASS", "WARN")
        for v in verdict.get("validations", [])
        if v.get("validator_id") == "quote_integrity"
    )
    evidence_required = label in ("ENTAILED", "CONTRADICTED")

    step_now = now()
    steps = [
        {
            "step_id": f"{hid}-1",
            "step_type": "agent_message",
            "producer": {"component": "analyzer_agent"},
            "started_at": step_now,
            "ended_at": step_now,
            "inputs": {"hypothesis_id": hid},
            "outputs": {"label": label, "confidence": decision["confidence"]},
        },
        {
            "step_id": f"{hid}-2",
            "step_type": "consistency_check",
            "producer": {"component": "validator"},
            "started_at": step_now,
            "ended_at": step_now,
            "inputs": {"draft_label": label},
            "outputs": {
                "pass": verdict.get("pass"),
                "validations": verdict.get("validations", []),
            },
        },
        {
            "step_id": f"{hid}-3",
            "step_type": "playbook_map",
            "producer": {"component": "playbook"},
            "started_at": step_now,
            "ended_at": step_now,
            "inputs": {"hypothesis_id": hid, "label": label},
            "outputs": risk,
        },
    ]

    rt: Runtrace = state["runtrace"]
    rt.add_hypothesis_trace(
        {
            "hypothesis_id": hid,
            "dataset_hypothesis_key": HID_TO_NDA[hid],
            "hypothesis_text": hypothesis_text(hid),
            "gold_label": gold,
            "latency_ms": 0.0,
            "retrieval": {
                "mode": rt.retriever_mode,
                "query": hypothesis_text(hid),
                "context": [],
            },
            "compliant_evidence_required": evidence_required,
            "quote_integrity_pass": qi_pass,
            "retry_count": 0,
            "steps": steps,
            "decision": decision,
            "validations": verdict.get("validations", []),
        }
    )

    state.setdefault("prior_decisions", []).append(
        {"hypothesis_id": hid, "label": label, "severity": risk["severity"]}
    )
    return state


def _route_after_orchestrator(state: AnalysisState) -> str:
    return "analyzer" if state.get("current_hid") else END


def build_analysis_graph():
    g = StateGraph(AnalysisState)
    g.add_node("orchestrator", _node_orchestrator)
    g.add_node("analyzer", _node_analyzer)
    g.add_node("commit", _node_commit)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"analyzer": "analyzer", END: END},
    )
    g.add_edge("analyzer", "commit")
    g.add_edge("commit", "orchestrator")
    return g.compile()


def run_hypothesis_analysis(
    contract: dict[str, Any],
    gold_labels: dict[str, str],
    retriever_mode: str,
) -> Runtrace:
    register_contract(contract)
    rt = Runtrace(
        mode="hypothesis_analysis",
        retriever_mode=retriever_mode,
        contract=contract,
    )
    token = set_runtrace(rt)
    try:
        graph = build_analysis_graph()
        state: AnalysisState = {
            "contract_id": contract["contract_id"],
            "contract_chunks": contract["chunks"],
            "gold_labels": gold_labels,
            "hypotheses_queue": [f"H{n:02d}" for n in range(1, 18)],
            "prior_decisions": [],
            "runtrace": rt,
        }
        graph.invoke(state, {"recursion_limit": 200})
    finally:
        set_runtrace(None)
        _ = token
    return rt
