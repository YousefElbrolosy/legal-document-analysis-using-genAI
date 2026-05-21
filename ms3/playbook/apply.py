"""Deterministic MS1 playbook application.

Given an NLI label and a hypothesis id, returns the playbook-driven severity,
recommended_action, criticality, and the playbook_rule_ids that produced it.
This is *pure code* — no LLM call — and runs identically to the MS1 logic
the spec requires us to reuse without edits.
"""
from __future__ import annotations

from typing import Any

from ms3.playbook.loader import get_checks, load_playbook


def apply(hypothesis_id: str, label: str) -> dict[str, Any]:
    pb = load_playbook()
    gd = pb["global_defaults"]
    default = gd["label_to_default_decision"].get(label)
    if default is None:
        raise ValueError(f"unknown label: {label}")
    severity = default["severity"]
    action = default["action"]

    check = get_checks().get(hypothesis_id)
    if check is None:
        raise ValueError(f"unknown hypothesis_id: {hypothesis_id}")
    overrides = check.get("overrides", {}) or {}
    if label in overrides:
        severity = overrides[label].get("severity", severity)
        action = overrides[label].get("action", action)

    return {
        "severity": severity,
        "recommended_action": action,
        "criticality": check["criticality"],
        "playbook_rule_ids": [hypothesis_id],
        "status": gd["label_to_status"][label],
        "evidence_required": label in gd["evidence_required_for"],
    }
