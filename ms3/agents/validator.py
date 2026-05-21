"""Validator agent: audits the Analyzer's draft for quote-integrity and
playbook-consistency. Returns PASS or FAIL+reason, which the orchestrator
uses to decide whether to ask the Analyzer for one retry.
"""
from __future__ import annotations

import json
from typing import Any

from ms3.agents.tools import set_agent, validate_quote, playbook_lookup


def validate_draft(
    *, contract_id: str, hid: str, draft: dict[str, Any]
) -> dict[str, Any]:
    """Pure code (no LLM call) — quote-integrity + playbook consistency."""
    set_agent("validator_agent")
    try:
        validations: list[dict[str, Any]] = []
        ok = True

        # 1. Quote integrity for supporting evidence
        supporting = (draft.get("evidence") or {}).get("supporting") or []
        bad_quotes: list[str] = []
        for ev in supporting:
            q = ev.get("quote", "")
            res = validate_quote(contract_id=contract_id, quote=q)
            if not res.get("ok"):
                bad_quotes.append(q[:80])
        if bad_quotes:
            ok = False
            validations.append(
                {
                    "validator_id": "quote_integrity",
                    "status": "FAIL",
                    "message": f"{len(bad_quotes)} quote(s) not found in contract: {bad_quotes}",
                    "related_hypothesis_id": hid,
                }
            )
        else:
            validations.append(
                {
                    "validator_id": "quote_integrity",
                    "status": "PASS",
                    "message": "all supporting quotes are substrings of contract chunks",
                    "related_hypothesis_id": hid,
                }
            )

        # 2. Playbook consistency (severity/action must match deterministic rule)
        label = draft.get("label")
        expected = playbook_lookup(hypothesis_id=hid, label=label)
        risk = draft.get("risk") or {}
        if (
            risk.get("severity") != expected["severity"]
            or risk.get("recommended_action") != expected["recommended_action"]
        ):
            validations.append(
                {
                    "validator_id": "playbook_consistency",
                    "status": "WARN",
                    "message": (
                        f"risk fields disagree with playbook (got "
                        f"{risk.get('severity')}/{risk.get('recommended_action')}, "
                        f"expected {expected['severity']}/{expected['recommended_action']})"
                    ),
                    "related_hypothesis_id": hid,
                }
            )

        # 3. Evidence required when label demands it
        if expected["evidence_required"] and not supporting:
            ok = False
            validations.append(
                {
                    "validator_id": "evidence_required",
                    "status": "FAIL",
                    "message": f"label {label} requires at least one supporting evidence quote",
                    "related_hypothesis_id": hid,
                }
            )

        return {
            "pass": ok,
            "feedback": json.dumps(
                [v for v in validations if v["status"] == "FAIL"], ensure_ascii=False
            )
            if not ok
            else "",
            "validations": validations,
            "playbook_expected": expected,
        }
    finally:
        set_agent("orchestrator")
