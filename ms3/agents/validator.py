"""Pure-code post-check called from the orchestrator's `commit` node.

NOT an LLM agent. Just a Python function that:
  1. Substring-checks every `evidence.supporting[*].quote` against the
     contract chunks (quote integrity).
  2. Looks up the deterministic playbook severity/action for the predicted
     label.
  3. Confirms ENTAILED/CONTRADICTED labels carry at least one supporting
     quote (evidence-required).

Returns a dict consumed by `_node_commit`:
  {
    "pass": bool,
    "validations": [{validator_id, status, message, related_hypothesis_id}, ...],
    "playbook_expected": {severity, recommended_action, criticality, ...},
  }
"""
from __future__ import annotations

from typing import Any

from ms3.agents.tools import playbook_lookup, validate_quote


def validate_draft(
    *, contract_id: str, hid: str, draft: dict[str, Any]
) -> dict[str, Any]:
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

    # 2. Playbook lookup — used by commit to override risk fields.
    label = draft.get("label")
    expected = playbook_lookup(hypothesis_id=hid, label=label)

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
        "validations": validations,
        "playbook_expected": expected,
    }
