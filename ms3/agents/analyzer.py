"""Analyzer agent: per-hypothesis NLI + evidence + playbook application.

Memory: keeps the prior-hypothesis label summary on the same contract so
later hypotheses can be consistent (e.g. if H05 was ENTAILED, H06 should
usually not be CONTRADICTED about the same shared-with-third-parties text).

Tools: vector_rag, graph_rag, playbook_lookup, get_contract_chunk.
Feedback: the Validator can ask for one retry with a critique attached.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ms3.agents.base import run_tool_loop
from ms3.agents.tools import (
    TOOL_SCHEMAS_ANALYZER,
    set_agent,
    set_hypothesis,
)
from ms3.models.llm import chat
from ms3.playbook.loader import criticality, hypothesis_text

LABELS = ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")

SYSTEM = (
    "You are the Hypothesis-Analyzer agent in a multi-agent NDA review system. "
    "For one ContractNLI hypothesis at a time, you must:\n"
    "  1. Optionally call vector_rag or graph_rag to recall how similar clauses were historically interpreted.\n"
    "  2. Identify supporting (or counter) evidence quotes from the *target contract* (NOT the precedents).\n"
    "  3. Decide the label: ENTAILED, CONTRADICTED, or NOT_MENTIONED.\n"
    "  4. Call playbook_lookup(hypothesis_id, label) to get the deterministic severity/action.\n"
    "  5. Emit ONE final JSON object (no prose) with the schema given below.\n"
    "Rules:\n"
    " - Quotes in `evidence.supporting[*].quote` MUST be substrings of the contract text.\n"
    " - If label is NOT_MENTIONED, leave `evidence.supporting` empty.\n"
    " - Use the historical precedents only to understand interpretation; ground the decision in the target contract text.\n"
    " - `confidence` is a float in [0,1].\n"
)

FINAL_FORMAT = """Final JSON schema:
{
  "label": "ENTAILED" | "CONTRADICTED" | "NOT_MENTIONED",
  "confidence": 0.0-1.0,
  "evidence": {
    "supporting": [{"chunk_id": "...", "quote": "..."}, ...],
    "counter":    [{"chunk_id": "...", "quote": "..."}, ...]
  },
  "justification": {"claim": "...", "inference": "...", "limitations": "..."},
  "risk": {"severity":"LOW|MEDIUM|HIGH","recommended_action":"ACCEPT|CLARIFY|NEGOTIATE|ESCALATE","playbook_rule_ids":["H0X"],"criticality":"P0|P1|P2"}
}"""


def _user_prompt(
    contract_id: str,
    contract_chunks: list[dict[str, Any]],
    hid: str,
    prior_decisions: list[dict[str, Any]],
    validator_feedback: str | None,
) -> str:
    snippet = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in contract_chunks)
    if len(snippet) > 18000:
        snippet = snippet[:18000] + "\n…[truncated]"
    prior = (
        "\n".join(
            f"  - {d['hypothesis_id']}: {d['label']} (sev {d['severity']})"
            for d in prior_decisions
        )
        or "  (none yet)"
    )
    fb = ""
    if validator_feedback:
        fb = (
            "\nVALIDATOR FEEDBACK (you got this wrong last time — fix it):\n"
            f"  {validator_feedback}\n"
            "Try again with corrected evidence quotes.\n"
        )
    return (
        f"Contract id: {contract_id}\n"
        f"Hypothesis {hid} ({criticality(hid)}): {hypothesis_text(hid)}\n\n"
        f"Prior decisions on this contract:\n{prior}\n"
        f"{fb}\n"
        f"Contract text (chunked):\n{snippet}\n\n"
        f"{FINAL_FORMAT}"
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    # find first top-level object
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in analyzer output")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON braces in analyzer output")


def analyze_hypothesis(
    *,
    contract_id: str,
    contract_chunks: list[dict[str, Any]],
    hid: str,
    prior_decisions: list[dict[str, Any]],
    validator_feedback: str | None = None,
) -> dict[str, Any]:
    token_a = set_agent("analyzer_agent")
    token_h = set_hypothesis(hid)
    try:
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": _user_prompt(
                    contract_id, contract_chunks, hid, prior_decisions, validator_feedback
                ),
            },
        ]
        final_text, _ = run_tool_loop(messages, TOOL_SCHEMAS_ANALYZER, max_iters=6)
        try:
            return _extract_json(final_text)
        except Exception:
            # Last-ditch retry asking for just the JSON
            msg = chat(
                messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not contain parseable final JSON. "
                            "Output ONLY the JSON object now."
                        ),
                    }
                ],
                tools=None,
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            return _extract_json(msg.content or "{}")
    finally:
        set_hypothesis(None)
        set_agent("orchestrator")
        # restore tokens (not strictly needed but tidy)
        _ = (token_a, token_h)
