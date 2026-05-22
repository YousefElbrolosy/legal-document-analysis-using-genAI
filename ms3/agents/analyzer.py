"""Analyzer agent: per-hypothesis NLI + evidence + playbook application.

Memory: cumulative list of prior hypothesis labels on the same contract,
so later hypotheses can stay consistent.

Tools: vector_rag, graph_rag, playbook_lookup, get_contract_chunk.

One LLM tool-loop per hypothesis. The (previously costly) JSON-fallback
second call has been removed — if the model fails to produce parseable
JSON we return a NOT_MENTIONED stub and let the trace record the issue.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ms3.agents.base import run_tool_loop
from ms3.agents.tools import TOOL_SCHEMAS_ANALYZER, set_agent, set_hypothesis
from ms3.playbook.loader import criticality, hypothesis_text

LABELS = ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED")

# Cap how much contract text gets sent per hypothesis. Bigger numbers => more
# tokens per call. 8000 chars ≈ 2000 tokens is a reasonable middle ground.
MAX_CONTRACT_CHARS = 8000
# Cap how many tool rounds the model may make per hypothesis.
MAX_TOOL_ITERS = 3

SYSTEM = (
    "You are the Hypothesis-Analyzer agent. For ONE ContractNLI hypothesis:\n"
    "  1. (Optional) call vector_rag or graph_rag to recall similar precedents.\n"
    "  2. Decide the label (ENTAILED, CONTRADICTED, NOT_MENTIONED), citing\n"
    "     short verbatim quotes from the target contract as evidence.\n"
    "  3. Emit ONE final JSON object only (no prose) with the schema below.\n"
    "Quotes MUST be substrings of the contract text. If NOT_MENTIONED, leave\n"
    "evidence.supporting empty. Be concise."
)

FINAL_FORMAT = (
    'Final JSON schema:\n'
    '{"label":"ENTAILED|CONTRADICTED|NOT_MENTIONED","confidence":0.0-1.0,'
    '"evidence":{"supporting":[{"chunk_id":"...","quote":"..."}],"counter":[]},'
    '"justification":{"claim":"...","inference":"...","limitations":"..."}}'
)


def _user_prompt(
    contract_id: str,
    contract_chunks: list[dict[str, Any]],
    hid: str,
    prior_decisions: list[dict[str, Any]],
) -> str:
    snippet = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in contract_chunks)
    if len(snippet) > MAX_CONTRACT_CHARS:
        snippet = snippet[:MAX_CONTRACT_CHARS] + "\n…[truncated]"
    prior = (
        "\n".join(
            f"  - {d['hypothesis_id']}: {d['label']}" for d in prior_decisions[-8:]
        )
        or "  (none yet)"
    )
    return (
        f"Contract id: {contract_id}\n"
        f"Hypothesis {hid} ({criticality(hid)}): {hypothesis_text(hid)}\n\n"
        f"Recent prior decisions:\n{prior}\n\n"
        f"Contract text (chunked):\n{snippet}\n\n"
        f"{FINAL_FORMAT}"
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1)
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


_NOT_MENTIONED_STUB = {
    "label": "NOT_MENTIONED",
    "confidence": 0.3,
    "evidence": {"supporting": [], "counter": []},
    "justification": {
        "claim": "(analyzer produced no parseable JSON)",
        "inference": "default fallback",
        "limitations": "model output could not be parsed",
    },
    "risk": {
        "severity": "MEDIUM",
        "recommended_action": "CLARIFY",
        "playbook_rule_ids": [],
        "criticality": "P2",
    },
}


def analyze_hypothesis(
    *,
    contract_id: str,
    contract_chunks: list[dict[str, Any]],
    hid: str,
    prior_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    set_agent("analyzer_agent")
    set_hypothesis(hid)
    try:
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": _user_prompt(contract_id, contract_chunks, hid, prior_decisions),
            },
        ]
        final_text, _ = run_tool_loop(
            messages, TOOL_SCHEMAS_ANALYZER, max_iters=MAX_TOOL_ITERS
        )
        try:
            return _extract_json(final_text)
        except Exception:
            stub = dict(_NOT_MENTIONED_STUB)
            stub["risk"] = {**_NOT_MENTIONED_STUB["risk"], "playbook_rule_ids": [hid]}
            return stub
    finally:
        set_hypothesis(None)
        set_agent("orchestrator")
