"""Conversation agent: stateful legal assistant for one contract.

Memory: full chat history (system + user/assistant pairs) for the session.
Tools: vector_rag, graph_rag, web_search, get_contract_chunk.
Feedback: every user follow-up acts as feedback to the prior turn.
"""
from __future__ import annotations

from typing import Any

from ms3.agents.base import run_tool_loop
from ms3.agents.tools import TOOL_SCHEMAS_CONVERSATION, set_agent

SYSTEM = (
    "You are an expert legal aide reviewing an uploaded NDA for the user. "
    "You have access to:\n"
    "  - vector_rag / graph_rag: historical interpretations of similar clauses,\n"
    "  - get_contract_chunk: fetch any chunk of the active contract by id,\n"
    "  - web_search: external sources (use SPARINGLY and ONLY when the user explicitly asks for outside information or current statute/case-law).\n"
    "Rules:\n"
    " - Ground every concrete claim about the contract in actual quoted text from the contract.\n"
    " - When you cite external sources, include the URL.\n"
    " - Be concise and insightful — short, citation-rich answers beat long, vague ones."
)


class ConversationSession:
    def __init__(self, contract_id: str, contract_text_preview: str):
        self.contract_id = contract_id
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Active contract id: {contract_id}\n"
                    f"Contract text (truncated):\n{contract_text_preview[:8000]}"
                    f"{'…[truncated]' if len(contract_text_preview) > 8000 else ''}\n\n"
                    "Acknowledge briefly and wait for my questions."
                ),
            },
        ]
        set_agent("conversation_agent")
        bootstrap_reply, self.messages = run_tool_loop(
            self.messages,
            tools=TOOL_SCHEMAS_CONVERSATION,
            max_iters=2,
            max_tokens=200,
        )
        self.bootstrap_reply = bootstrap_reply

    def ask(self, user_text: str) -> str:
        set_agent("conversation_agent")
        self.messages.append({"role": "user", "content": user_text})
        reply, self.messages = run_tool_loop(
            self.messages,
            tools=TOOL_SCHEMAS_CONVERSATION,
            max_iters=6,
        )
        return reply
