"""Graph RAG retriever — NetworkX + embedding-based rule entry.

Same retrieval contract as the vector retriever; `source: "graph_db"`.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import networkx as nx
from sentence_transformers import util

from ms3.config import (
    DEFAULT_GRAPH_CLAUSES_PER_RULE,
    DEFAULT_GRAPH_TOP_K_RULES,
    GRAPH_DB_PATH,
)
from ms3.models.embeddings import get_embedder
from ms3.rag.graph_index import ensure_index


class GraphRetriever:
    def __init__(self, graph_path: str | None = None):
        if graph_path is None:
            ensure_index()
            graph_path = str(GRAPH_DB_PATH)
        self.graph_path = graph_path
        with open(graph_path, "rb") as f:
            self.G: nx.DiGraph = pickle.load(f)
        self.embed_model = get_embedder()
        self.rule_nodes = [
            n for n, attr in self.G.nodes(data=True) if attr.get("type") == "Hypothesis"
        ]
        self.rule_texts = [self.G.nodes[n].get("text", "") for n in self.rule_nodes]
        self.rule_embeddings = (
            self.embed_model.encode(self.rule_texts) if self.rule_texts else None
        )

    def retrieve(
        self,
        query: str,
        top_k_rules: int = DEFAULT_GRAPH_TOP_K_RULES,
        clauses_per_rule: int = DEFAULT_GRAPH_CLAUSES_PER_RULE,
    ) -> list[dict[str, Any]]:
        if not self.rule_nodes:
            return []
        q_emb = self.embed_model.encode(query)
        similarities = util.cos_sim(q_emb, self.rule_embeddings)[0]
        k = min(top_k_rules, len(self.rule_nodes))
        top_indices = similarities.topk(k=k).indices.tolist()
        out: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            rule_node = self.rule_nodes[idx]
            rule_text = self.rule_texts[idx]
            score = float(similarities[idx])
            for clause_node in list(self.G.predecessors(rule_node))[:clauses_per_rule]:
                clause_data = self.G.nodes[clause_node]
                relation = self.G.get_edge_data(clause_node, rule_node).get(
                    "relation", "Unknown"
                )
                out.append(
                    {
                        "source": "graph_db",
                        "text": (
                            f"Graph Traversal -> Rule: '{rule_text}'\n"
                            f"Historical Outcome: {relation}\n"
                            f"Supporting Clause: '{clause_data.get('text', '')}'"
                        ),
                        "metadata": {
                            "rule": rule_node,
                            "clause": clause_node,
                            "relation": relation,
                            "rank": rank,
                        },
                        "score": score,
                    }
                )
        return out


@lru_cache(maxsize=1)
def get_graph_retriever() -> GraphRetriever:
    return GraphRetriever()
