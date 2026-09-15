"""Build / ensure the NetworkX graph index from the ContractNLI training split.

Lifted from `ms2/ms2-graph-indexing.ipynb`.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx

from ms3.config import GRAPH_DB_PATH, TRAIN_SPLIT_PATH


def _build_graph(train_data: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    labels_map = train_data.get("labels", {})
    chunk_idx = 0
    for doc in train_data.get("documents", []):
        doc_id = str(doc["id"])
        text = doc["text"]
        doc_spans = doc.get("spans", [])
        annotations = doc.get("annotation_sets", [{}])[0].get("annotations", {})
        for hyp_id, ann in annotations.items():
            choice = ann.get("choice")
            span_indices = ann.get("spans", [])
            hyp_text = labels_map.get(hyp_id, {}).get("hypothesis", "")
            if choice not in ("Entailment", "Contradiction") or not span_indices:
                continue
            for idx in span_indices:
                if idx >= len(doc_spans):
                    continue
                span_val = doc_spans[idx]
                if isinstance(span_val, list) and len(span_val) == 2:
                    ev_text = text[span_val[0] : span_val[1]]
                elif isinstance(span_val, str):
                    ev_text = span_val
                else:
                    continue
                clause_id = f"Clause_{doc_id}_{chunk_idx}"
                rule_id = f"Rule_{hyp_id}"
                G.add_node(clause_id, type="Clause", text=ev_text, source_doc=doc_id)
                if not G.has_node(rule_id):
                    G.add_node(rule_id, type="Hypothesis", text=hyp_text)
                G.add_edge(clause_id, rule_id, relation=choice)
                chunk_idx += 1
    return G


def ensure_index(force: bool = False) -> Path:
    GRAPH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if GRAPH_DB_PATH.exists() and not force:
        print(f"[graph_index] reusing existing graph at {GRAPH_DB_PATH}")
        return GRAPH_DB_PATH
    with open(TRAIN_SPLIT_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    G = _build_graph(train_data)
    with open(GRAPH_DB_PATH, "wb") as f:
        pickle.dump(G, f)
    print(
        f"[graph_index] built {G.number_of_nodes()} nodes / "
        f"{G.number_of_edges()} edges; saved to {GRAPH_DB_PATH}"
    )
    return GRAPH_DB_PATH


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ensure", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.ensure or args.force:
        ensure_index(force=args.force)
