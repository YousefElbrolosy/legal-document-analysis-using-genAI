"""Build / ensure the ChromaDB vector index from the ContractNLI training split.

Lifted from `ms2/ms2-vector-indexing.ipynb` and parameterised so paths come
from `ms3.config` instead of hardcoded Kaggle paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb

from ms3.config import (
    CHROMA_COLLECTION,
    TRAIN_SPLIT_PATH,
    VECTOR_DB_PATH,
)
from ms3.models.embeddings import get_embedder


def _build_chunks(train_data: dict) -> tuple[list[str], list[dict], list[str]]:
    labels_map = train_data.get("labels", {})
    docs: list[str] = []
    metas: list[dict] = []
    ids: list[str] = []
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
                docs.append(
                    f"Hypothesis: '{hyp_text}'\n"
                    f"Historical Outcome: {choice}\n"
                    f"Supporting Clause: '{ev_text}'"
                )
                metas.append(
                    {"source_doc": doc_id, "hypothesis": hyp_id, "outcome": choice}
                )
                ids.append(f"{doc_id}_{chunk_idx}")
                chunk_idx += 1
    return docs, metas, ids


def ensure_index(force: bool = False) -> Path:
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    if force:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    if collection.count() > 0 and not force:
        print(f"[vector_index] reusing existing index ({collection.count()} items)")
        return VECTOR_DB_PATH

    with open(TRAIN_SPLIT_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    docs, metas, ids = _build_chunks(train_data)
    if not docs:
        raise RuntimeError("No structured chunks built — check train.json structure")
    print(f"[vector_index] embedding {len(docs)} chunks...")
    embeddings = get_embedder().encode(docs, show_progress_bar=True).tolist()
    batch = 5000
    for i in range(0, len(docs), batch):
        collection.add(
            documents=docs[i : i + batch],
            embeddings=embeddings[i : i + batch],
            ids=ids[i : i + batch],
            metadatas=metas[i : i + batch],
        )
        print(f"[vector_index] inserted batch {i // batch + 1}")
    print(f"[vector_index] stored at {VECTOR_DB_PATH}")
    return VECTOR_DB_PATH


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ensure", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.ensure or args.force:
        ensure_index(force=args.force)
