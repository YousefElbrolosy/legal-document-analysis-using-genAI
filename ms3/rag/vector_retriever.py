"""Vector RAG retriever — Chroma + sentence-transformers.

Same retrieval contract as MS2: returns
`[{"source": "vector_db", "text": str, "metadata": dict, "score": float}, ...]`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb

from ms3.config import (
    CHROMA_COLLECTION,
    DEFAULT_VECTOR_TOP_K,
    VECTOR_DB_PATH,
)
from ms3.models.embeddings import get_embedder
from ms3.rag.vector_index import ensure_index


class VectorRetriever:
    def __init__(self, index_path: str | None = None, collection_name: str = CHROMA_COLLECTION):
        if index_path is None:
            ensure_index()
            index_path = str(VECTOR_DB_PATH)
        self.index_path = index_path
        self.client = chromadb.PersistentClient(path=index_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.embed_model = get_embedder()

    def retrieve(self, query: str, top_k: int = DEFAULT_VECTOR_TOP_K) -> list[dict[str, Any]]:
        q_emb = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=top_k)
        out: list[dict[str, Any]] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[None] * len(docs[0])]
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            score = None if dist is None else float(1.0 - dist)
            out.append(
                {
                    "source": "vector_db",
                    "text": doc,
                    "metadata": meta,
                    "score": score,
                }
            )
        return out


@lru_cache(maxsize=1)
def get_vector_retriever() -> VectorRetriever:
    return VectorRetriever()
