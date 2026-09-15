"""Lazy singleton for the sentence-transformers embedding model.

`all-MiniLM-L6-v2` is small enough to keep on CPU and shared across the
vector + graph retrievers.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ms3.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device="cpu")
