"""Load a ContractNLI document into the runtrace-schema `contract` block.

Supports three contract specifiers:
  - "test:N"  -> the N-th document in assets/test.json
  - "dev:N"   -> the N-th document in assets/dev.json
  - path/to/file.{txt,pdf,docx,json}  -> external upload
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ms3.config import DEV_SPLIT_PATH, TEST_SPLIT_PATH
from ms3.playbook.loader import DATASET_CHOICE_TO_LABEL, NDA_TO_HID


def _chunk(text: str, max_len: int = 1200) -> list[dict[str, Any]]:
    """Simple paragraph-aware chunker — preserves char spans."""
    chunks: list[dict[str, Any]] = []
    i, n = 0, len(text)
    idx = 0
    while i < n:
        end = min(i + max_len, n)
        # backtrack to a paragraph or sentence boundary if possible
        if end < n:
            nl = text.rfind("\n\n", i, end)
            if nl > i + max_len // 2:
                end = nl
            else:
                dot = text.rfind(". ", i, end)
                if dot > i + max_len // 2:
                    end = dot + 1
        chunks.append(
            {
                "chunk_id": f"chunk_{idx}",
                "text": text[i:end],
                "span": {"char_start": i, "char_end": end},
            }
        )
        idx += 1
        i = end
    return chunks


def _doc_to_contract(doc: dict[str, Any], split_name: str) -> tuple[dict[str, Any], dict[str, str]]:
    text = doc["text"]
    contract = {
        "contract_id": str(doc.get("id", "ad-hoc")),
        "source_type": "txt",
        "source_name": doc.get("file_name", "unknown"),
        "split": split_name,
        "language": "en",
        "hash_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chunks": _chunk(text),
    }
    gold: dict[str, str] = {}
    annotations = doc.get("annotation_sets", [{}])[0].get("annotations", {})
    for nda_key, ann in annotations.items():
        hid = NDA_TO_HID.get(nda_key)
        if not hid:
            continue
        gold[hid] = DATASET_CHOICE_TO_LABEL.get(ann.get("choice", "NotMentioned"), "NOT_MENTIONED")
    # default missing hypotheses to NOT_MENTIONED so all 17 are present
    for hid in (f"H{n:02d}" for n in range(1, 18)):
        gold.setdefault(hid, "NOT_MENTIONED")
    return contract, gold


def load_contract(spec: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Returns (contract_block, gold_labels_by_hid)."""
    if spec.startswith("test:") or spec.startswith("dev:"):
        split, idx_s = spec.split(":", 1)
        path = TEST_SPLIT_PATH if split == "test" else DEV_SPLIT_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data["documents"]
        return _doc_to_contract(docs[int(idx_s)], "test" if split == "test" else "dev")

    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(spec)
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _doc_to_contract(data["documents"][0], "ad-hoc")
    text = p.read_text(encoding="utf-8", errors="ignore")
    contract = {
        "contract_id": p.stem,
        "source_type": "txt",
        "source_name": p.name,
        "split": "ad-hoc",
        "language": "en",
        "hash_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chunks": _chunk(text),
    }
    gold = {f"H{n:02d}": "NOT_MENTIONED" for n in range(1, 18)}
    return contract, gold


def list_contracts(split: str, limit: int | None = None) -> list[int]:
    path = TEST_SPLIT_PATH if split == "test" else DEV_SPLIT_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data["documents"])
    if limit is not None:
        n = min(n, limit)
    return list(range(n))
