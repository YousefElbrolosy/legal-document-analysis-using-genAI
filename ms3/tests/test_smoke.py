"""Pure-python smoke tests — do not require network or heavy ML deps."""
from __future__ import annotations

import json
import sys
import types

import pytest


# Stub heavy deps so the modules import without pip install -r requirements.txt
def _stub_heavy_deps():
    if "sentence_transformers" not in sys.modules:
        class _Dummy:
            def __init__(self, *a, **k): pass
            def encode(self, *a, **k):
                import numpy as np
                return np.zeros((len(a[0]) if a and hasattr(a[0], "__len__") and not isinstance(a[0], str) else 1, 8))

        m = types.ModuleType("sentence_transformers")
        m.SentenceTransformer = _Dummy
        m.util = types.SimpleNamespace(cos_sim=lambda a, b: __import__("numpy").zeros((1, 1)))
        sys.modules["sentence_transformers"] = m
    if "chromadb" not in sys.modules:
        m = types.ModuleType("chromadb")
        m.PersistentClient = lambda *a, **k: None
        sys.modules["chromadb"] = m
    if "openai" not in sys.modules:
        m = types.ModuleType("openai")
        m.OpenAI = lambda *a, **k: None
        sys.modules["openai"] = m
    if "ollama" not in sys.modules:
        m = types.ModuleType("ollama")
        m.Client = lambda *a, **k: None
        sys.modules["ollama"] = m
    if "langgraph" not in sys.modules:
        sys.modules["langgraph"] = types.ModuleType("langgraph")
    if "langgraph.graph" not in sys.modules:
        m = types.ModuleType("langgraph.graph")
        m.END = "__end__"
        class _SG:
            def __init__(self, *a, **k): pass
            def add_node(self, *a, **k): pass
            def add_edge(self, *a, **k): pass
            def add_conditional_edges(self, *a, **k): pass
            def set_entry_point(self, *a, **k): pass
            def compile(self): return self
            def invoke(self, *a, **k): return a[0] if a else {}
        m.StateGraph = _SG
        sys.modules["langgraph.graph"] = m


_stub_heavy_deps()


def test_playbook_loads_and_hashes():
    from ms3.playbook.loader import get_checks, ruleset_hash
    assert len(get_checks()) == 17
    assert len(ruleset_hash()) == 64


def test_playbook_apply_h04_not_mentioned_is_high_clarify():
    from ms3.playbook.apply import apply
    r = apply("H04", "NOT_MENTIONED")
    assert r["severity"] == "HIGH"
    assert r["recommended_action"] == "CLARIFY"
    assert r["criticality"] == "P0"


def test_playbook_apply_h01_entailed_defaults_to_accept():
    from ms3.playbook.apply import apply
    r = apply("H01", "ENTAILED")
    assert r["severity"] == "LOW"
    assert r["recommended_action"] == "ACCEPT"


def test_contract_loader_test_zero():
    from ms3.contract_loader import load_contract
    c, gold = load_contract("test:0")
    assert c["contract_id"]
    assert c["chunks"]
    assert len(gold) == 17
    assert all(g in ("ENTAILED", "CONTRADICTED", "NOT_MENTIONED") for g in gold.values())


def test_runtrace_schema_validates_minimal_chat_run():
    """A conversation-mode trace with 0 hypotheses must validate."""
    from ms3.runtrace.builder import Runtrace
    from ms3.runtrace.validate import validate_runtrace

    contract = {
        "contract_id": "demo",
        "source_type": "txt",
        "hash_sha256": "a" * 64,
        "chunks": [{"chunk_id": "c0", "text": "hello world", "span": {"char_start": 0, "char_end": 11}}],
    }
    rt = Runtrace(mode="conversation", retriever_mode="vector", contract=contract)
    rt.add_turn("user", "Hi")
    rt.add_turn("assistant", "Hello!")
    validate_runtrace(rt.to_dict())


def test_runtrace_schema_rejects_hypothesis_run_with_fewer_than_17():
    from ms3.runtrace.builder import Runtrace
    from ms3.runtrace.validate import validate_runtrace

    contract = {
        "contract_id": "demo",
        "source_type": "txt",
        "hash_sha256": "a" * 64,
        "chunks": [{"chunk_id": "c0", "text": "x", "span": {"char_start": 0, "char_end": 1}}],
    }
    rt = Runtrace(mode="hypothesis_analysis", retriever_mode="vector", contract=contract)
    with pytest.raises(ValueError):
        validate_runtrace(rt.to_dict())


def test_validator_quote_integrity_detects_bad_quote():
    from ms3.agents.tools import register_contract
    from ms3.agents.validator import validate_draft

    contract = {
        "contract_id": "vd1",
        "source_type": "txt",
        "hash_sha256": "a" * 64,
        "chunks": [
            {
                "chunk_id": "c0",
                "text": "The Receiving Party shall not disclose any Confidential Information.",
                "span": {"char_start": 0, "char_end": 67},
            }
        ],
    }
    register_contract(contract)
    draft = {
        "label": "ENTAILED",
        "confidence": 0.9,
        "evidence": {
            "supporting": [
                {"chunk_id": "c0", "quote": "shall not disclose any Confidential Information"},
                {"chunk_id": "c0", "quote": "this quote does not exist in the contract"},
            ],
            "counter": [],
        },
        "justification": {"claim": "x", "inference": "y", "limitations": "z"},
        "risk": {"severity": "LOW", "recommended_action": "ACCEPT", "playbook_rule_ids": ["H01"], "criticality": "P2"},
    }
    verdict = validate_draft(contract_id="vd1", hid="H01", draft=draft)
    assert verdict["pass"] is False
    assert any(v["validator_id"] == "quote_integrity" and v["status"] == "FAIL" for v in verdict["validations"])


def test_validator_passes_when_all_quotes_valid():
    from ms3.agents.tools import register_contract
    from ms3.agents.validator import validate_draft

    contract = {
        "contract_id": "vd2",
        "source_type": "txt",
        "hash_sha256": "a" * 64,
        "chunks": [
            {
                "chunk_id": "c0",
                "text": "The Receiving Party shall not disclose any Confidential Information.",
                "span": {"char_start": 0, "char_end": 67},
            }
        ],
    }
    register_contract(contract)
    draft = {
        "label": "ENTAILED",
        "confidence": 0.9,
        "evidence": {
            "supporting": [{"chunk_id": "c0", "quote": "shall not disclose"}],
            "counter": [],
        },
        "justification": {"claim": "x", "inference": "y", "limitations": "z"},
        "risk": {"severity": "LOW", "recommended_action": "ACCEPT", "playbook_rule_ids": ["H01"], "criticality": "P2"},
    }
    verdict = validate_draft(contract_id="vd2", hid="H01", draft=draft)
    assert verdict["pass"] is True
