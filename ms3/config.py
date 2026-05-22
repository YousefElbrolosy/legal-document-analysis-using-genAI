"""Central config for the MS3 multi-agent system.

Loads `.env` from `ms3/.env` if present. All paths are absolute so the CLI
works regardless of the current working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
MS3_ROOT = Path(__file__).resolve().parent

load_dotenv(MS3_ROOT / ".env", override=True)

ASSETS_DIR = REPO_ROOT / "assets"
PLAYBOOK_PATH = ASSETS_DIR / "playbook.yaml"
TEST_SPLIT_PATH = ASSETS_DIR / "test.json"
TRAIN_SPLIT_PATH = ASSETS_DIR / "train.json"
DEV_SPLIT_PATH = ASSETS_DIR / "dev.json"
RUNTRACE_SCHEMA_PATH = ASSETS_DIR / "runtrace_ms3.schema.json"

DBS_DIR = MS3_ROOT / "DBs"
VECTOR_DB_PATH = DBS_DIR / "vector_db"
GRAPH_DB_PATH = DBS_DIR / "graph_db" / "contractnli_graph.pkl"
RUNS_DIR = MS3_ROOT / "runs"

# Provider switch — one of: openai | nvidia | ollama | openrouter.
# Controls which client `chat()` uses in models/llm.py.
PROVIDER = os.environ.get("MS3_PROVIDER", "openai").lower()

# OpenAI (default). Standard api.openai.com endpoint.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# NVIDIA Build (free, OpenAI-compatible).
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# OpenRouter (OpenAI-compatible).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Ollama Cloud (uses the ollama Python SDK, not OpenAI shape).
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")

MODEL = os.environ.get("MS3_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.environ.get("MS3_TEMPERATURE", "0.1"))
TOP_P = float(os.environ.get("MS3_TOP_P", "0.9"))
MAX_TOKENS = int(os.environ.get("MS3_MAX_TOKENS", "1024"))
SEED = int(os.environ.get("MS3_SEED", "42"))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "m2_training_rag"
DEFAULT_VECTOR_TOP_K = 5
DEFAULT_GRAPH_TOP_K_RULES = 2
DEFAULT_GRAPH_CLAUSES_PER_RULE = 5

VALIDATOR_MAX_RETRIES = 1
