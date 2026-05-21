# MS3 — Multi-Agent NDA Review System

A LangGraph-driven multi-agent system that analyzes ContractNLI NDAs against the 17 hypothesis playbook (mode `hypothesis`) and provides an interactive legal-assistant REPL (mode `chat`). Uses Qwen2.5-72B-Instruct via OpenRouter — no fine-tuned weights.

## Setup

```bash
pip install -r ms3/requirements.txt
cp ms3/.env.example ms3/.env
# edit ms3/.env and set OPENROUTER_API_KEY
```

Indexes are built on first run from `assets/train.json` and persisted under `ms3/DBs/`.

## CLI

```bash
# Analyse one contract against the 17 hypotheses
python -m ms3.cli --mode hypothesis --contract test:0 --retriever vector --out /tmp/r.json

# Conversational legal-assistant REPL
python -m ms3.cli --mode chat --contract test:0 --retriever graph

# Full evaluation over the test split (produces evaluation_metrics.csv + ms3_runtraces.zip)
python -m ms3.cli --mode eval --split test
```

## Agents

| Agent | Role | Tools |
|---|---|---|
| Orchestrator | Routes between modes, owns the run-level runtrace | — |
| Analyzer | Iterates the 17 hypotheses, produces label + evidence + risk | `vector_rag`, `graph_rag`, `playbook_lookup`, `get_contract_chunk` |
| Validator | Audits Analyzer evidence; can request one retry | `validate_quote`, `playbook_lookup` |
| Conversation | Stateful legal-assistant chat | `vector_rag`, `graph_rag`, `web_search`, `get_contract_chunk` |

## Runtrace

Every run emits a JSON file conforming to `assets/runtrace_ms3.schema.json`. Tool calls are recorded with `tool_name`, `agent`, `args`, `output`, `latency_ms`, and an aggregate `tool_call_summary_by_agent`.
