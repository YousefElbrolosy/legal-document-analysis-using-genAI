# MS3 — Multi-Agent NDA Review System

A LangGraph-driven multi-agent system that analyzes ContractNLI NDAs against the 17-hypothesis playbook (`--mode hypothesis`) and provides an interactive legal-assistant REPL (`--mode chat`). The active provider is **Ollama Cloud** (Gemma 4 31B by default); OpenRouter / Qwen 2.5 is kept as a fallback in code. No fine-tuned weights are used.

## Setup

```bash
pip install -r ms3/requirements.txt
cp ms3/.env.example ms3/.env
# Edit ms3/.env and set OLLAMA_API_KEY (or OPENROUTER_API_KEY for the fallback).
```

On first run, the Chroma vector index and NetworkX graph are built from `assets/train.json` and persisted under `ms3/DBs/`. Subsequent runs reuse them.

## CLI overview

`python -m ms3.cli --mode {hypothesis | chat | eval} [flags]`

| Flag | Modes | Description |
|---|---|---|
| `--mode {hypothesis,chat,eval}` | all | **Required.** Which pipeline to run. |
| `--contract <spec>` | hypothesis, chat | `test:N`, `dev:N`, or a path to a `.txt` / `.json` file. |
| `--retriever {vector,graph}` | all | Which RAG branch the run uses. Defaults to `vector`. |
| `--out <path>` | hypothesis | Override the runtrace output path. |
| `--split {test,dev}` | eval | Which ContractNLI split to evaluate. Defaults to `test`. |
| `--limit N` | eval | Cap the eval to the first N contracts. |
| `--indices "<spec>"` | eval | Run only specific **positional** indices, e.g. `"0,5,10-20"`. |
| `--contract-ids "<spec>"` | eval | Run only specific **contract IDs**, e.g. `"22,25,36,539-"`. |
| `--out-dir <dir>` | eval | Where per-contract runtraces are written. Defaults to `ms3/runs/`. |

`--indices` and `--contract-ids` are mutually exclusive.

## Common recipes

### Analyse one contract against the 17 hypotheses
```bash
python -m ms3.cli --mode hypothesis --contract test:0 --retriever vector --out /tmp/r.json
python -m ms3.cli --mode hypothesis --contract test:0 --retriever graph
python -m ms3.cli --mode hypothesis --contract /path/to/contract.txt --retriever vector
```

### Conversational legal-assistant REPL
```bash
python -m ms3.cli --mode chat --contract test:0 --retriever vector
```
Inside the REPL:
- `:exit` — write the session runtrace and quit.
- `:save` — snapshot the runtrace without ending the session.
- `:tools` — print per-agent tool-call counts so far.
- `Ctrl-D` — quit (same as `:exit`).

### Full evaluation over a split
```bash
# Whole test split (produces evaluation_metrics.csv + ms3_runtraces.zip)
python -m ms3.cli --mode eval --split test --retriever vector

# Smoke test on the first 5 contracts
python -m ms3.cli --mode eval --split test --retriever vector --limit 5

# Eval on the dev split with graph RAG
python -m ms3.cli --mode eval --split dev --retriever graph
```

### Resume a partial evaluation

If a run stops part-way through, you can re-run only the missing contracts. The CSV/zip aggregation always reads **every** `*.runtrace.json` in the out-dir, so previously completed contracts are kept.

```bash
# By contract ID (what's printed in the runtrace filename)
python -m ms3.cli --mode eval --split test --contract-ids "22,25,36,354,539-"

# By positional index (0-based position in the split)
python -m ms3.cli --mode eval --split test --indices "9,12,16,73,104-"
```

Range syntax for both:
- `"22,25,36"` — exact IDs / positions
- `"100-200"` — closed range
- `"100-"` — open-ended (up to the last contract in the split)

### Find which contracts are still missing
```bash
python3 -c "
import json, os
with open('assets/test.json') as f:
    ids = {int(d['id']) for d in json.load(f)['documents']}
have = {int(f.replace('.runtrace.json','')) for f in os.listdir('ms3/runs')
        if f.endswith('.runtrace.json') and not f.endswith('.chat.runtrace.json')}
print('missing:', sorted(ids - have))
"
```

## Inspecting a runtrace

```bash
# Summary metrics
jq '.metrics' ms3/runs/1.runtrace.json

# Per-hypothesis prediction vs gold
jq '.hypothesis_traces[] | {hid: .hypothesis_id, pred: .decision.label, gold: .gold_label, qi: .quote_integrity_pass}' ms3/runs/1.runtrace.json

# Tool-call counts per agent (spec §2h)
jq '.tool_call_summary_by_agent' ms3/runs/1.runtrace.json
```

## Direct entry points

The CLI is the recommended interface, but individual modules can be invoked too:

```bash
# Force-rebuild the vector / graph indexes
python -m ms3.rag.vector_index --force
python -m ms3.rag.graph_index --force

# Run eval directly (same flags as the CLI eval mode)
python -m ms3.eval.run_eval --split test --contract-ids "22,25,36"

# Smoke tests (no network, no LLM)
python -m pytest ms3/tests/test_smoke.py -q
```

## Agents

| Agent | Role | Memory | Tools |
|---|---|---|---|
| **Orchestrator** | LangGraph state machine; routes between modes, owns the run-level runtrace, performs deterministic post-checks. | Run-level state (current contract, hypotheses queue, prior decisions). | `validate_quote`, `playbook_lookup` (called as pure code from `commit`). |
| **Analyzer** | Per-hypothesis NLI: label + evidence + justification. | Prior decisions on the same contract (last 8). | `vector_rag`, `graph_rag`, `get_contract_chunk`. |
| **Conversation** | Interactive legal assistant on one uploaded contract. | Full chat history for the session. | `vector_rag`, `graph_rag`, `get_contract_chunk`, `web_search`. |

`validate_draft` (in `agents/validator.py`) is **pure-code** post-checking, not an LLM agent. It runs inside the orchestrator's `commit` node to produce `quote_integrity_pass` and the `validations[]` array.

## Runtrace

Every run emits a JSON file conforming to `assets/runtrace_ms3.schema.json`.

| Mode | Filename | `hypothesis_traces` |
|---|---|---|
| `hypothesis` | `<contract_id>.runtrace.json` | exactly 17 |
| `chat` | `<contract_id>.chat.runtrace.json` | empty (allowed by schema) |

Tool calls are recorded with `tool_name`, `agent`, `args`, `output`, `latency_ms`, plus the aggregate `tool_call_summary_by_agent` (spec §2h: tool calls per agent).
