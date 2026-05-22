"""Run the multi-agent pipeline across a ContractNLI split.

Writes one runtrace per contract under `out_dir`, then aggregates MS1-parity
metrics and merges them into `evaluation_metrics.csv`. Finally zips the
runtraces into `ms3_runtraces.zip` (deliverable 5c).

Parallelism: contracts are processed concurrently with a ThreadPoolExecutor
(LLM calls are I/O-bound, so threads are sufficient — no GIL contention on
the network wait). Each worker runs inside its own `contextvars.copy_context()`
so the per-run state (`_current_runtrace`, `_current_agent`,
`_current_hypothesis`) is isolated. Retrievers are warmed once on the main
thread before the pool spins up so the lazy `lru_cache` singletons don't race.
"""
from __future__ import annotations

import contextvars
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ms3.agents.orchestrator import run_hypothesis_analysis
from ms3.config import REPO_ROOT, RUNS_DIR
from ms3.contract_loader import list_contracts, load_contract
from ms3.eval.aggregate_csv import write_combined
from ms3.eval.metrics import aggregate
from ms3.runtrace.builder import write_runtrace

ZIP_PATH = REPO_ROOT / "ms3_runtraces.zip"

DEFAULT_WORKERS = 8


def _warm_retriever(retriever: str) -> None:
    if retriever == "vector":
        from ms3.rag.vector_retriever import get_vector_retriever
        get_vector_retriever()
    elif retriever == "graph":
        from ms3.rag.graph_retriever import get_graph_retriever
        get_graph_retriever()


def _process_one(
    split: str, idx: int, retriever: str, out_dir: Path
) -> tuple[Path, dict] | None:
    contract, gold = load_contract(f"{split}:{idx}")
    rt = run_hypothesis_analysis(contract, gold, retriever_mode=retriever)
    path = out_dir / f"{contract['contract_id']}.runtrace.json"
    write_runtrace(rt, path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return path, data


def run_eval(
    split: str = "test",
    limit: int | None = None,
    out_dir: Path | None = None,
    retriever: str = "vector",
    workers: int = DEFAULT_WORKERS,
) -> tuple[Path, Path]:
    out_dir = out_dir or RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list_contracts(split=split, limit=limit)

    _warm_retriever(retriever)

    runtraces: list[dict] = []
    runtrace_paths: list[Path] = []

    if workers <= 1:
        for i in tqdm(indices, desc=f"eval/{split}"):
            try:
                result = _process_one(split, i, retriever, out_dir)
            except Exception as e:
                tqdm.write(f"[eval] contract idx={i} failed: {e}")
                continue
            if result is None:
                continue
            path, data = result
            runtrace_paths.append(path)
            runtraces.append(data)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for i in indices:
                ctx = contextvars.copy_context()
                fut = ex.submit(ctx.run, _process_one, split, i, retriever, out_dir)
                futures[fut] = i
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"eval/{split} (workers={workers})",
            ):
                idx = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    tqdm.write(f"[eval] contract idx={idx} failed: {e}")
                    continue
                if result is None:
                    continue
                path, data = result
                runtrace_paths.append(path)
                runtraces.append(data)

    metrics = aggregate(runtraces)
    csv_path = write_combined(metrics)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in runtrace_paths:
            zf.write(p, arcname=p.name)
    return csv_path, ZIP_PATH


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retriever", default="vector", choices=["vector", "graph"])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()
    csv_p, zip_p = run_eval(
        split=args.split,
        limit=args.limit,
        retriever=args.retriever,
        workers=args.workers,
    )
    print(f"csv -> {csv_p}")
    print(f"zip -> {zip_p}")
