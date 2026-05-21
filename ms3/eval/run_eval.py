"""Run the multi-agent pipeline across a ContractNLI split.

Writes one runtrace per contract under `out_dir`, then aggregates MS1-parity
metrics and merges them into `evaluation_metrics.csv`. Finally zips the
runtraces into `ms3_runtraces.zip` (deliverable 5c).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tqdm import tqdm

from ms3.agents.orchestrator import run_hypothesis_analysis
from ms3.config import REPO_ROOT, RUNS_DIR
from ms3.contract_loader import list_contracts, load_contract
from ms3.eval.aggregate_csv import write_combined
from ms3.eval.metrics import aggregate
from ms3.runtrace.builder import write_runtrace

ZIP_PATH = REPO_ROOT / "ms3_runtraces.zip"


def run_eval(
    split: str = "test",
    limit: int | None = None,
    out_dir: Path | None = None,
    retriever: str = "vector",
) -> tuple[Path, Path]:
    out_dir = out_dir or RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list_contracts(split=split, limit=limit)
    runtraces: list[dict] = []
    runtrace_paths: list[Path] = []
    for i in tqdm(indices, desc=f"eval/{split}"):
        contract, gold = load_contract(f"{split}:{i}")
        try:
            rt = run_hypothesis_analysis(contract, gold, retriever_mode=retriever)
            path = out_dir / f"{contract['contract_id']}.runtrace.json"
            write_runtrace(rt, path)
            runtrace_paths.append(path)
            with open(path, "r", encoding="utf-8") as f:
                runtraces.append(json.load(f))
        except Exception as e:
            tqdm.write(f"[eval] contract {contract['contract_id']} failed: {e}")
            continue

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
    args = ap.parse_args()
    csv_p, zip_p = run_eval(split=args.split, limit=args.limit, retriever=args.retriever)
    print(f"csv -> {csv_p}")
    print(f"zip -> {zip_p}")
