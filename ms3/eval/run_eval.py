"""Run the multi-agent pipeline across a ContractNLI split.

Writes one runtrace per contract under `out_dir`, then aggregates MS1-parity
metrics and merges them into `evaluation_metrics.csv`. Finally zips the
runtraces into `ms3_runtraces.zip` (deliverable 5c).

Supports resuming: `--indices "22,25,36,46-"` runs only the specified
contracts, and the final CSV is computed from ALL runtrace files present
in `out_dir` (both new and previously-saved ones).
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


def parse_indices(spec: str, total: int) -> list[int]:
    """Parse a comma-separated POSITIONAL-INDEX spec into a sorted unique list.

    Supports:
        "22,25,36"            -> [22, 25, 36]
        "46-50"               -> [46, 47, 48, 49, 50]
        "22,25,36,46-"        -> [22, 25, 36, 46, ..., total-1]
        "22,25,36,46-100"     -> [22, 25, 36, 46, ..., 100]
    """
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = (int(b) if b else total - 1)
            for k in range(start, end + 1):
                if 0 <= k < total:
                    out.add(k)
        else:
            k = int(part)
            if 0 <= k < total:
                out.add(k)
    return sorted(out)


def parse_contract_ids(spec: str, all_ids: list[str]) -> list[int]:
    """Parse a contract-ID spec into POSITIONAL indices.

    `all_ids` is the ordered list of contract IDs (strings) in the split.
    The spec items are matched against these IDs (numerically for ranges).

    Supports:
        "22,25,36"         -> indices where id is exactly 22, 25, 36
        "539-"             -> indices where id >= 539 (open-ended)
        "100-500"          -> indices where 100 <= id <= 500
        "22,25,36,539-"    -> combined
    """
    int_ids = [int(i) for i in all_ids]
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b) if b else max(int_ids)
            for pos, cid in enumerate(int_ids):
                if lo <= cid <= hi:
                    out.add(pos)
        else:
            target = int(part)
            for pos, cid in enumerate(int_ids):
                if cid == target:
                    out.add(pos)
    return sorted(out)


def run_eval(
    split: str = "test",
    limit: int | None = None,
    out_dir: Path | None = None,
    retriever: str = "vector",
    indices: list[int] | None = None,
) -> tuple[Path, Path]:
    out_dir = out_dir or RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = list_contracts(split=split, limit=limit)

    # Produce / refresh runtraces for the requested indices.
    new_paths: list[Path] = []
    for i in tqdm(indices, desc=f"eval/{split}"):
        contract, gold = load_contract(f"{split}:{i}")
        try:
            rt = run_hypothesis_analysis(contract, gold, retriever_mode=retriever)
            path = out_dir / f"{contract['contract_id']}.runtrace.json"
            write_runtrace(rt, path)
            new_paths.append(path)
        except Exception as e:
            tqdm.write(f"[eval] contract {contract['contract_id']} failed: {e}")
            continue

    # Aggregate metrics from EVERY runtrace currently in out_dir so previously
    # completed contracts are included alongside the freshly produced ones.
    all_paths = sorted(out_dir.glob("*.runtrace.json"))
    runtraces: list[dict] = []
    for p in all_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                runtraces.append(json.load(f))
        except Exception as e:
            tqdm.write(f"[eval] could not load {p.name}: {e}")
    tqdm.write(f"[eval] aggregating {len(runtraces)} runtraces from {out_dir}")

    metrics = aggregate(runtraces)
    csv_path = write_combined(metrics)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in all_paths:
            zf.write(p, arcname=p.name)
    return csv_path, ZIP_PATH


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retriever", default="vector", choices=["vector", "graph"])
    ap.add_argument(
        "--indices",
        default=None,
        help='Comma-separated indices / ranges, e.g. "22,25,36,46-" (open-ended) '
             'or "22,25,36,46-100". Overrides --limit.',
    )
    args = ap.parse_args()

    indices_list = None
    if args.indices:
        # Resolve "-" (open-ended) against the split size.
        from ms3.contract_loader import list_contracts as _lc
        total = len(_lc(split=args.split, limit=None))
        indices_list = parse_indices(args.indices, total)
        print(f"[eval] running indices: {indices_list}")

    csv_p, zip_p = run_eval(
        split=args.split,
        limit=args.limit,
        retriever=args.retriever,
        indices=indices_list,
    )
    print(f"csv -> {csv_p}")
    print(f"zip -> {zip_p}")
