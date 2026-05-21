"""Merge the MS1 and MS3 rows into the final evaluation CSV.

MS1 CSV columns (existing):
  label_accuracy, groundedness, quote_integrity_pass_rate, avg_latency_ms, evaluation_timestamp

MS3 final CSV adds a `system` column so the two systems can be compared in
the same file (deliverable 5b).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ms3.config import REPO_ROOT

MS1_CSV = REPO_ROOT / "evaluation_metrics.csv"
FINAL_CSV = REPO_ROOT / "evaluation_metrics.csv"

COLUMNS = [
    "system",
    "label_accuracy",
    "groundedness",
    "quote_integrity_pass_rate",
    "avg_latency_ms",
    "evaluation_timestamp",
]


def _read_existing() -> list[dict[str, str]]:
    if not MS1_CSV.exists():
        return []
    with open(MS1_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    # Backfill a `system` column for legacy rows that lack it.
    for r in rows:
        if "system" not in r or not r["system"]:
            r["system"] = "fine_tuned_ms1"
    return rows


def write_combined(ms3_metrics: dict[str, Any]) -> Path:
    rows = _read_existing()
    # drop any prior multi_agent_ms3 row(s) so re-runs aren't duplicated
    rows = [r for r in rows if r.get("system") != "multi_agent_ms3"]
    rows.append(
        {
            "system": "multi_agent_ms3",
            "label_accuracy": f"{ms3_metrics['label_accuracy']:.6f}",
            "groundedness": f"{ms3_metrics['groundedness']:.6f}",
            "quote_integrity_pass_rate": f"{ms3_metrics['quote_integrity_pass_rate']:.6f}",
            "avg_latency_ms": f"{ms3_metrics['avg_latency_ms']:.2f}",
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return FINAL_CSV
