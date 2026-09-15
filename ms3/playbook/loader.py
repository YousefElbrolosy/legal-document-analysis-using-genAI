"""Read the unedited MS1 playbook.yaml and expose its content + a sha256 hash.

The dataset uses keys `nda-1`..`nda-20` (skipping 6, 9, 14 → 17 valid keys).
The playbook uses sequential `H01`..`H17`. Mapping is positional (hypothesis
text matches verbatim between the two).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import yaml

from ms3.config import PLAYBOOK_PATH

# Positional mapping (verified against assets/test.json labels — see plan):
NDA_TO_HID: dict[str, str] = {
    "nda-1": "H01",
    "nda-2": "H02",
    "nda-3": "H03",
    "nda-4": "H04",
    "nda-5": "H05",
    "nda-7": "H06",
    "nda-8": "H07",
    "nda-10": "H08",
    "nda-11": "H09",
    "nda-12": "H10",
    "nda-13": "H11",
    "nda-15": "H12",
    "nda-16": "H13",
    "nda-17": "H14",
    "nda-18": "H15",
    "nda-19": "H16",
    "nda-20": "H17",
}
HID_TO_NDA: dict[str, str] = {v: k for k, v in NDA_TO_HID.items()}

DATASET_CHOICE_TO_LABEL: dict[str, str] = {
    "Entailment": "ENTAILED",
    "Contradiction": "CONTRADICTED",
    "NotMentioned": "NOT_MENTIONED",
}


@lru_cache(maxsize=1)
def load_playbook() -> dict[str, Any]:
    with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def ruleset_hash() -> str:
    pb = load_playbook()
    canonical = json.dumps(pb, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=1)
def get_checks() -> dict[str, dict[str, Any]]:
    return {c["hypothesis_id"]: c for c in load_playbook()["checks"]}


def hypothesis_text(hid: str) -> str:
    return get_checks()[hid]["hypothesis_text"]


def hypothesis_title(hid: str) -> str:
    return get_checks()[hid]["title"]


def criticality(hid: str) -> str:
    return get_checks()[hid]["criticality"]
