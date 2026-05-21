"""Validate a runtrace dict against assets/runtrace_ms3.schema.json."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from ms3.config import RUNTRACE_SCHEMA_PATH


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    with open(RUNTRACE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return Draft202012Validator(schema)


def validate_runtrace(obj: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        msgs = "\n".join(
            f"  - {'/'.join(str(p) for p in e.absolute_path)}: {e.message}" for e in errors[:10]
        )
        raise ValueError(f"runtrace failed schema validation:\n{msgs}")
