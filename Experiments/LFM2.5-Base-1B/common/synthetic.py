"""Shared validation and budget helpers for synthetic SFT artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_budget(real_tokens: int, synthetic_tokens: int, maximum: float = 0.30) -> dict:
    if real_tokens < 0 or synthetic_tokens < 0:
        raise ValueError("token counts must be non-negative")
    total = real_tokens + synthetic_tokens
    fraction = synthetic_tokens / total if total else 0.0
    if fraction > maximum:
        raise ValueError(
            f"synthetic fraction {fraction:.2%} exceeds budget {maximum:.2%}"
        )
    return {
        "real_tokens": real_tokens,
        "synthetic_tokens": synthetic_tokens,
        "synthetic_fraction": fraction,
        "maximum": maximum,
        "passed": True,
    }


def whitespace_tokens(rows: list[dict], fields: tuple[str, ...]) -> int:
    return sum(
        len(str(row.get(field, "")).split())
        for row in rows
        for field in fields
    )
