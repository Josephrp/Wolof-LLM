#!/usr/bin/env python3
"""Validate localized Wolof tool-use rows while preserving JSON schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.synthetic import read_jsonl, write_jsonl


def validate_tool_row(row: dict) -> tuple[bool, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False, "messages_not_list"
    has_user = False
    has_call = False
    for message in messages:
        if not isinstance(message, dict):
            return False, "message_not_object"
        has_user |= message.get("role") == "user" and bool(message.get("content"))
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return False, "invalid_arguments_json"
            if not isinstance(arguments, dict):
                return False, "arguments_not_object"
            has_call = True
    return has_user and has_call, "ok" if has_user and has_call else "missing_user_or_call"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate localized Wolof tool-use JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    accepted = []
    for row in read_jsonl(args.input):
        valid, reason = validate_tool_row(row)
        if valid:
            accepted.append({**row, "schema_valid": True, "localization": "wo"})
        else:
            print(f"reject: {reason}", file=sys.stderr)
    write_jsonl(args.output, accepted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
