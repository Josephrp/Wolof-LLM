"""Qwen3.5 chat-template anchoring for Wolof informal orthography SFT."""

from __future__ import annotations

from typing import Any

# Registered during vocab extension; also referenced in system prompts.
ANCHOR_SPECIALS = (
    "<|lang_wo|>",
    "<|ortho_ns|>",
)

SYSTEM_ANCHOR = (
    "You are a Wolof language assistant. "
    "Prefer informal / non-standard Wolof Latin orthography "
    "(phone and social spelling), not school/CLAD formal orthography. "
    f"{ANCHOR_SPECIALS[0]} {ANCHOR_SPECIALS[1]}"
)

THINK_TASKS = frozenset({"qa", "instruction"})


def normalize_messages(value: Any) -> list[dict[str, str]]:
    import json

    if isinstance(value, str):
        value = json.loads(value)
    elif not isinstance(value, list) and hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError("messages must be a list or JSON list")
    result: list[dict[str, str]] = []
    role_map = {"human": "user", "gpt": "assistant"}
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("each message must be a mapping")
        role = role_map.get(message.get("from"), message.get("role"))
        content = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant", "tool"} or content is None:
            raise ValueError(f"invalid message: {message!r}")
        result.append({"role": str(role), "content": str(content)})
    if not any(message["role"] == "assistant" for message in result):
        raise ValueError("conversation has no assistant turn")
    return result


def inject_anchors(messages: list[dict[str, str]], *, task: str | None = None) -> list[dict[str, str]]:
    """Prepend system anchor and optionally wrap assistant with Qwen <think> tags."""
    out = [m for m in messages if m["role"] != "system"]
    out.insert(0, {"role": "system", "content": SYSTEM_ANCHOR})
    if task and task.lower() in THINK_TASKS:
        wrapped: list[dict[str, str]] = []
        for message in out:
            if message["role"] == "assistant" and "<think>" not in message["content"]:
                content = f"<think>\n</think>\n{message['content']}"
                wrapped.append({"role": "assistant", "content": content})
            else:
                wrapped.append(message)
        return wrapped
    return out
