"""Cohere / Tiny Aya chat-template anchoring for Wolof orthography SFT.

Anchors live in the developer system preamble (not Qwen <think> wrapping).
Supports informal (`ns`) and school/CLAD (`std`) orthography variants.
"""

from __future__ import annotations

from typing import Any, Literal

OrthoKey = Literal["ns", "std"]

# Registered during vocab extension; also referenced in system prompts.
ANCHOR_SPECIALS = (
    "<|lang_wo|>",
    "<|ortho_ns|>",
    "<|ortho_std|>",
)

SYSTEM_ANCHOR_NS = (
    "You are a Wolof language assistant for GalsenAI.\n"
    "Prefer informal / non-standard Wolof Latin orthography (phone and social spelling),\n"
    "not school/CLAD formal orthography.\n"
    f"{ANCHOR_SPECIALS[0]} {ANCHOR_SPECIALS[1]}"
)

SYSTEM_ANCHOR_STD = (
    "You are a Wolof language assistant for GalsenAI.\n"
    "Prefer school / CLAD formal Wolof Latin orthography (standard spelling),\n"
    "not informal phone/social orthography.\n"
    f"{ANCHOR_SPECIALS[0]} {ANCHOR_SPECIALS[2]}"
)

SYSTEM_ANCHORS: dict[str, str] = {
    "ns": SYSTEM_ANCHOR_NS,
    "std": SYSTEM_ANCHOR_STD,
}

# Back-compat alias used by older call sites / docs.
SYSTEM_ANCHOR = SYSTEM_ANCHOR_NS

# Cohere chat markers used for assistant-only label masking.
START_RESPONSE = "<|START_RESPONSE|>"
END_RESPONSE = "<|END_RESPONSE|>"


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


def inject_anchors(
    messages: list[dict[str, str]],
    *,
    task: str | None = None,
    ortho: OrthoKey | str = "ns",
) -> list[dict[str, str]]:
    """Prepend Cohere developer system anchor for the chosen orthography."""
    del task  # reserved for callers; Tiny Aya v1 does not wrap assistant content
    key = str(ortho or "ns").lower()
    if key not in SYSTEM_ANCHORS:
        raise ValueError(f"unknown ortho anchor {ortho!r}; use ns or std")
    out = [m for m in messages if m["role"] != "system"]
    out.insert(0, {"role": "system", "content": SYSTEM_ANCHORS[key]})
    return out
