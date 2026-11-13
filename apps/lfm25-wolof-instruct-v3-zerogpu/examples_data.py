"""Wolof Instruct-v3 demo prompts (chat-oriented)."""

from __future__ import annotations

SYSTEM_WOLOF = (
    "Yawu ab jëmmal bu am solo. Tontu ci Wolof su laaj bi ci Wolof ee; "
    "su laaj bi ci Angale ee, mën nga tontu ci Angale walla Wolof."
)
SYSTEM_WOLOF_SHORT = "Tontu ci Wolof rekk, bu gaaw te leer."
SYSTEM_BILINGUAL = (
    "You are an LFM2.5 Wolof assistant from GalsenAI. "
    "Prefer Wolof when the user writes in Wolof; otherwise answer briefly in English then Wolof."
)
SYSTEM_TRANSLATE = (
    "Yawu jàllalekat. Su ñu laaj ci Angale, jàllalel ci Wolof; "
    "su ñu laaj ci Wolof, jàllalel ci Angale. Bu gatt te leer."
)

CHAT_PROMPTS: list[str] = [
    "Bind ak jàmm?",
    "Lan mooy Dakar?",
    "Naka nga def? Waxal ma ci Wolof lu am solo ci Senegaal.",
    "Ñan a ngi ci Senegal?",
    "Lan mooy Wolof, te nan lañu koy jëfandikoo ci Senegaal?",
    "Translate to Wolof: Good morning, how are you?",
    "Waxal ma ci Wolof: what is the capital of Senegal?",
    "Bindal ma ab xew-xew bu gatt ci Dakar.",
]
