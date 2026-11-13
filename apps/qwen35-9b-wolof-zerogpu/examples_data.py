"""Wolof prompts for the Qwen3.5-9B ZeroGPU demo.

Anchors (`<|lang_wo|>` `<|ortho_ns|>`) are injected by `app.py` — keep UI copy free of them.
"""

from __future__ import annotations

# Matches Experiments/Qwen3.5-Wolof-Family/common/anchoring.py (without exposing anchors in UI).
SYSTEM_DEFAULT = (
    "Yawu jëmmal Wolof. Tontu ci Wolof bu gaaw te leer, "
    "ak orthographie informal / téléphone (pas CLAD formel)."
)
SYSTEM_TRANSLATE = (
    "Yawu jàllalekat. Su ñu laaj ci Angale, jàllalel ci Wolof; "
    "su ñu laaj ci Wolof, jàllalel ci Angale. Bu gatt te leer."
)
SYSTEM_STORY = (
    "Yawu boroom xew-xew. Bindal xew-xew yu gatt yu am solo, ci Wolof bu leer."
)
SYSTEM_TEACHER = (
    "Yawu jàngalekat. Fàttalil lu laaj bi laaj, te joxal misaal yu yomb ci Wolof."
)

# CPT continuation seeds — anchors prepended automatically in app.py.
CPT_PREFIXES: list[str] = [
    "Dakar mooy",
    "Senegaal ab réew bu",
    "Wolof mooy làkk wu",
    "Jàmm ngaam,",
    "Ci ëllëg,",
    "Ndaje gi,",
]

CHAT_PROMPTS: list[str] = [
    "Naka nga def? Waxal ma lu am solo ci sa bés.",
    "Lan mooy Dakar, te lan mooy yu ko defar?",
    "Ñan a ngi ci Senegaal, te lan lañu di def?",
    "Bindal ma ab bataaxal bu gatt ngir ñaan ab xarit ci jàmm.",
    "Nan lañu koy togg ceebu jën? Waxal ma ay etap yu yomb.",
    "Translate to Wolof: Good morning, how are you today?",
    "Jàllalel ci Angale: Dakar mooy péeyu Senegaal.",
    "Bindal ma ab xew-xew bu gatt ci Marché Sandaga.",
    "Lan mooy teranga, te nan lañuy jëfandikoo ci Senegaal?",
    "Waxal ma ci Wolof: what should a visitor do on their first day in Dakar?",
]
