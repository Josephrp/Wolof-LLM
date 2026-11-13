"""Wolof demo prompts: CLM document starts, SFT questions, and system strings."""

from __future__ import annotations

# Soft system / style lines (chat models fold these into the user turn).
SYSTEM_WOLOF = (
    "Yawu ab jëmmal bu am solo. Tontu ci Wolof su laaj bi ci Wolof ee; "
    "su laaj bi ci Angale ee, mën nga tontu ci Angale walla Wolof."
)
SYSTEM_WOLOF_SHORT = "Tontu ci Wolof rekk, bu gaaw te leer."
SYSTEM_BILINGUAL = (
    "You are MORENA helping with Wolof. Prefer Wolof when the user writes in Wolof; "
    "otherwise answer briefly in English then Wolof."
)
SYSTEM_COMPLETION_HINT = "Ci Wolof:"

# Document-start prefixes from Evaluation/data/corpus_test_clm.csv (verbalized pack).
CLM_PREFIXES: list[str] = [
    "waalo waalo bokkoon na ci yénn ci nguuri senegaal yi fi nekkoo. buuru waalo dañu ko daa tànn ca",
    "séeréer (mba seereer) genn giir la gu nekk senegaal. siin ak saalum ñooy seen i nguur. ci lim ñooy",
    "bërki-démb, ci altine ji, rajo rfi dafa juum tàgge ñaar ci njiiti réewu senegaal yi woon : abdu",
    "njiiti réewi g8 yi ñoo ngi woote ngir jeexal fitna ci siri te ñu tàmbali waxtaan ngir jàmm ci lu",
    "cosaan cosaan ci angale mooy tradition. ci faranse mooy tradition. ci xëtt wi, dama cee fas yeeni",
    "ci weer wu nekk, diego dina dugal 5 000 dollaru, di li mu jot ci liggéeyam, ci ab",
    "dama wara defaral sama xarit aniwerseer bu yéeme ayu-bis",
    "ci kaw loolu aman jeli mbubb maak fas wa , solal mardose , waral ko , jiitu ko",
]

# Experiments/MORENA/data/sft/sample_wolof_sft.jsonl
SFT_PROMPTS: list[str] = [
    "Bind ak jàmm?",
    "Lan mooy Dakar?",
    "Waxal ma ci Wolof: hello",
    "Ñan a ngi ci Senegal?",
    "Translate to Wolof: Good morning",
]

# Conversational / encyclopedic (usable as completion OR chat user text).
WOLOF_NATURAL: list[str] = [
    "Naka nga def? Waxal ma ci Wolof lu am solo ci Dakar.",
    "Senegaal mooy réew mu nekk ci Afrik sowwu-joxoom.",
    "Dakar mooy péeyu Senegaal te dafa am ay nit ñu bari.",
    "Lan mooy Wolof, te nan lañu koy jëfandikoo ci Senegaal?",
]
