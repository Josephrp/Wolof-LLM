"""Environment-driven configuration for the Wolof orthography annotation Space."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """All knobs overridable via Space secrets / variables."""

    # Auth: write token required to append votes + consensus rows.
    hf_token: str | None

    # Source corpus to annotate (read).
    source_dataset: str
    source_config: str | None
    source_split: str
    source_text_column: str
    source_id_column: str | None

    # Working store for individual votes (read/write).
    votes_dataset: str

    # Consensus / gold labels (read/write).
    consensus_dataset: str

    # How many agreeing volunteers seal a label.
    consensus_threshold: int

    # Sampling / UX
    max_text_chars: int
    prefetch_pool: int
    # Cap source rows loaded into memory (0 = all). Useful on small Spaces.
    source_max_rows: int
    allow_private: bool
    app_title: str
    require_volunteer_id: bool

    @property
    def labels(self) -> tuple[str, ...]:
        return ("standard", "non_standard", "mixed", "not_wolof")


def load_settings() -> Settings:
    threshold = _env_int("CONSENSUS_THRESHOLD", 2)
    if threshold < 2:
        threshold = 2
    if threshold > 10:
        threshold = 10

    return Settings(
        hf_token=_env("HF_TOKEN") or _env("HUGGING_FACE_HUB_TOKEN"),
        source_dataset=_env("SOURCE_DATASET", "galsenai/cleaned_data_for_clm")
        or "galsenai/cleaned_data_for_clm",
        source_config=_env("SOURCE_CONFIG"),
        source_split=_env("SOURCE_SPLIT", "train") or "train",
        source_text_column=_env("SOURCE_TEXT_COLUMN", "text") or "text",
        source_id_column=_env("SOURCE_ID_COLUMN"),
        votes_dataset=_env("VOTES_DATASET", "galsenai/wolof_orthography_votes")
        or "galsenai/wolof_orthography_votes",
        consensus_dataset=_env(
            "CONSENSUS_DATASET", "galsenai/wolof_orthography_consensus"
        )
        or "galsenai/wolof_orthography_consensus",
        consensus_threshold=threshold,
        max_text_chars=_env_int("MAX_TEXT_CHARS", 1200),
        prefetch_pool=_env_int("PREFETCH_POOL", 64),
        source_max_rows=_env_int("SOURCE_MAX_ROWS", 50000),
        allow_private=_env_bool("ALLOW_PRIVATE_DATASETS", True),
        app_title=_env("APP_TITLE", "Wolof Orthography Annotator")
        or "Wolof Orthography Annotator",
        require_volunteer_id=_env_bool("REQUIRE_VOLUNTEER_ID", True),
    )
