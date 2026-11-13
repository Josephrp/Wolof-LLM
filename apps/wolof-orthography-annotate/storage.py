"""Hub-backed storage for source rows, votes, and consensus labels."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, dataset_info
from huggingface_hub.utils import RepositoryNotFoundError

from config import Settings
from consensus import tally_labels, winning_label
from orthography import normalize_for_id

logger = logging.getLogger(__name__)

VOTE_FEATURES = Features(
    {
        "row_id": Value("string"),
        "text": Value("string"),
        "label": Value("string"),
        "volunteer_id": Value("string"),
        "source_dataset": Value("string"),
        "source_split": Value("string"),
        "timestamp": Value("string"),
    }
)

CONSENSUS_FEATURES = Features(
    {
        "row_id": Value("string"),
        "text": Value("string"),
        "label": Value("string"),
        "n_agreeing": Value("int64"),
        "volunteer_ids": Sequence(Value("string")),
        "source_dataset": Value("string"),
        "source_split": Value("string"),
        "timestamp": Value("string"),
        "votes_snapshot": Value("string"),
    }
)


def make_row_id(text: str, source_dataset: str, index: int, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    digest = hashlib.sha1(
        f"{source_dataset}\0{index}\0{normalize_for_id(text)}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{index:08d}_{digest}"


class AnnotationStore:
    """Thread-safe store that persists votes/consensus on the Hub."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._api = HfApi(token=settings.hf_token)
        self._source_rows: list[dict[str, Any]] = []
        self._votes: Dataset | None = None
        self._consensus_ids: set[str] = set()
        self._votes_by_row: dict[str, list[dict[str, Any]]] = {}
        self._volunteer_done: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ load
    def bootstrap(self) -> str:
        with self._lock:
            self._ensure_remote_datasets()
            n_src = self._load_source()
            self._load_votes()
            self._load_consensus_ids()
            return (
                f"Loaded {n_src:,} source rows from `{self.settings.source_dataset}` "
                f"({self.settings.source_split}). "
                f"Votes: {len(self._votes) if self._votes is not None else 0:,}. "
                f"Consensus sealed: {len(self._consensus_ids):,}. "
                f"Threshold: {self.settings.consensus_threshold}."
            )

    def _token_kwargs(self) -> dict[str, Any]:
        return {"token": self.settings.hf_token} if self.settings.hf_token else {}

    def _ensure_remote_datasets(self) -> None:
        if not self.settings.hf_token:
            logger.warning("No HF_TOKEN — consensus/votes writes will fail.")
            return
        for repo_id in (self.settings.votes_dataset, self.settings.consensus_dataset):
            try:
                dataset_info(repo_id, token=self.settings.hf_token)
            except RepositoryNotFoundError:
                logger.info("Creating dataset repo %s", repo_id)
                self._api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=True,
                    exist_ok=True,
                )
                feats = (
                    VOTE_FEATURES
                    if repo_id == self.settings.votes_dataset
                    else CONSENSUS_FEATURES
                )
                empty = Dataset.from_dict({name: [] for name in feats}, features=feats)
                empty.push_to_hub(
                    repo_id,
                    private=True,
                    token=self.settings.hf_token,
                    commit_message="Initialize empty annotation table",
                )

    def _load_source(self) -> int:
        split = self.settings.source_split
        max_rows = self.settings.source_max_rows
        if max_rows and max_rows > 0:
            split = f"{split}[:{max_rows}]"
        ds = load_dataset(
            self.settings.source_dataset,
            name=self.settings.source_config,
            split=split,
            **self._token_kwargs(),
        )
        col = self.settings.source_text_column
        if col not in ds.column_names:
            raise ValueError(
                f"Column '{col}' missing from {self.settings.source_dataset}. "
                f"Have: {ds.column_names}"
            )
        id_col = self.settings.source_id_column
        rows: list[dict[str, Any]] = []
        for i, ex in enumerate(ds):
            text = normalize_for_id(str(ex[col]))
            if not text:
                continue
            explicit = str(ex[id_col]) if id_col and id_col in ex else None
            rows.append(
                {
                    "row_id": make_row_id(
                        text, self.settings.source_dataset, i, explicit
                    ),
                    "text": text,
                    "source_index": i,
                }
            )
        self._source_rows = rows
        return len(rows)

    def _load_votes(self) -> None:
        try:
            votes = load_dataset(
                self.settings.votes_dataset,
                split="train",
                **self._token_kwargs(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load votes dataset: %s", exc)
            votes = Dataset.from_dict(
                {name: [] for name in VOTE_FEATURES}, features=VOTE_FEATURES
            )
        self._votes = votes
        self._votes_by_row = {}
        self._volunteer_done = set()
        for row in votes:
            rid = str(row["row_id"])
            vid = str(row["volunteer_id"]).strip().lower()
            self._votes_by_row.setdefault(rid, []).append(dict(row))
            self._volunteer_done.add((vid, rid))

    def _load_consensus_ids(self) -> None:
        try:
            cons = load_dataset(
                self.settings.consensus_dataset,
                split="train",
                **self._token_kwargs(),
            )
            self._consensus_ids = {str(r["row_id"]) for r in cons}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load consensus dataset: %s", exc)
            self._consensus_ids = set()

    # ----------------------------------------------------------------- query
    def stats(self) -> dict[str, int]:
        with self._lock:
            open_rows = sum(
                1
                for r in self._source_rows
                if r["row_id"] not in self._consensus_ids
            )
            return {
                "source_rows": len(self._source_rows),
                "votes": len(self._votes) if self._votes is not None else 0,
                "consensus": len(self._consensus_ids),
                "open_rows": open_rows,
                "threshold": self.settings.consensus_threshold,
            }

    def next_item(self, volunteer_id: str) -> dict[str, Any] | None:
        """Pick an open row this volunteer has not labeled yet."""
        vid = volunteer_id.strip().lower()
        with self._lock:
            pool = self.settings.prefetch_pool
            candidates: list[dict[str, Any]] = []
            for row in self._source_rows:
                rid = row["row_id"]
                if rid in self._consensus_ids:
                    continue
                if (vid, rid) in self._volunteer_done:
                    continue
                candidates.append(row)
                if len(candidates) >= pool:
                    break
            if not candidates:
                # Fallback: any non-consensus even if already voted (should be rare).
                for row in self._source_rows:
                    if row["row_id"] not in self._consensus_ids:
                        candidates.append(row)
                    if len(candidates) >= pool:
                        break
            if not candidates:
                return None
            # Prefer rows closest to consensus (most votes already).
            def score(r: dict[str, Any]) -> tuple[int, int]:
                votes = self._votes_by_row.get(r["row_id"], [])
                return (-len(votes), r["source_index"])

            candidates.sort(key=score)
            chosen = candidates[0]
            existing = self._votes_by_row.get(chosen["row_id"], [])
            return {
                "row_id": chosen["row_id"],
                "text": chosen["text"],
                "n_votes_so_far": len(existing),
                "labels_so_far": tally_labels(existing),
            }

    # ----------------------------------------------------------------- write
    def submit_vote(
        self, volunteer_id: str, row_id: str, text: str, label: str
    ) -> dict[str, Any]:
        if label not in self.settings.labels:
            raise ValueError(f"Invalid label: {label}")
        vid = volunteer_id.strip()
        if not vid:
            raise ValueError("volunteer_id is required")
        vid_key = vid.lower()

        with self._lock:
            if row_id in self._consensus_ids:
                return {
                    "ok": False,
                    "message": "This item is already sealed by consensus. Fetch another.",
                    "sealed": True,
                }
            if (vid_key, row_id) in self._volunteer_done:
                return {
                    "ok": False,
                    "message": "You already annotated this item. Fetch another.",
                    "sealed": False,
                }

            stamp = datetime.now(timezone.utc).isoformat()
            vote = {
                "row_id": row_id,
                "text": text,
                "label": label,
                "volunteer_id": vid,
                "source_dataset": self.settings.source_dataset,
                "source_split": self.settings.source_split,
                "timestamp": stamp,
            }
            self._append_vote(vote)
            sealed = self._maybe_seal_consensus(row_id, text)
            msg = f"Saved vote `{label}`."
            if sealed:
                msg += (
                    f" Consensus reached ({self.settings.consensus_threshold} "
                    f"agreeing) → wrote to `{self.settings.consensus_dataset}`."
                )
            return {"ok": True, "message": msg, "sealed": sealed, "label": label}

    def _append_vote(self, vote: dict[str, Any]) -> None:
        assert self._votes is not None
        new_ds = Dataset.from_list([vote], features=VOTE_FEATURES)
        self._votes = concatenate_datasets([self._votes, new_ds])
        rid = vote["row_id"]
        self._votes_by_row.setdefault(rid, []).append(vote)
        self._volunteer_done.add((vote["volunteer_id"].lower(), rid))
        self._push_with_retry(
            self._votes,
            self.settings.votes_dataset,
            f"Add vote {vote['row_id']} by {vote['volunteer_id']}",
        )

    def _maybe_seal_consensus(self, row_id: str, text: str) -> bool:
        votes = self._votes_by_row.get(row_id, [])
        won = winning_label(votes, self.settings.consensus_threshold)
        if won is None:
            return False
        best_label, agree_ids = won
        tally = tally_labels(votes)

        stamp = datetime.now(timezone.utc).isoformat()
        row = {
            "row_id": row_id,
            "text": text,
            "label": best_label,
            "n_agreeing": len(agree_ids),
            "volunteer_ids": agree_ids,
            "source_dataset": self.settings.source_dataset,
            "source_split": self.settings.source_split,
            "timestamp": stamp,
            "votes_snapshot": json.dumps(tally, ensure_ascii=False),
        }
        try:
            existing = load_dataset(
                self.settings.consensus_dataset,
                split="train",
                **self._token_kwargs(),
            )
        except Exception:  # noqa: BLE001
            existing = Dataset.from_dict(
                {name: [] for name in CONSENSUS_FEATURES},
                features=CONSENSUS_FEATURES,
            )
        # Idempotent: skip if already present remotely
        if row_id in {str(r["row_id"]) for r in existing}:
            self._consensus_ids.add(row_id)
            return True
        merged = concatenate_datasets(
            [existing, Dataset.from_list([row], features=CONSENSUS_FEATURES)]
        )
        self._push_with_retry(
            merged,
            self.settings.consensus_dataset,
            f"Seal consensus {row_id} → {best_label}",
        )
        self._consensus_ids.add(row_id)
        return True

    def _push_with_retry(self, ds: Dataset, repo_id: str, message: str) -> None:
        if not self.settings.hf_token:
            raise RuntimeError("HF_TOKEN is required to save annotations.")
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                ds.push_to_hub(
                    repo_id,
                    private=True,
                    token=self.settings.hf_token,
                    commit_message=message[:200],
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to push to {repo_id}: {last_err}")
