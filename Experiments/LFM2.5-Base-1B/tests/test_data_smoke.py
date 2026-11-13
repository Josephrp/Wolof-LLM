"""Data helpers smoke tests — mocked tiny rows, no Hub required."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from common.data import (
    charger_sft_messages,
    charger_splits_local,
    paqueter,
)


def test_charger_splits_local(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"text": ["salaam", "nanga def"]}).to_parquet(
        data / "wolof_bpe_train_v1.parquet"
    )
    pd.DataFrame({"text": ["jamm"]}).to_parquet(
        data / "wolof_eval_clean_v1.parquet"
    )
    train, ev = charger_splits_local(tmp_path, data_root=data)
    assert train == ["salaam", "nanga def"]
    assert ev == ["jamm"]


def test_charger_sft_messages_jsonl(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "Nanga def?"},
            {
                "role": "assistant",
                "content": "Maangi fi",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": '{"q": "fi"}',
                        }
                    }
                ],
            },
        ],
        "task": "instruction",
        "source": "unit",
        "prompt_lang": "wo",
        "meta": '{"seed": 1}',
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = charger_sft_messages(path)
    assert len(rows) == 1
    assert rows[0]["meta"] == {"seed": 1}
    assert rows[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
        "q": "fi"
    }


def test_charger_sft_messages_parquet_directory(tmp_path: Path):
    pd.DataFrame(
        [
            {
                "messages": [{"role": "assistant", "content": "Jamm"}],
                "task": "instruction",
                "meta": '{"source": "mixture"}',
            }
        ]
    ).to_parquet(tmp_path / "train.parquet")
    rows = charger_sft_messages(tmp_path, split="train")
    assert len(rows) == 1
    assert rows[0]["meta"] == {"source": "mixture"}


def test_paqueter_eos_override():
    class FakeTok:
        eos_token_id = 7

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, ids):
            return "<eos>"

    # two texts -> [1,2,3,7,1,2,3,7] -> 2 blocks of longueur=4
    ds = paqueter(["a", "b"], FakeTok(), longueur=4, eos_token_id=7)
    assert len(ds) == 2
    assert ds[0]["input_ids"] == [1, 2, 3, 7]
    assert ds[1]["input_ids"] == [1, 2, 3, 7]


def test_paqueter_asserts_missing_eos():
    class FakeTok:
        eos_token_id = None

        def encode(self, text, add_special_tokens=False):
            return [1]

        def decode(self, ids):
            return "?"

    with pytest.raises(AssertionError):
        paqueter(["a"], FakeTok(), longueur=2)
