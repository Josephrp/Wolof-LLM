"""Offline tests for data, tokenizer, config, metric, and synthetic gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from common.config import (
    ConfigError,
    sha256_file,
    validate_clm_config,
    validate_sft_config,
)
from common.metrics import compute_bpb
from common.model import (
    ecrire_tie_flag,
    lire_tie_flag,
    lora_requires_weight_tying,
    verifier_vocab_compatible,
)
from common.synthetic import token_budget

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def load_script(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def fixture_frame() -> pd.DataFrame:
    rows = []
    for task in ("translation", "qa", "instruction"):
        for index in range(20):
            rows.append({"id": f"{task}-{index}", "task": task, "source": f"s{index % 2}"})
    return pd.DataFrame(rows)


def test_mixture_caps_seed_disjoint_and_metadata_shape():
    module = load_script("scripts/data/21_build_sft_mixture.py")
    frame = fixture_frame()
    one = module._sample_by_task(frame, {"translation": 5, "qa": 7, "instruction": 9}, 42, None)
    two = module._sample_by_task(frame, {"translation": 5, "qa": 7, "instruction": 9}, 42, None)
    assert one["id"].tolist() == two["id"].tolist()
    assert module._task_counts(one) == {"translation": 5, "qa": 7, "instruction": 9}
    train, val = module._carve_val(one, 0.2, 42)
    assert set(train["id"]).isdisjoint(val["id"])
    assert set(train["id"]) | set(val["id"]) == set(one["id"])
    assert set(val["task"]) == set(one["task"])
    assert module._translation_pct(module._task_counts(one)) < 0.35
    calls = []

    def loader(dataset_id, split, token):
        calls.append((dataset_id, split, token))
        return "train-dataset"

    assert module._load_train_only("private/data", "token", loader) == "train-dataset"
    assert calls == [("private/data", "train", "token")]
    metadata = module._build_metadata(
        cfg_path=ROOT / "configs/sft/mixture_v1.yaml",
        dataset_id="private/data",
        seed=42,
        val_pct=0.2,
        max_per_task=None,
        task_caps={},
        n_hub_train=len(frame),
        train_df=train,
        val_df=val,
        selected=one,
        translation_max_pct=0.35,
        train_path=ROOT / "data/train.parquet",
        val_path=ROOT / "data/val.parquet",
        notes=None,
    )
    assert metadata["schema_version"] == 1
    assert metadata["source_split"] == "train"
    assert metadata["hub_test_touched"] is False
    assert metadata["n_train"] + metadata["n_val"] == metadata["n_selected"]


def test_sft_preparation_accepts_parquet_array_messages_and_rejects_empty_output():
    module = load_script("scripts/sft/10_prepare_sft.py")

    class ArrayMessages:
        @staticmethod
        def tolist():
            return [
                {"role": "user", "content": "Naka nga def?"},
                {"role": "assistant", "content": "Mangi fi rekk."},
            ]

    messages = module.normalize_messages(ArrayMessages())
    assert [message["role"] for message in messages] == ["user", "assistant"]

    class MaskingTokenizer:
        @staticmethod
        def apply_chat_template(_messages, **kwargs):
            assert kwargs["return_assistant_tokens_mask"] is True
            return {"input_ids": [10, 20, 30], "assistant_masks": [0, 1, 1]}

    assert module.assistant_only_encoding(messages, MaskingTokenizer()) == (
        [10, 20, 30],
        [-100, 20, 30],
    )
    with pytest.raises(ValueError, match="all 2 rows"):
        module.require_nonempty_prepared("train", 2, 0)


def test_audit_stat_json_and_size_helpers():
    module = load_script("scripts/data/20_audit_datasets.py")
    assert module._pctiles([1, 2, 3, 100]) == {"p50": 3, "p90": 100, "p99": 100}
    assert module._safe_json_loads('{"x": 1}') == ({"x": 1}, True)
    assert module._safe_json_loads("{")[1] is False
    assert module._assert_size("x", 10, 10)["ok"]
    assert not module._assert_size("x", 9, 10)["ok"]


class LayoutTokenizer:
    def __init__(self, pad, bos, eos, size):
        self.pad_token_id = pad
        self.bos_token_id = bos
        self.eos_token_id = eos
        self._vocab = {f"t{i}": i for i in range(size)}

    def get_vocab(self):
        return self._vocab

    def __len__(self):
        return len(self._vocab)


def test_tokenizer_layout_validation_65k_and_128k():
    module = load_script("scripts/clm/02_build_tokenizer.py")
    assert module.validate_family_layout(LayoutTokenizer(0, 1, 7, 10), "65k")["append_start"] == 10
    metadata = module.validate_family_layout(LayoutTokenizer(124893, 124894, 124900, 128000), "128k")
    assert metadata["special_ids"]["eos"] == 124900
    with pytest.raises(ValueError):
        module.validate_family_layout(LayoutTokenizer(0, 1, 7, 128000), "128k")


def test_padded_embedding_capacity_is_preserved_and_accepted():
    module = load_script("scripts/clm/02_build_tokenizer.py")
    assert module.embedding_rows_after_extension(65_536, 64_432) == 65_536
    assert module.embedding_rows_after_extension(65_536, 68_400) == 68_400

    padded = LayoutTokenizer(0, 1, 7, 64_432)
    verifier_vocab_compatible(65_536, padded)
    with pytest.raises(ValueError):
        verifier_vocab_compatible(64_000, padded)


def test_resize_append_compatibility_and_specials():
    module = load_script("scripts/clm/02b_resize_ext_to_size.py")
    base = LayoutTokenizer(0, 1, 7, 10)
    extended = LayoutTokenizer(0, 1, 7, 12)
    assert module._verify_compatible_base(base, extended, 7) == 10
    assert module.embedding_rows_after_extension(12, 14) == 14

    old_weight = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    new_weight = torch.cat([old_weight.clone(), torch.zeros(2, 2)])
    new_weight[10:] = -1
    assert module.semantic_prefix_preserved(old_weight, new_weight, 10)
    new_weight[9] = -1
    assert not module.semantic_prefix_preserved(old_weight, new_weight, 10)

    extended._vocab["t0"] = 9
    with pytest.raises(ValueError):
        module._verify_compatible_base(base, extended, 7)


def test_tie_flags_and_merge_tolerance():
    config = SimpleNamespace(tie_word_embeddings=True)
    assert lire_tie_flag(config)
    assert lora_requires_weight_tying(config, ["embed_tokens", "lm_head"])
    assert not lora_requires_weight_tying(config, None)
    ecrire_tie_flag(config, False)
    assert not config.tie_word_embeddings
    assert not lora_requires_weight_tying(config, ["embed_tokens", "lm_head"])
    merge = load_script("scripts/clm/05_merge_adapter.py")
    assert merge.merge_within_tolerance(0.5, 0.5019, 0.002)
    assert not merge.merge_within_tolerance(0.5, 0.503, 0.002)


def test_merge_finds_nested_run_by_journal_id(tmp_path: Path):
    merge = load_script("scripts/clm/05_merge_adapter.py")
    run_dir = tmp_path / "runs/clm/350m/v2_ext4000_lora"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "journal.json").write_text(
        json.dumps({"run": "v2_350m_ext4000_lora"}),
        encoding="utf-8",
    )

    assert (
        merge._find_run_dir(tmp_path, tmp_path, "v2_350m_ext4000_lora")
        == run_dir
    )
    assert (
        merge.artifact_reference(run_dir / "adapter", tmp_path)
        == "runs/clm/350m/v2_ext4000_lora/adapter"
    )
    assert (
        merge.artifact_reference(tmp_path.parent / "external/private/model", tmp_path)
        == "model"
    )


class TinyTokenizer:
    bos_token_id = None
    cls_token_id = None
    model_max_length = 100

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 4 for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=3)

    def forward(self, input_ids):
        shape = (*input_ids.shape, 4)
        return SimpleNamespace(logits=torch.zeros(shape, device=input_ids.device))


def test_windowed_bpb_scores_boundaries_and_matching_bytes():
    result = compute_bpb(UniformModel(), TinyTokenizer(), ["abcde"], max_len=99, device="cpu")
    assert result["n_tokens"] == 4
    assert result["n_bytes"] == 4
    assert result["bpb"] == pytest.approx(2.0)


def base_clm_config():
    return {
        "run": "x", "size": "350m", "base_id": "LiquidAI/x", "methode": "lora",
        "tokenizer": "tokenizers/ext", "tokenizer_family": "65k", "eos_token_id": 7,
        "dataset_id": "galsenai/cleaned_data_for_clm",
        "lora": {"modules_to_save": ["embed_tokens", "lm_head"]},
        "entrainement": {"max_steps": 1, "seq_len": 8, "micro_bs": 1, "grad_acc": 1, "lr": 1e-4, "seed": 42},
    }


def test_config_fail_fast_invariants():
    assert validate_clm_config(base_clm_config())["methode"] == "lora"
    bad = base_clm_config()
    bad["methode"] = "magic"
    with pytest.raises(ConfigError):
        validate_clm_config(bad)
    bad = base_clm_config()
    bad["lora"] = {}
    with pytest.raises(ConfigError):
        validate_clm_config(bad)
    bad = base_clm_config()
    bad["eos_token_id"] = 2
    with pytest.raises(ConfigError):
        validate_clm_config(bad)
    bad = base_clm_config()
    bad["dataset_id"] = "galsenai/wolof_centalized_corpus"
    with pytest.raises(ConfigError):
        validate_clm_config(bad)


def test_all_training_configs_validate():
    for path in (ROOT / "configs" / "clm").rglob("*.yaml"):
        validate_clm_config(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    for path in (ROOT / "configs" / "scaling").glob("*.yaml"):
        validate_clm_config(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    for path in (ROOT / "configs" / "sft").rglob("*.yaml"):
        if path.name not in {"mixture_v1.yaml", "dpo_optional.yaml"}:
            validate_sft_config(yaml.safe_load(path.read_text(encoding="utf-8")), path)


def test_eval_yaml_schema_and_unique_outputs():
    outputs = set()
    paths = list((REPO / "Evaluation" / "configs" / "clm").glob("lfm_*.yaml"))
    assert len(paths) == 15
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert {"base_model_id", "corpus_csv", "output_json"} <= config.keys()
        assert config["output_json"] not in outputs
        outputs.add(config["output_json"])
        if config.get("placeholder"):
            assert "galsenai/" in config["base_model_id"]


def test_synthetic_validation_and_budget(tmp_path: Path):
    reason = load_script("scripts/data/22_synth_reasoning.py")
    good = {"prompt": "p", "reasoning": "1. benn\n2. ñaar", "answer": "3", "gold_answer": "3"}
    assert reason.valid_reasoning(good) == (True, "ok")
    mismatch = {**good, "answer": "4"}
    assert reason.valid_reasoning(mismatch)[1] == "answer_mismatch"
    assert token_budget(70, 30)["synthetic_fraction"] == 0.3
    with pytest.raises(ValueError):
        token_budget(60, 40)
    path = tmp_path / "x"
    path.write_text("lock", encoding="utf-8")
    assert len(sha256_file(path)) == 64


def test_publication_manifest_rejects_missing_review():
    module = load_script("scripts/publication/40_build_model_card.py")
    with pytest.raises(ValueError):
        module.validate_manifest({"model_name": "x"})
