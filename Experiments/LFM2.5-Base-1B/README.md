# LFM2.5-Base-1B (maps to LiquidAI/LFM2.5-1.2B-Base)

Wolof adaptation of Liquid AI **LFM2.5** across the full text family (230M → 8B-A1B), with hero track on **1.2B-Base**.

| | |
|---|---|
| **Hero base** | [`LiquidAI/LFM2.5-1.2B-Base`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) |
| **CPT data** | [`galsenai/cleaned_data_for_clm`](https://huggingface.co/datasets/galsenai/cleaned_data_for_clm) |
| **SFT data** | [`galsenai/wolof_sft`](https://huggingface.co/datasets/galsenai/wolof_sft) |
| **Raw corpus (audit)** | [`galsenai/wolof_centalized_corpus`](https://huggingface.co/datasets/galsenai/wolof_centalized_corpus) |
| **Master plan** | [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) |
| **Comprehensive TODOs** | [`COMPREHENSIVE_TODOS.md`](./COMPREHENSIVE_TODOS.md) |
| **File/line anchors** | [`FILE_TODOS.md`](./FILE_TODOS.md) |
| **Per-size plans** | [`docs/MODEL_PLANS.md`](./docs/MODEL_PLANS.md) |
| **Datasets** | [`docs/DATASETS.md`](./docs/DATASETS.md) |
| **Synthetic** | [`docs/SYNTHETIC_DATA.md`](./docs/SYNTHETIC_DATA.md) |
| **Small→scale** | [`docs/SCALING_LADDER.md`](./docs/SCALING_LADDER.md) |
| **Template** | `../SmolLM3-3B-Base/` |
| **Eval** | `../../Evaluation/scripts/evaluate_clm.py` |

## Status

Scaffold + data scripts ready (`uv` project, `common/`, audit card, SFT mixture builder, append-only tokenizer script). **HF_TOKEN required** for private Hub loads (`cleaned_data_for_clm`, `wolof_sft`). Canonical queue: [`COMPREHENSIVE_TODOS.md`](./COMPREHENSIVE_TODOS.md).

## Setup (`uv`)

Prefer `uv run …` inside this directory; do not mix system Python with training deps.

```bash
cd Experiments/LFM2.5-Base-1B
uv python install
uv sync --group train --group eval --group data --group dev
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Data-only / CI hosts (no CUDA wheels required):

```bash
uv sync --group data --group dev
```

### Hugging Face auth

Private `galsenai/*` datasets need a token:

```bash
uv run huggingface-cli login
# or export HF_TOKEN=hf_...
```

### GPU / LFM class check

```bash
uv run python scripts/env/00_verify_env.py
# data-only:
uv run python scripts/env/00_verify_env.py --allow-cpu
```

### Cache dirs (train machine)

Point HF and uv caches at a large disk before long downloads:

```bash
export HF_HOME=/path/to/hf_cache
export UV_CACHE_DIR=/path/to/uv_cache
```

On Linux train hosts, pin torch to the CUDA index documented in `pyproject.toml` (`cu128` preferred; uncomment `[tool.uv.sources]` after verifying the wheel index).

## Intended first commands

```bash
uv run python scripts/data/20_audit_datasets.py --out results/data_audit.json
uv run python scripts/clm/06_probe_modules.py --base-id LiquidAI/LFM2.5-350M-Base
uv run python scripts/clm/02_build_tokenizer.py --config configs/tokenizer-ext4000-65k.yaml
uv run python scripts/clm/03_train.py --config configs/scaling/smoke_350m_ext_lora.yaml --smoke
# … promote via SCALING_LADDER gates …
```

## Family (train from Base)

| Size | Base Hub | Tok | Role |
|---|---|---|---|
| 230M | `LFM2.5-230M-Base` | 65k | Edge smoke / GGUF |
| 350M | `LFM2.5-350M-Base` | 65k | Ablation workhorse |
| **1.2B** | `LFM2.5-1.2B-Base` | 65k | **Hero** |
| 2.6B | `LFM2.5-2.6B-Base` | 128k | Quality + long-context |
| 8B-A1B | `LFM2.5-8B-A1B-Base` | 128k | MoE agentic |

## Data policy (short)

- CPT **only** on `cleaned_data_for_clm` (code-stripped, clean test).
- SFT on **reweighted** `wolof_sft` (do not train raw 58% translation).
- Synthetic for reasoning / long-context / prefs — gated (see SYNTHETIC_DATA).
- Centralized corpus = audit + synth seeds, not default CPT.

## Code-complete gated paths

Offline validation now covers the 65k and 128k tokenizer layouts, CLM/SFT config safety, windowed BPB, SFT preparation/training/evaluation/merge, synthetic filters, and publication manifests. Compute gates remain intentionally separate.

```bash
# Production SFT mixture (requires private-dataset auth)
uv run python scripts/data/21_build_sft_mixture.py --config configs/sft/mixture_v1.yaml

# Prepare and smoke SFT after a merged CPT checkpoint exists
uv run python scripts/sft/10_prepare_sft.py --input data/sft/mixture_v1 --smoke
uv run python scripts/sft/11_train_sft.py --config configs/sft/1.2b/smoke.yaml --smoke

# Validate family commands without training
uv run python scripts/family/30_run_family.py --size 350m --dry-run
uv run --directory ../../Evaluation python scripts/run_lfm_family.py --dry-run
```

## Production orchestration and Trackio

`configs/production_pipeline.yaml` is the production source of truth. Before
training, it requires authenticated upload and parsed readback of the exact
smoke-run fragment from the private Hugging Face Bucket, including checksum and
object revision. A durable local Trackio database is also required. The live
private Space dashboard is a separate availability gate: `RUNNING` is ideal,
while a quota-paused dashboard is recorded as degraded and does not block
training because metrics continue to persist remotely in the Bucket. Local-only
tracking never satisfies the production gate.

```bash
# Validate commands and dependencies without GPU/network work
TRACKIO_DISABLED=1 uv run python scripts/pipeline/50_run_production.py --dry-run

# Production/resume (set only in the detached queue environment)
uv run python scripts/pipeline/50_run_production.py --resume

# Machine-readable monitoring
uv run trackio list runs --project lfm25-wolof --json
uv run trackio list alerts --project lfm25-wolof --json
uv run trackio get run --project lfm25-wolof --run <run-name> --json
uv run trackio get metric --project lfm25-wolof --run <run-name> --metric eval_loss --json
```

The 2.6B 128k and 8B-A1B MoE stages remain runtime-gated: the queue cannot train
them until tokenizer-special and model-probe target validation succeeds.
