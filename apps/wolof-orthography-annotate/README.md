---
title: Wolof Orthography Annotator
emoji: 📝
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Crowdsource Wolof standard vs non-standard spelling labels
tags:
  - wolof
  - orthography
  - annotation
  - crowdsourcing
  - african-languages
---

# Wolof Orthography Annotator

Crowdsourcing Space for labeling Wolof **standard** vs **non_standard** Latin
spelling (plus `mixed` / `not_wolof`). When enough volunteers agree, the row is
written to a consensus Hub dataset.

See [ORTHOGRAPHY.md](ORTHOGRAPHY.md) for why this matters for LLM training.

## Flow

1. Volunteer enters an id (HF username recommended).
2. App shows **one** source text.
3. Volunteer picks a label and submits.
4. Vote is appended to `VOTES_DATASET`.
5. When `CONSENSUS_THRESHOLD` distinct volunteers pick the **same** label,
   the row is sealed into `CONSENSUS_DATASET`.

## Space secrets / variables

| Name | Required | Default | Role |
|---|---|---|---|
| `HF_TOKEN` | **yes** (write) | — | Create/append votes + consensus datasets |
| `SOURCE_DATASET` | no | `galsenai/cleaned_data_for_clm` | Corpus to annotate |
| `SOURCE_CONFIG` | no | _(none)_ | Optional dataset config |
| `SOURCE_SPLIT` | no | `train` | Split name |
| `SOURCE_TEXT_COLUMN` | no | `text` | Text field |
| `SOURCE_ID_COLUMN` | no | _(none)_ | Optional stable id column |
| `VOTES_DATASET` | no | `galsenai/wolof_orthography_votes` | Per-volunteer votes |
| `CONSENSUS_DATASET` | no | `galsenai/wolof_orthography_consensus` | Gold labels |
| `CONSENSUS_THRESHOLD` | no | `2` | Agreeing volunteers to seal (2–10) |
| `SOURCE_MAX_ROWS` | no | `50000` | Cap rows loaded (`0` = all) |
| `MAX_TEXT_CHARS` | no | `1200` | Display truncation |
| `APP_TITLE` | no | `Wolof Orthography Annotator` | UI title |

Set `CONSENSUS_THRESHOLD=3` for stricter gold.

## Deploy on Hugging Face Spaces

1. Create a **Gradio** Space (CPU is enough).
2. Upload **this folder** as the Space root (or push via git).
3. Add secret **`HF_TOKEN`** with **write** access to the votes/consensus repos
   (and read on the source dataset if private).
4. Optionally set the env vars above as Space variables.
5. On first vote, empty private vote/consensus datasets are created if missing.

## Local run

```bash
cd apps/wolof-orthography-annotate
pip install -r requirements.txt
cp .env.example .env   # fill HF_TOKEN + dataset repos
python app.py
```

## Labels

- `standard` — school / CLAD Latin with special letters
- `non_standard` — informal phone/social spelling
- `mixed` — both or ambiguous
- `not_wolof` — not usable Wolof

Heuristics in the UI are **hints only**; they never auto-label.
