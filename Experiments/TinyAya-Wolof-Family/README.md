# Tiny Aya Wolof Family (3.35B)

Distinct experiment tree for adapting [Cohere Labs Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-base) to informal Wolof.

## Access & license

- **Gated Hub models** — accept terms for `CohereLabs/tiny-aya-base`, `tiny-aya-earth`, and `tiny-aya-global` before download (`huggingface-cli login` + gate accept). **CPT training still requires `tiny-aya-base` gate** (`configs/family.yaml` / `model_id`); fertility + tokenizer download default to `tiny-aya-earth` (shared 262k CohereTokenizer) when base is harder to access.
- **License** — stock Tiny Aya and derivatives: **CC-BY-NC-4.0** + [Cohere Labs AUP](https://cohere.com/terms-of-use). Research / non-commercial only. Do not republish as Apache/MIT. Model cards keep this note via `43_prepare_artifact.py`.
- Publish private research artifacts under `Tonic/` with the same non-commercial notice.

## Hub metrics (LFM-style)

Publishing a CPT / Instruct / DPO merge via `06_publish_artifacts.py` attaches metrics when local files exist:

1. `43_prepare_artifact.py` → `metrics.json` + `training_manifest.json` + CC-BY-NC card on the artifact
2. Upload `metrics_clm.json` (primary ortho ns+mixed BPB) and optional `metrics_clm_original_test.json`
3. `56_publish_hub_metrics.py` → composed HF `metrics.json` + README `model-index` + Evaluation section

Stable paths: `results/bpb/` (see `configs/publication/eval_family.yaml`), `results/sft/eval_from_*`.

## Pipeline

1. Materialize NTP `70_20_10` + stratified SFT
2. Ortho-label into **two variants** (FR/EN kept in NTP; `not_wolof` dropped):
   - **ns** — keep `non_standard`+`mixed` → `data/ortho_ns_mixed_v1` + `<|ortho_ns|>`
   - **std** — keep `standard`+`mixed` → `data/ortho_std_mixed_v1` + `<|ortho_std|>`
3. Fertility probe on stock CohereTokenizer (tune `n_tokens`)
4. Append-only vocab extension (both anchors; mine ns+std CPT; preserve `bos=2` / `eos=3` / `pad=0`)
5. Resize same tokenizer onto earth + global Instruct starts
6. **Per ortho**: CPT **3 epochs** → triple SFT **3 epochs** (earth / global / CPT) → pick best → optional DPO
7. Hub publish (datasets + models, private) for both tracks

**Incremental Hub uploads:** every successful stage of `50_run_family.py` immediately publishes available artifacts via `06_publish_artifacts.py` (datasets, fertility JSON, tokenizer, ext inits, CPT/Instruct merges, BPB/eval JSON, state). After `merge_cpt`, each `sft_*` merge, `dpo`, `pick_best`, and `eval`, model repos also get metrics/model-index when files exist. Look for `HUB_UPLOAD_OK` / `HUB_METRICS_OK`; Hub failures are logged without aborting training.

## H200 kernels (torch 2.11 + cu128)

Keep `torch==2.11.0` from the **cu128** index (not default cu130 wheels). CPT/SFT/DPO load via `common/kernels.py`:

1. **Attention** — `attn_implementation=flash_attention_3` with fallbacks: Hub `kernels-community/vllm-flash-attn3` → FA2 → SDPA (cuDNN-first)
2. **Fused CE** — Apple `cut-cross-entropy` (`cce_patch`) for the ~262k tied vocab (soft-fail if Cohere2 patch missing)
3. **Packing** — leave `packing: false` until FA3 smoke is clean; then flip SFT configs

Optional: `pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128` or rely on `kernels` alone.

## Trackio / Gradio

CPT, SFT, and DPO log to project `tinyaya-wolof` via Hugging Face Trackio (`report_to=trackio`).

```bash
# One-time: private Gradio Space + Bucket + .env.trackio
uv sync --group train
uv run python scripts/publication/44_setup_trackio_space.py --preferred-namespace Tonic
# Writes .env.trackio with TRACKIO_SPACE_ID / TRACKIO_BUCKET_ID
# Dashboard: https://huggingface.co/spaces/Tonic/tinyaya-wolof-trackio
```

`50_run_family.py` loads `.env.trackio` and prints `TRACKIO_READY`. Production runs set `TINYAYA_PRODUCTION=1` (requires Space + Bucket). Smoke may proceed without them; set `TRACKIO_DISABLED=1` only to silence logging.

Anchoring uses the Cohere **developer system preamble** with `<|lang_wo|>` `<|ortho_ns|>` — no Qwen `<think>` wrap. SFT labels supervise tokens between `<|START_RESPONSE|>` … `<|END_RESPONSE|>`.

## Quick start (GPU volume)

```bash
cd /mnt/volume_w2l8nkp/Wolof-LLM-main/Experiments/TinyAya-Wolof-Family
uv sync --group train --group data
uv run python scripts/publication/44_setup_trackio_space.py --preferred-namespace Tonic
# Accept gated access first, then:
# Dual orthography smoke (ns + std). Use --ortho ns|std to run one track.
uv run python scripts/publication/50_run_family.py --smoke --ortho both
```

Production (after smoke):

```bash
uv run python scripts/publication/50_run_family.py --resume --ortho both
```

Single stage:

```bash
uv run python scripts/publication/50_run_family.py --only-stage fertility
```

Durable state: `results/full_pipeline/tinyaya_state.json`. Success marker: `TINYAYA_FAMILY_OK`.

## Hub map

| Role | Repo |
|------|------|
| CPT data | `Tonic/TinyAya-Wolof-NTP-ortho_ns_mixed_v1` |
| SFT filtered | `Tonic/TinyAya-Wolof-SFT-ortho_ns_mixed_stratified_v1` |
| SFT prepared | `Tonic/TinyAya-Wolof-SFT-prepared_anchored_v1` |
| Prefs | `Tonic/TinyAya-Wolof-prefs_v1` |
| CPT model | `Tonic/TinyAya-3.35B-Wolof-CPT-v1` |
| Instruct (from CPT) | `Tonic/TinyAya-3.35B-Wolof-Instruct-v1` |
| InstructEarth | `Tonic/TinyAya-3.35B-Wolof-InstructEarth-v1` |
| InstructGlobal | `Tonic/TinyAya-3.35B-Wolof-InstructGlobal-v1` |
| DPO | `Tonic/TinyAya-3.35B-Wolof-DPO-v1` |

## Non-goals

Qwen/LFM size ladders, `tiny-aya-fire`/`water`/thinker/32K, editing other experiment trees, commercial redistribution beyond CC-BY-NC.
