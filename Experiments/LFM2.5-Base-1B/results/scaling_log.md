# Scaling ladder log — Stages 0–14 + early gates
# Fill Status / Notes after each gate. See docs/SCALING_LADDER.md + COMPREHENSIVE_TODOS.md.

| Gate / Stage | Description | Status | Date | Notes |
|---|---|---|---|---|
| G-DATA | `results/data_audit.json` + CLM test=4282; SFT task table ±1% | **pass** | 2026-09-16 | Full private-Hub audit: CLM 958,646/4,282, raw 1,018,520, SFT 990,320/2,250; null text 0; overlap sample 0/2,000. |
| G-UV | `uv sync --group train`; `00_verify_env.py` green; Lfm2 loads | **pass** | 2026-09-16 | NVIDIA H200: torch 2.11.0+cu128, CUDA available, bf16 supported; full dependency sync and env verifier passed; 350M loaded and completed a finite tiny forward. |
| G-COMMON | `common.*` imports; Hub dry-run `--max-rows 100` | **pass** | 2026-09-16 | Prior Hub dry-run passed; offline suite now covers config, metrics, synthetic and tokenizer helpers. |
| G-TOK | Tokenizer ext validations; fertility drop; eos preserved | **micro-smoke only** | 2026-09-16 | Temporary ext32 pipeline preserved eos=7 and padded embedding capacity; round trip 500/500. No fertility drop on the tiny sample and norm ratio 0.714 warning; canonical ext4000 remains pending. |
| G-TOK-65K | 65k family ext4000 green | **pending (ext32 smoke passed)** | 2026-09-16 | Canonical ext4000 artifacts/config were not modified. |
| G-MIX | mixture_v1 translation ≤35%; val stratified | **pass** | 2026-09-16 | Full production mixture: 338,408 train / 6,906 val from 345,314 selected rows; seven tasks represented; translation 34.75%; Hub test untouched. |
| G-SCRIPTS-CLM | smoke train 350M completes, no NaN | **micro-smoke only** | 2026-09-16 | Temporary default-tokenizer LoRA completed 2 finite steps; temporary ext32 LoRA with `modules_to_save` completed 1 finite step. Canonical 10-step default+ext smokes remain pending. |
| Stage 0 | Env + forward pass 350M/1.2B + probe modules | **partial** | 2026-09-16 | H200 env, 350M module probe, and 350M tiny forward passed. 1.2B config-only/meta module probe passed without downloading weights; full 1.2B load/forward remains pending. |
| Stage 1 | Data audit fresh | **pass** | 2026-09-16 | Full audit artifacts generated from all three private Hub datasets. |
| Stage 2 | Tokenizer ext4000 | pending | | |
| Stage 3 | Smoke CPT 350M (default + ext LoRA) | **micro-smoke only** | 2026-09-16 | 2-step default LoRA and 1-step ext32 LoRA passed on short sequence; this does not satisfy the canonical 10-step gate. |
| Stage 4 | Mini CPT 350M | pending | | |
| Stage 5 | Smoke CPT 1.2B | pending | | |
| Stage 6 | Mini CPT 1.2B | pending | | |
| Stage 7 | Hero CPT 1.2B (full) | pending | | |
| Stage 8 | Merge + BPB gate | pending | | |
| Stage 9 | SFT smoke on merged | pending | | |
| Stage 10 | SFT mini mixture | pending | | |
| Stage 11 | SFT full mixture_v1 | pending | | |
| Stage 12 | Eval harness / Hub metrics | pending | | |
| Stage 13 | Family 230M / 350M / 2.6B | pending | | |
| Stage 14 | Publish / findings | pending | | |

## Code readiness update — 2026-09-16

SFT, optional DPO, synthetic filtering, 128k layout validation, per-size config placeholders, evaluation orchestration, and publication-manifest validation are implemented and offline-tested. The full production SFT mixture is now validated on the authorized VM. No production model training, model evaluation, human review, or publication gate changed status.

## Trackio production gate — 2026-09-16

Trackio 0.38.0 is locked into the `uv` environment. Private Hub model
write/upload/readback passed in namespace `Tonic`. `G-TRACKIO-REMOTE` passed:
the setup gate made the Trackio Bucket private, uploaded a new smoke run,
downloaded and parsed the exact metric record, confirmed its SHA-256 and Bucket
object revision, queried the durable local database, and retained private Space
status. The separate live-dashboard gate is degraded with
`dashboard_status=paused_quota`; this is non-blocking because no local-only
fallback is accepted and Bucket persistence is independently verified.

The detached resumable production queue launched at 2026-09-16 20:04 UTC with
PID/session `1061998`. It survived SSH disconnect, completed the Trackio, Hub
write, environment, full-mixture, and 350M probe stages, and was progressing
through the 1.2B probe at the first observation. The H200 reported 10% GPU
utilization and 10,827 MiB used; approximately 1.2 TiB remained free. Production
model quality and completion gates remain pending.

### 15-minute monitor tick — 2026-09-16 20:28 UTC

The queue had stopped at `resize_350m_ext4000` with 9 stages complete. The
failure was a source assertion bug: the padded LFM embedding tensor was compared
against a shorter semantic-token prefix after extension rows correctly
overwrote padding capacity. The resize now preserves and compares only semantic
base rows, retains padded capacity, and records physical embedding dimensions.
The regression test covers padding-row overwrite. Local and remote validation
both passed (31 tests, Ruff, compileall).

The queue resumed from checkpoint without rerunning completed stages.
`resize_350m_ext4000` passed on attempt 2, `resize_350m_ext2000` also completed,
and `resize_350m_ext8000` completed. The next red gate correctly rejected stale
generic LoRA placeholders (`o_proj`, `gate_proj`, `up_proj`, `down_proj`) that
do not exist in the 350M probe. All three validated 350M recipes now use the
probe-approved `q_proj`, `k_proj`, and `v_proj` targets; a regression test locks
that contract. Validation passed locally and remotely (32 tests, Ruff,
compileall, production dry-run), and the queue resumed again without
duplicating completed stages.

At the final observation, 13 stages were complete, no stage was failed, and
`tokenizer_ablation_350m` had started its first 400-step ext2000 LoRA run under
PID/session `1065987`. The private Trackio Bucket returned current telemetry for
that exact run (`7b9f737313144985a21ad76a7a0b5fbc`), confirming persistence
after both resumes. No loss or BPB had been emitted yet while data packing was
still CPU-bound; the H200 held 8,973 MiB with 0% instantaneous utilization.
Approximately 1.2 TiB remained free. No private publication manifest changes
were present.

### 15-minute monitor tick — 2026-09-16 20:43 UTC

No new stage completed and no red gate appeared: the queue remained healthy at
13 complete stages in the first ext2000 run of `tokenizer_ablation_350m`.
Training reached step 39/400 with finite loss 4.298; baseline BPB was 3.6607.
The data pipeline reported 60,895 train blocks and 317 evaluation blocks.
Fertility and post-training BPB were not yet emitted.

The H200 was actively training at 37% instantaneous utilization with 15,041 MiB
VRAM used; approximately 1.2 TiB remained free. The private Trackio Bucket had a
readable fragment updated at 20:43:48 UTC and continued recording current
pipeline telemetry. No publication manifest was created or changed. No
intervention was required.

### 15-minute monitor tick — 2026-09-16 20:58 UTC

The pipeline remained healthy at 13 completed top-level stages with
`tokenizer_ablation_350m` active. Its ext2000 sub-run completed all 400 steps:
finite train loss 2.584, final evaluation loss 2.315, and BPB improved from
3.6607 to 1.5845 (56.7%). The ext4000 sub-run then started and progressed to
roughly halfway, with latest observed finite loss 2.469 and baseline BPB
3.7926. Tokenizer metadata reports fertility 2.1693 before and after for the
ext2000/ext4000/ext8000 tokenizers.

The H200 reported 30% instantaneous utilization and 25,899 MiB VRAM used;
approximately 1.2 TiB remained free. Authenticated private-Bucket listing and
readback were fresh through 20:58:36 UTC. No red gate, failed stage, or
publication-manifest change appeared, so no intervention was performed.

### 15-minute monitor tick — 2026-09-16 21:13 UTC

The top-level queue remained healthy at 13 completed stages. The ext4000
ablation completed 400/400 with mean train loss 2.587, final evaluation loss
2.312, and BPB 3.7926→1.5790 (58.4%). The ext8000 run then reached approximately
200/400 with latest finite loss 2.390, evaluation loss 2.603, and baseline BPB
3.9775. Reported tokenizer fertility remained 2.1693 before and after.

The H200 reported 41% utilization and 26,763 MiB VRAM used; approximately
1.2 TiB remained free (60% disk usage). Authenticated private-Bucket readback
confirmed the active ablation run and an object revision uploaded at 21:08:28
UTC. No failed/red gate or publication change appeared. No intervention was
required.

### 15-minute monitor tick — 2026-09-16 21:28 UTC

`tokenizer_ablation_350m` completed, raising the top-level count from 13 to 14,
and `cpt_350m` started. By final BPB, ext8000 is the metric winner:
3.9775→1.5612 (60.8%), versus ext4000 1.5790 and ext2000 1.5845. Its mean train
loss was 2.592 and final evaluation loss 2.306. Fertility remained tied at
2.1693, and no separate winner-selection artifact was emitted.

Within `cpt_350m`, the default-tokenizer smoke produced train/eval losses
5.448/5.229 and BPB 3.3941→3.2228; the extended-tokenizer smoke produced
4.893/4.422 and BPB 3.7926→2.7250. The family then advanced to the 1,500-step
default-tokenizer run, which was still preparing data at observation.

The H200 held 8,971 MiB VRAM with 0% instantaneous utilization during that CPU
preparation; approximately 1.2 TiB remained free (60% disk usage). Authenticated
private-Bucket readback found 169 files and a latest immutable object at
21:27:16 UTC. No failed/red gate or private publication change appeared, so no
intervention was required.

### 15-minute monitor tick — 2026-09-16 21:43 UTC

The queue remained healthy at 14 completed stages with
`configs/clm/350m/v1_default_lora.yaml` (`v1_350m_default_lora`) active inside
`cpt_350m`. It reached approximately step 600/1500; latest finite train loss
was 3.177, latest evaluation loss was 3.245 at step 500, and baseline BPB was
3.3941. Final BPB was not yet available.

The H200 reported 38% utilization and 84,134 MiB VRAM used; approximately
1.2 TiB remained free (60% disk usage). Authenticated private-Bucket readback
confirmed both the orchestrator and exact v1 run IDs in an immutable object
uploaded at 21:32:23 UTC. The durable local Trackio DB remained actively
updated, with current records pending the next Bucket flush. No failed/red gate
or publication change appeared, so no intervention was required.

### 15-minute monitor tick — 2026-09-16 21:58 UTC

The queue remained healthy at 14 completed stages. The active
`v1_350m_default_lora` run advanced from approximately 600 to 1300/1500. Latest
finite train loss was 3.023, latest evaluation loss was 2.961 at step 1200, and
final BPB remained pending from the 3.3941 baseline. Checkpoints 500 and 1000
were present, preserving resumability.

The H200 reported 100% utilization and 84,212 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). The private Bucket still contained the
exact active run ID, with its latest immutable object unchanged at 21:32:23 UTC.
The local Trackio DB was updated within two seconds of inspection and held
pending active-run records for the next remote flush. No failed/red gate or
publication change appeared, so no intervention was required.

### 15-minute monitor tick — 2026-09-16 22:13 UTC

`v1_350m_default_lora` completed 1500/1500 with final train/evaluation losses
2.9237/2.9433 and BPB 3.3941→2.1247 (37.4% gain). Its final checkpoint is
`checkpoint-1500`. The queue remained healthy at 14 top-level stages and
advanced to `configs/clm/350m/v1_default_full.yaml`, which reached step 505/1500
with latest train/evaluation losses 2.210/2.221.

The H200 reported 100% utilization and 86,174 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Private-Bucket readback was fresh
through 22:09:06 UTC and contained the successor run. No red gate or publication
change appeared.

The completed LoRA run's local tracking metadata was incorrectly marked
degraded after Transformers had already finalized Trackio and the wrapper called
`finish()` a second time. Bucket persistence was successful. The wrapper was
made idempotent for this exact initialized-run case, regression-tested locally
and remotely (10 tests plus Ruff), and synced before subsequent child runs.

### 15-minute monitor tick — 2026-09-16 22:28 UTC

The queue remained healthy at 14 completed top-level stages.
`v1_350m_default_full` advanced from step 505 to approximately 1300/1500. Its
latest finite train loss was 2.120 and evaluation loss was 2.136 at step 1200.
Final BPB remained pending, with checkpoints 500 and 1000 available.

The H200 reported 51% instantaneous utilization and 86,216 MiB VRAM used;
approximately 1.1 TiB remained free (62% disk usage). Authenticated private
Bucket readback retained the active full-tuning run, with its latest immutable
object at 22:09:06 UTC; the durable local database continued updating and held
pending active-run records. No failure, red gate, publication change, or
intervention appeared.

### 15-minute monitor tick — 2026-09-16 22:43 UTC

`v1_350m_default_full` completed 1500/1500 with final train/evaluation losses
2.1086/2.1354 and BPB 3.3436→1.4400 (56.9% gain). Its final checkpoint is
`checkpoint-1500`. The queue remained healthy at 14 completed top-level stages
and advanced to `configs/clm/350m/v1_ext4000_lora.yaml`, observed at step
302/1500 with latest train loss 2.321 at step 300 and evaluation loss 2.421 at
step 200.

The H200 reported 47% utilization and 86,376 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Authenticated private-Bucket readback
was fresh through 22:37:08 UTC and contained the exact active extended-tokenizer
run. No failed/red gate or publication change appeared, and no new intervention
was required.

### 15-minute monitor tick — 2026-09-16 22:58 UTC

The queue remained healthy at 14 completed top-level stages.
`v1_350m_ext4000_lora` advanced from step 302 to 989/1500. The latest emitted
train loss was 2.107 at step 900 and evaluation loss was 2.123 at step 800.
Final BPB remained pending; `checkpoint-500` was durable.

The H200 reported 100% utilization and 86,414 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Authenticated private-Bucket readback
retained the exact active run, with the latest immutable object at 22:37:08 UTC;
the durable local Trackio database continued updating with active-run records
pending the next flush. No failed/red gate, publication change, or intervention
appeared.

### 15-minute monitor tick — 2026-09-16 23:13 UTC

`v1_350m_ext4000_lora` completed 1500/1500 with final train/evaluation losses
2.0261/2.0483 and BPB 3.7926→1.4784 (61.0% gain). Its final checkpoint is
`checkpoint-1500`, and its Trackio state is `synced`, confirming the
idempotent-finalization recovery on a newly started run.

The queue remained healthy at 14 completed top-level stages and advanced to
`configs/clm/350m/v1_ext4000_full.yaml`. That run was loading the model and
preparing data; no optimizer step or BPB result was available yet. The H200
reported 0% instantaneous utilization and 69,838 MiB VRAM used during this
preparation; approximately 1.1 TiB remained free (62% disk usage). The private
Bucket was fresh through 23:10:31 UTC. No failed/red gate, publication change,
or intervention appeared.

### 15-minute monitor tick — 2026-09-16 23:28 UTC

The queue remained healthy at 14 completed top-level stages.
`v1_350m_ext4000_full` advanced from data preparation to step 720/1500. Its
latest finite train loss was 2.102 at step 700 and evaluation loss was 2.118 at
step 600. Final BPB remained pending; `checkpoint-500` was durable.

The H200 reported 100% utilization and 87,114 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). The private Bucket was fresh through
23:16:53 UTC, and local Trackio state remained active and durable. No failed/red
gate, publication change, or intervention appeared.

### 15-minute monitor tick — 2026-09-16 23:43 UTC

`v1_350m_ext4000_full` completed 1500/1500 with final train/evaluation losses
2.0366/2.0626 and BPB 3.7629→1.3869 (63.1% gain). Its final checkpoint is
`checkpoint-1500`, and Trackio finalized as `synced`.

The queue remained healthy at 14 completed top-level stages and advanced to
`configs/clm/350m/v2_default_lora.yaml`. That run was loading/preparing data, so
no optimizer step or BPB result was available yet. The H200 reported 4%
instantaneous utilization and 69,570 MiB VRAM used during preparation;
approximately 1.1 TiB remained free (63% disk usage). The private Bucket was
fresh through 23:41:17 UTC. No failed/red gate, publication change, or
intervention appeared.

### 15-minute monitor tick — 2026-09-16 23:58 UTC

The queue remained healthy at 14 completed top-level stages.
`v2_350m_default_lora` advanced from preparation to 603/3000. Its latest finite
train loss was 3.455 and evaluation loss was 3.449, both emitted at step 500.
Final BPB remained pending; checkpoints 400 and 500 were present.

The H200 reported 99% utilization and 84,544 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Authenticated private-Bucket readback
was fresh through 23:46:15 UTC and contained the exact active v2 run. No
failed/red gate, publication change, or intervention appeared.

### 15-minute monitor tick — 2026-09-17 00:13 UTC

The queue remained healthy at 14 completed top-level stages.
`v2_350m_default_lora` advanced from 603 to 1240/3000. Its latest finite train
loss was 3.174 at step 1200 and evaluation loss was 3.134 at step 1100. Final
BPB remained pending; checkpoints 1100 and 1200 were present, with older
checkpoints rotated by the configured retention limit.

The H200 reported 100% utilization and 84,564 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Authenticated private-Bucket readback
still contained the exact active run, with its latest immutable object at
23:46:15 UTC; the durable local Trackio database remained actively updated. No
failed/red gate, publication change, or intervention appeared.

### 15-minute monitor tick — 2026-09-17 00:58 UTC

`v2_350m_default_lora` completed 3000/3000 with final train/evaluation losses
2.9541/2.9517 and BPB 3.3941→2.1579 (36.4% gain). Its final checkpoint is
`checkpoint-3000`, and Trackio finalized as `synced`.

The queue remained healthy at 14 completed top-level stages and advanced to
`configs/clm/350m/v2_default_full.yaml`, observed at step 39/3000 before its
first logged loss or checkpoint. The H200 reported 100% utilization and
76,244 MiB VRAM used; approximately 1.1 TiB remained free (62% disk usage).
Authenticated private-Bucket readback was fresh through 00:58:16 UTC and
contained the exact successor run. No failed/red gate, publication change, or
intervention appeared.

### Resumed monitor — 2026-09-17 02:39 UTC

The detached queue remained healthy at 14 completed top-level stages.
`v2_350m_default_full` completed by configured early stopping at 2500/3000
(patience 5), with final train/evaluation losses 2.3400/2.3403 and BPB
3.3436→1.6338 (51.1% gain). Its latest checkpoint is `checkpoint-2500`, and
Trackio finalized as `synced`.

The queue advanced to `configs/clm/350m/v2_ext4000_lora.yaml`, observed at
2462/3000. Its latest finite train loss was 2.020 at step 2400 and evaluation
loss was 2.017 at step 2300; checkpoints 2300 and 2400 were present. The H200
reported 100% utilization and 86,658 MiB VRAM used; approximately 1.1 TiB
remained free (62% disk usage). Authenticated private-Bucket readback was fresh
through 01:45:39 UTC and contained the exact active run. No failed/red gate or
publication change appeared, and no pipeline recovery was required.

### Resumed monitor and merge recovery — 2026-09-17 04:49 UTC

The remaining 350M CPT runs completed while foreground monitoring was
interrupted. `v2_350m_ext4000_lora` finished 3000/3000 with train/evaluation
losses 2.0064/2.0062 and BPB 3.7926→1.3622 (64.1% gain).
`v2_350m_ext4000_full` then completed by configured early stopping at 2500/3000
with losses 2.2518/2.2518 and BPB 3.7629→1.5667 (58.4% gain). Trackio finalized
both runs as `synced`, and `cpt_350m` became the fifteenth completed top-level
stage.

`merge_cpt_350m` failed its first attempt because the merge resolver searched
for the run ID as a directory name, while the validated adapter lives at
`runs/clm/350m/v2_ext4000_lora`. The resolver now falls back to matching the
canonical `run` ID in nested `journal.json` files. The regression passed locally
(23 tests plus Ruff) and remotely (13 tests plus Ruff). The detached queue was
resumed from checkpoint as PID 1080794; merge attempt 2 found the adapter,
loaded the model, and remained alive.

At the resumed observation the H200 held 10,135 MiB of 143,771 MiB VRAM with
0% instantaneous utilization during merge setup; approximately 1.1 TiB remained
free (63% disk usage). The private Bucket was fresh through 03:33:04 UTC with
the failed merge event; the resumed merge's durable local Trackio run was
active. No private publication had been created yet, and no red quality gate
was bypassed.

### Publication and SFT preparation recovery — 2026-09-17 05:02 UTC

Merge attempt 2 passed its BPB preservation gate. Full SFT mixture preparation
and private 350M CPT publication followed: `Tonic/LFM2.5-350M-Wolof-CPT` was
uploaded privately, read back, and recorded at revision
`d16bb3dbd1938a16fc0026aab29332bdb28e6a92`.

The first SFT launch exposed two preparation defects: parquet list columns were
materialized by pandas as array-like values, and prefix differencing was not
stable for the LFM chat template. This had caused the earlier preparation stage
to emit zero-row parquet files while returning success. Preparation now
normalizes array-like messages, uses the tokenizer's native assistant-token
mask, and fails closed on an empty input or output split. Fourteen local and
remote focused tests passed; a remote 1,000-row smoke for each split produced
1,000/1,000 rows with zero rejects.

The invalid generated parquet files were removed and the queue resumed without
touching completed model checkpoints or bypassing a quality gate. PID 1083472
was alive in `prepare_sft_65k` attempt 2, regenerating the full 338,408/6,906
train/validation mixture. The private Trackio Bucket advanced to 04:54:24 UTC
(207 fragments), including the synced private publication run. GPU was idle
during CPU preparation (8,071/143,771 MiB VRAM), with roughly 1.1 TiB disk free.

### 15-minute monitor tick — 2026-09-17 00:43 UTC

The queue remained healthy at 14 completed top-level stages.
`v2_350m_default_lora` advanced from 1930 to 2601/3000. Its latest finite train
loss was 2.948 at step 2600 and evaluation loss was 2.956 at step 2500. Final
BPB remained pending; checkpoints 2500 and 2600 were present under the
configured retention limit.

The H200 reported 100% utilization and 84,596 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Authenticated private-Bucket readback
still contained the exact active run, with its latest immutable object at
23:46:15 UTC; the durable local Trackio database remained actively updated. No
failed/red gate, publication change, or intervention appeared.

### 15-minute monitor tick — 2026-09-17 00:28 UTC

The queue remained healthy at 14 completed top-level stages.
`v2_350m_default_lora` advanced from 1240 to 1930/3000. Its latest finite train
loss was 3.005 at step 1900 and evaluation loss was 3.000 at step 1800. Final
BPB remained pending; checkpoints 1800 and 1900 were present under the
configured retention limit.

The H200 reported 100% utilization and 84,580 MiB VRAM used; approximately
1.1 TiB remained free (62% disk usage). Authenticated private-Bucket readback
still contained the exact active run, with its latest immutable object at
23:46:15 UTC; the durable local Trackio database remained actively updated. No
failed/red gate, publication change, or intervention appeared.
