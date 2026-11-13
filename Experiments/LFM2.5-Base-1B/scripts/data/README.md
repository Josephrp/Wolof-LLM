# Data scripts

Order:

1. **audit** — `20_audit_datasets.py`
2. **mixture** — `21_build_sft_mixture.py`
3. **orthography split** — `24_split_orthography.py` (+ `25_submit_ortho_job.py` for HF Jobs)

Requires `HF_TOKEN` for private `galsenai/*` (read) and for `--push` / Jobs (write).

## Orthography partitions

Deterministic labels via `common/orthography.py` → Hub datasets under
`ORTHO_NAMESPACE` (default `Tonic`):

```bash
# rebuild self-contained Jobs script after editing the classifier
uv run python scripts/data/_build_ortho_job.py

# full remote job
uv run python scripts/data/25_submit_ortho_job.py --flavor cpu-xl --timeout 4h

# local smoke
uv run python scripts/data/24_split_orthography.py --smoke --out-dir results/ortho_split
```
