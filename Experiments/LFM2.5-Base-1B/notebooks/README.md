# Notebooks — exploratory only

Training, audit, and eval **scripts** under `scripts/` are the source of truth.
Notebooks here are for inspection / demos, not for gated ladder runs.

```bash
cd Experiments/LFM2.5-Base-1B
uv sync --group notebook --group data
uv run jupyter lab
# or: uv run jupyter notebook
```
