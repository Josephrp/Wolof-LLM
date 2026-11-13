#!/usr/bin/env python3
"""Submit procedural orthography split as an HF Jobs UV job.

Requires a write-capable HF token in HF_TOKEN (read on private galsenai/*).

  set HF_TOKEN=hf_...
  uv run python scripts/data/25_submit_ortho_job.py
  uv run python scripts/data/25_submit_ortho_job.py --smoke
  uv run python scripts/data/25_submit_ortho_job.py --flavor cpu-xl --timeout 4h
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
JOB_SCRIPT = Path(__file__).resolve().parent / "24_split_orthography_job.py"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flavor", default="cpu-xl")
    p.add_argument("--timeout", default="4h")
    p.add_argument("--namespace", default=os.environ.get("ORTHO_NAMESPACE", "Tonic"))
    p.add_argument("--smoke", action="store_true", help="Cap at 500 rows/split")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--public", action="store_true")
    p.add_argument("--skip-clm", action="store_true")
    p.add_argument("--skip-raw", action="store_true")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument(
        "--detach",
        action="store_true",
        help="Do not stream logs / wait (default: wait)",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: set HF_TOKEN (write) before submitting the remote job.",
            file=sys.stderr,
        )
        return 2
    if not JOB_SCRIPT.is_file():
        print(f"ERROR: missing {JOB_SCRIPT}", file=sys.stderr)
        return 2

    max_rows = args.max_rows
    if args.smoke and max_rows is None:
        max_rows = 500

    env = {
        "ORTHO_NAMESPACE": args.namespace,
    }
    if max_rows is not None:
        env["ORTHO_MAX_ROWS"] = str(max_rows)
    if args.public:
        env["ORTHO_PUBLIC"] = "1"
    if args.skip_clm:
        env["ORTHO_SKIP_CLM"] = "1"
    if args.skip_raw:
        env["ORTHO_SKIP_RAW"] = "1"
    if args.skip_sft:
        env["ORTHO_SKIP_SFT"] = "1"

    api = HfApi(token=token)
    me = api.whoami()
    print(f"Submitting as {me.get('name')} · flavor={args.flavor} · timeout={args.timeout}")
    print(f"Script: {JOB_SCRIPT}")
    print(f"Env: {env}")

    job = api.run_uv_job(
        str(JOB_SCRIPT),
        flavor=args.flavor,
        timeout=args.timeout,
        env=env,
        secrets={"HF_TOKEN": token},
        name="wolof-ortho-split",
        labels={"project": "lfm25-wolof", "task": "ortho-split"},
    )
    job_id = getattr(job, "id", None) or getattr(job, "job_id", None) or str(job)
    print(f"Job submitted: {job_id}")
    print(f"Inspect: https://huggingface.co/jobs/{me.get('name')}/{job_id}")

    if args.detach:
        return 0

    print("--- logs ---")
    try:
        for line in api.fetch_job_logs(job_id=str(job_id)):
            print(line, end="" if str(line).endswith("\n") else "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"(log stream ended/failed: {exc})")
    info = api.inspect_job(job_id=str(job_id))
    status = getattr(info, "status", info)
    print(f"Final status: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
