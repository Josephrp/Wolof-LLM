#!/usr/bin/env python3
"""Full Qwen3.5 Wolof family ladder: regenerate full data, then train 08b→2b→4b→9b.

Clears smoke-tainted completed shared/data stages and size-specific train marks so
production runs use full NTP/SFT corpora and 3-epoch CPT/SFT (+ DPO).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, utc_now

STATE = ROOT / "results" / "full_pipeline" / "qwen35_family_state.json"
SIZES_DEFAULT = ["08b", "2b", "4b", "9b"]

# Shared stages that smoke may have truncated — must re-run for production.
SMOKE_TAINTED_SHARED = {
    "materialize",
    "ortho_filter",
    "anchor_prepare",
    "prefs",
    "tokenizer",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Qwen3.5 Wolof size ladder")
    parser.add_argument("--sizes", nargs="+", default=SIZES_DEFAULT)
    parser.add_argument(
        "--keep-smoke-marks",
        action="store_true",
        help="Do not clear completed keys (dangerous if smoke data is still on disk)",
    )
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument(
        "--from-size",
        default=None,
        help="Start ladder at this size (inclusive), skipping earlier sizes",
    )
    return parser


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def clear_smoke_marks(sizes: list[str]) -> dict:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if STATE.is_file():
        backup = STATE.with_name(f"qwen35_family_state.smoke_backup_{utc_now().replace(':', '')}.json")
        shutil.copy2(STATE, backup)
        print(f"backed up state -> {backup}", flush=True)
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {"completed": {}, "uploads": {}, "updated_at": None}

    completed = dict(state.get("completed") or {})
    removed: list[str] = []
    for key in list(completed):
        if key in SMOKE_TAINTED_SHARED:
            removed.append(key)
            completed.pop(key, None)
            continue
        for size in sizes:
            if key.endswith(f":{size}") or key == f"ext_init:{size}":
                removed.append(key)
                completed.pop(key, None)
                break
    state["completed"] = completed
    state["mode"] = "production_ladder"
    state["cleared_at"] = utc_now()
    state["cleared_keys"] = removed
    state["updated_at"] = utc_now()
    atomic_json_write(STATE, state)
    print(f"cleared {len(removed)} smoke/production marks: {removed}", flush=True)
    return state


def main() -> int:
    args = build_parser().parse_args()
    load_trackio_env()
    os.environ["QWEN_PRODUCTION"] = "1"
    os.environ.setdefault("QWEN_PIPELINE_RUN_ID", f"qwen35-ladder-{utc_now()[:10]}")

    sizes = list(args.sizes)
    if args.from_size:
        if args.from_size not in sizes:
            raise SystemExit(f"--from-size {args.from_size} not in {sizes}")
        sizes = sizes[sizes.index(args.from_size) :]

    if not args.keep_smoke_marks:
        clear_smoke_marks(sizes)

    print(
        "ladder sizes:",
        sizes,
        "trackio:",
        os.environ.get("TRACKIO_SPACE_ID"),
        flush=True,
    )

    for size in sizes:
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/publication/50_run_family.py",
            "--only-size",
            size,
            "--resume",
        ]
        if args.skip_dpo:
            cmd.append("--skip-dpo")
        if args.skip_publish:
            cmd.append("--skip-publish")
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"LADDER_SIZE_OK {size}", flush=True)

    print("FAMILY_LADDER_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
