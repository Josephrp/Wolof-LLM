"""Build self-contained HF Jobs script from common/orthography.py + job main."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORTHO = ROOT / "common" / "orthography.py"
MAIN = Path(__file__).resolve().parent / "_ortho_job_main.py"
OUT = Path(__file__).resolve().parent / "24_split_orthography_job.py"

HEADER = '''#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.26.0",
#   "pyarrow",
# ]
# ///
"""HF Jobs entrypoint: procedural Wolof orthography split + Hub upload.

Submit via:
  uv run python scripts/data/25_submit_ortho_job.py
"""

'''


def main() -> None:
    ortho = ORTHO.read_text(encoding="utf-8")
    idx = ortho.find("from __future__")
    if idx < 0:
        raise RuntimeError("orthography.py missing future import")
    ortho_body = ortho[idx:]
    main_body = MAIN.read_text(encoding="utf-8")
    OUT.write_text(HEADER + ortho_body + "\n\n" + main_body, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
