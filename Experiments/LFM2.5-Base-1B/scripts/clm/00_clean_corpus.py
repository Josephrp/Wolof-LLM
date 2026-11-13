#!/usr/bin/env python3
"""REGEN ONLY — prefer Hub dataset galsenai/cleaned_data_for_clm.

This script is intentionally a stub. Full cleaning logic lives in the SmolLM3
template and should only be re-ported if the cleaning policy changes:

  Experiments/SmolLM3-3B-Base/scripts/00_clean_corpus.py

Default CPT data path for LFM2.5 Wolof:
  load_dataset("galsenai/cleaned_data_for_clm")

Pass --force-regen to acknowledge you intend to regenerate (still not
implemented here — exits with guidance).
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "REGEN ONLY stub. Prefer Hub galsenai/cleaned_data_for_clm. "
            "See Experiments/SmolLM3-3B-Base/scripts/00_clean_corpus.py"
        )
    )
    ap.add_argument(
        "--force-regen",
        action="store_true",
        help="Acknowledge regen intent (full port not yet available)",
    )
    args = ap.parse_args()

    if not args.force_regen:
        print(
            "00_clean_corpus.py is REGEN ONLY.\n"
            "Prefer Hub: galsenai/cleaned_data_for_clm\n"
            "Template: Experiments/SmolLM3-3B-Base/scripts/00_clean_corpus.py\n"
            "Re-run with --force-regen only if you intend to regenerate "
            "(full port not implemented yet).",
            file=sys.stderr,
        )
        return 2

    print(
        "--force-regen set, but full clean_corpus port is not implemented yet.\n"
        "Port from Experiments/SmolLM3-3B-Base/scripts/00_clean_corpus.py "
        "or keep using the Hub cleaned dataset.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
