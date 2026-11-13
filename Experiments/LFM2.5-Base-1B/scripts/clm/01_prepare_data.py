#!/usr/bin/env python3
"""REGEN ONLY — prefer Hub dataset galsenai/cleaned_data_for_clm.

This script is intentionally a stub. Full local parquet prepare logic lives in:

  Experiments/SmolLM3-3B-Base/scripts/01_prepare_data.py

For LFM2.5 Wolof CPT, load train/test from the Hub via common.data.charger_splits_hub
instead of regenerating wolof_bpe_train_v1.parquet locally.

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
            "See Experiments/SmolLM3-3B-Base/scripts/01_prepare_data.py"
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
            "01_prepare_data.py is REGEN ONLY.\n"
            "Prefer Hub: galsenai/cleaned_data_for_clm via common.data.charger_splits_hub\n"
            "Template: Experiments/SmolLM3-3B-Base/scripts/01_prepare_data.py\n"
            "Re-run with --force-regen only if you intend to regenerate "
            "(full port not implemented yet).",
            file=sys.stderr,
        )
        return 2

    print(
        "--force-regen set, but full prepare_data port is not implemented yet.\n"
        "Port from Experiments/SmolLM3-3B-Base/scripts/01_prepare_data.py "
        "or keep using the Hub cleaned dataset.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
