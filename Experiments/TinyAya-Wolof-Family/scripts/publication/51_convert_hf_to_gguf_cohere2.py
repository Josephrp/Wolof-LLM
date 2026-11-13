#!/usr/bin/env python3
"""convert_hf_to_gguf.py wrapper that skips untied Cohere2 lm_head.weight.

llama.cpp Cohere2 always aliases output ← token_embd (TENSOR_DUPLICATED), so an
untied HF lm_head cannot be represented and must be dropped at convert time.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: 51_convert_hf_to_gguf_cohere2.py <convert_hf_to_gguf.py> [args...]", file=sys.stderr)
        return 2
    convert_script = Path(sys.argv[1]).resolve()
    if not convert_script.is_file():
        print(f"missing convert script: {convert_script}", file=sys.stderr)
        return 2
    llama_root = convert_script.parent
    sys.path.insert(0, str(llama_root))
    sys.path.insert(1, str(llama_root / "gguf-py"))

    from conversion import command_r
    from conversion.base import logger

    original = command_r.Cohere2Model.modify_tensors

    def modify_tensors(self, data_torch, name, bid):
        if name == "lm_head.weight":
            logger.warning(
                "Skipping untied lm_head.weight for Cohere2 GGUF "
                "(llama.cpp ties output to token_embd)."
            )
            return iter(())
        return original(self, data_torch, name, bid)

    command_r.Cohere2Model.modify_tensors = modify_tensors
    sys.argv = [str(convert_script), *sys.argv[2:]]
    runpy.run_path(str(convert_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
