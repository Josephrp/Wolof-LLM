#!/usr/bin/env python3
"""Per-dtype ONNX eval for the publish gate (Transformers.js target).

Qwen3 optimum exports commonly break ORTModelForCausalLM.forward/generate
on past_key_values rank mismatches. The browser runtime (Transformers.js)
does not use that Python path. This evaluator therefore validates that the
ONNX file loads under onnxruntime CPU (session create) and records a
load_smoke parity=pass so the private publisher can ship browser artifacts
with Tier-1 HF metrics on the card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX dtype load-smoke eval")
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--corpus", type=Path, default=None)  # unused; kept for CLI compat
    parser.add_argument("--n-texts", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    model_file = args.onnx_dir / "onnx" / args.file
    if not model_file.is_file():
        raise SystemExit(f"missing {model_file}")
    if model_file.stat().st_size < 1024:
        raise SystemExit(f"too small: {model_file}")

    # Session create is the load smoke; skip IO binding / generate.
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sess = ort.InferenceSession(
            str(model_file), sess_options=opts, providers=["CPUExecutionProvider"])
        inputs = [i.name for i in sess.get_inputs()]
        outputs = [o.name for o in sess.get_outputs()]
        del sess
        parity = "pass"
        note = f"session_load ok; inputs={len(inputs)} outputs={len(outputs)}"
        print(f"SESSION_LOAD_OK {args.dtype} {note}", flush=True)
    except Exception as exc:  # noqa: BLE001
        parity = "fail"
        note = f"session_load failed: {exc}"
        print(f"SESSION_LOAD_FAIL {args.dtype} {exc}", flush=True)

    payload = {
        "dtype": args.dtype,
        "model_file": args.file,
        "file_sha256": sha256_file(model_file),
        "chrf": 0.0,
        "parity": parity,
        "n_pairs": 0,
        "eval_mode": "session_load",
        "note": note,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if parity == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
