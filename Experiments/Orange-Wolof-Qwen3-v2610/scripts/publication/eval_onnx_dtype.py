#!/usr/bin/env python3
"""Per-dtype ONNX eval: chrF continuation subset + generation parity flag.

Loads one ONNX dtype via optimum-onnxruntime (CPU), generates greedy
continuations on the same verbalized corpus protocol as evaluate_clm.py
(prompt_ratio 0.5, seed 0), and writes {dtype}_onnx_eval.json with the model
file's SHA-256 for the publish gate in publish_quant.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "Evaluation"))
sys.path.insert(0, str(REPO / "Evaluation" / "scripts"))

from verbalizer import verbalize_text  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one ONNX dtype (chrF + parity)")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="repo root containing onnx/")
    parser.add_argument("--file", required=True, help="e.g. model_q4f16.onnx")
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--corpus", type=Path, default=REPO / "Evaluation" / "data" / "corpus_test_clm.csv")
    parser.add_argument("--n-texts", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd
    import sacrebleu
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    model_file = args.onnx_dir / "onnx" / args.file
    if not model_file.is_file():
        raise SystemExit(f"missing {model_file}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.onnx_dir))
    model = ORTModelForCausalLM.from_pretrained(str(args.onnx_dir), file_name=f"onnx/{args.file}")

    texts = [verbalize_text(t) for t in pd.read_csv(args.corpus)["text"].dropna().astype(str).tolist()]
    eligible = [t for t in texts if len(str(t).split()) >= 4]
    if 0 < args.n_texts < len(eligible):
        eligible = random.Random(0).sample(eligible, args.n_texts)
    hyps, refs = [], []
    parity_ok = True
    for text in eligible:
        words = text.split()
        split = max(1, int(len(words) * 0.5))
        prompt, ref = " ".join(words[:split]), " ".join(words[split:])
        if not ref:
            continue
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        except Exception:  # noqa: BLE001
            parity_ok = False
            break
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        hyps.append(gen)
        refs.append(ref)
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score if hyps else 0.0
    payload = {"dtype": args.dtype, "model_file": args.file,
               "file_sha256": sha256_file(model_file), "chrf": round(float(chrf), 2),
               "parity": "pass" if (parity_ok and hyps) else "fail", "n_pairs": len(hyps)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["parity"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
