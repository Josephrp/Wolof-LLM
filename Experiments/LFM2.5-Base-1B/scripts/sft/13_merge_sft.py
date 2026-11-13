#!/usr/bin/env python3
"""Merge an SFT LoRA adapter and validate vocabulary, specials, and tying."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from pathlib import Path

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data import charger_splits_hub
from common.metrics import compute_bpb
from common.model import ecrire_tie_flag, lire_tie_flag
from common.tracking import TrackingRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge SFT adapter into its CPT base")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-eos", type=int, default=7)
    parser.add_argument("--dataset-id", default="galsenai/cleaned_data_for_clm")
    parser.add_argument("--n-texts", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=0.002)
    return parser


def run(args: argparse.Namespace) -> int:
    if torch is None or PeftModel is None:
        raise RuntimeError("train dependencies are missing")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    tracking = TrackingRun(
        size=output.name,
        stage="sft-merge",
        tokenizer_family="128k-ext" if args.expected_eos == 128900 else "65k-ext",
        method="merge",
        seed=42,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", output.name),
        config=vars(args),
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    # 128k-ext init may pad embedding rows above tokenizer length; unused rows are OK.
    embed_rows = int(base.get_input_embeddings().weight.shape[0])
    tok_len = len(tokenizer)
    if embed_rows < tok_len or int(base.config.vocab_size) < tok_len:
        raise ValueError(
            f"vocab mismatch: model={base.config.vocab_size}, embed_rows={embed_rows}, tokenizer={tok_len}"
        )
    if embed_rows != int(base.config.vocab_size):
        raise ValueError(
            f"embed/config vocab mismatch: config={base.config.vocab_size}, embed_rows={embed_rows}"
        )
    if tokenizer.eos_token_id != args.expected_eos:
        raise ValueError(f"eos mismatch: got {tokenizer.eos_token_id}, expected {args.expected_eos}")
    expected_tie = lire_tie_flag(base.config)
    adapter_model = PeftModel.from_pretrained(base, args.adapter).eval()
    token = os.environ.get("HF_TOKEN")
    _, eval_texts = charger_splits_hub(args.dataset_id, token=token)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    before = compute_bpb(
        adapter_model,
        tokenizer,
        eval_texts[: args.n_texts],
        device=device,
    )
    merged = adapter_model.merge_and_unload().eval()
    input_weight = merged.get_input_embeddings().weight
    output_module = merged.get_output_embeddings()
    actual_tie = output_module is not None and input_weight.data_ptr() == output_module.weight.data_ptr()
    # Always persist the post-merge reality. PEFT merge can untie embeddings even when
    # the base expected tied weights; forcing expected_tie=True would mis-label the artifact.
    ecrire_tie_flag(merged.config, actual_tie)
    after = compute_bpb(
        merged,
        tokenizer,
        eval_texts[: args.n_texts],
        device=device,
    )
    delta = abs(after["bpb"] - before["bpb"])
    if delta > args.tolerance:
        raise ValueError(
            f"SFT merge BPB delta {delta:.6f} exceeds tolerance {args.tolerance:.6f}"
        )
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "base": args.base,
        "adapter": args.adapter,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "tie_expected": expected_tie,
        "tie_after_merge": actual_tie,
        "bpb_before_merge": before["bpb"],
        "bpb_after_merge": after["bpb"],
        "bpb_delta": delta,
        "bpb_tolerance": args.tolerance,
    }
    (output / "sft_merge_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    tracking.log(
        {
            "bpb_before_merge": before["bpb"],
            "bpb_after_merge": after["bpb"],
            "merge_delta": delta,
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
