#!/usr/bin/env python3
"""Evaluate an SFT checkpoint without fabricating unavailable metrics."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    import sacrebleu
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sacrebleu = None
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import compute_bpb
from common.tracking import TrackingRun

WOLOF_PROMPTS = [
    "Waxal ma ci sa bopp.", "Lan mooy njariñu jang?", "Bind ab bataaxal bu gatt.",
    "Naka lañuy aar àdduna?", "Leeralal ma ndimbalug dëkk.", "Joxe ñetti xelal ci liggéey.",
    "Tëral ab waxtaan ci wér-gu-yaram.", "Lan la njàngat mi tekki?", "Wax léeb bu gatt.",
    "Naka la nit man a jàng Wolof?", "Joxe misaal bu leer.", "Tënkal xibaar bii.",
    "Defar ab laaj ci mbay.", "Wax ci cosaanu Senegaal.", "Naka lañuy jëfandikoo jumtukaay?",
    "Bind tontu bu am teggin.", "Leeralal wuute gi.", "Joxe waxtaanu jënd ak jaay.",
    "Lan mooy dëgg-dëgg?", "Waxal ci solo si kàddu am.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Wolof SFT validation and prompts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Prepared SFT directory")
    parser.add_argument("--output", default="results/sft")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--translation-chrf", action="store_true")
    parser.add_argument("--clm-parquet", help="Optional parquet with text for BPB continuity")
    return parser


def _row_loss(model, tokenizer, text: str) -> float:
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.no_grad():
        return float(model(**batch, labels=batch["input_ids"]).loss)


def run(args: argparse.Namespace) -> int:
    if torch is None or AutoTokenizer is None:
        raise RuntimeError("evaluation dependencies are missing")
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data
    frame = pd.read_parquet(data / "val.parquet").iloc[: args.max_rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    ).eval()
    losses = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        losses[str(row.get("task", "unknown"))].append(_row_loss(model, tokenizer, row["text"]))
    summaries = {
        task: {"n": len(values), "mean_loss": sum(values) / len(values), "ppl": math.exp(sum(values) / len(values))}
        for task, values in losses.items()
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    tracking = TrackingRun(
        size=Path(args.model).name,
        stage="sft-eval",
        tokenizer_family="model-artifact",
        method="evaluation",
        seed=42,
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", Path(args.model).name),
        config=vars(args),
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=output / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)
    generations = []
    for prompt in WOLOF_PROMPTS:
        messages = [
            {"role": "system", "content": "Tontul ci Wolof bu leer te gatt."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=128, do_sample=False)
        answer = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        generations.append({"prompt": prompt, "response": answer})
    with (output / "qualitative.jsonl").open("w", encoding="utf-8") as handle:
        for row in generations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {"model": args.model, "per_task": summaries, "qualitative_count": len(generations)}
    if args.translation_chrf:
        if sacrebleu is None:
            raise RuntimeError("sacrebleu is required for --translation-chrf")
        translation = frame[frame["task"].astype(str) == "translation"]
        if {"reference", "prediction"} <= set(translation.columns):
            result["translation_chrf"] = sacrebleu.corpus_chrf(
                translation["prediction"].tolist(), [translation["reference"].tolist()]
            ).score
        else:
            result["translation_chrf"] = None
            result["translation_chrf_note"] = "not computed: reference/prediction columns absent"
    if args.clm_parquet:
        texts = pd.read_parquet(args.clm_parquet)["text"].dropna().astype(str).tolist()
        result["clm_continuity"] = compute_bpb(model, tokenizer, texts[: args.max_rows])
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    flat_metrics = {
        f"task/{task}/mean_loss": values["mean_loss"]
        for task, values in summaries.items()
    }
    flat_metrics.update(
        {
            "qualitative_count": len(generations),
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0
            ),
            "gate_status": 1,
        }
    )
    if "clm_continuity" in result:
        flat_metrics["bpb"] = result["clm_continuity"]["bpb"]
    tracking.log(flat_metrics)
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
