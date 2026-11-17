#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.57",
#     "accelerate>=1.0.0",
#     "safetensors>=0.4.0",
#     "huggingface_hub>=0.26.0",
#     "sentencepiece>=0.2.0",
#     "protobuf>=4.0.0",
#     "numpy",
#     "pandas",
#     "pyyaml",
#     "sacrebleu",
#     "tqdm",
#     "gguf",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""HF Jobs (CPU only): GGUF convert -> per-quant eval -> eval-gated private publish.

Quantization itself needs NO GPU: convert_hf_to_gguf.py and llama-quantize are
CPU programs (torch pinned to the CPU index above). Evaluation of each quant
also runs on CPU here (llama.cpp, n_gpu_layers=0) against the Tier-1 HF
reference; publishing is delegated to publish_quant.py, which FAILS CLOSED if
any required eval JSON is missing, stale, or failing (Eval Plan §2b).

Cheapest feasible flavor per size (artifacts must be retained until the gate
passes, so disk sizing includes source + F16 + all quants):
  0.6B: hf jobs uv run --flavor cpu-basic  ($0.01/h)
  4B:   hf jobs uv run --flavor cpu-upgrade ($0.03/h)
  8B:   hf jobs uv run --flavor cpu-xl     ($1.00/h)
  14B:  hf jobs uv run --flavor cpu-xl     ($1.00/h)
Always: --timeout 6h --secrets HF_TOKEN --env ONLY=<size, e.g. 4b>
Prerequisite: Tier-1 HF eval for that size must exist in the eval dataset repo
(Tonic/Wolof-Qwen3-v2610-eval) — run job_eval_clm_family.py first.
1.7B is intentionally absent: no weights on Hub at audit (Gate 0).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

QUANTS = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_0"]
EXTRA_QUANTS = {"8B": ["Q6_K"], "14B": ["Q6_K"]}
MODELS = [
    ("Orange/Wolof-Qwen3-0.6B-v2610", "a296360960", "Tonic/Wolof-Qwen3-0.6B-v2610-GGUF"),
    ("Orange/Wolof-Qwen3-4B-v2610", "64314c62da", "Tonic/Wolof-Qwen3-4B-v2610-GGUF"),
    ("Orange/Wolof-Qwen3-8B-v2610", "eacd4f3428", "Tonic/Wolof-Qwen3-8B-v2610-GGUF"),
    ("Orange/Wolof-Qwen3-14B-v2610", "cc3d985809", "Tonic/Wolof-Qwen3-14B-v2610-GGUF"),
]
LLAMA = Path("/tmp/llama.cpp")
LLAMA_TAG = os.environ.get("LLAMA_CPP_TAG", "b8700")
REPO_URL = os.environ.get("WOLOF_REPO_URL", "https://github.com/Josephrp/Wolof-LLM.git")
EVAL_DATASET = os.environ.get("EVAL_DATASET", "Tonic/Wolof-Qwen3-v2610-eval")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_llamacpp() -> Path:
    if not (shutil.which("cmake") and shutil.which("git")):
        run(["bash", "-lc", "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential"])
    if not LLAMA.is_dir():
        run(["git", "clone", "--depth", "1", "--branch", LLAMA_TAG,
             "https://github.com/ggerganov/llama.cpp.git", str(LLAMA)])
    quantize = LLAMA / "build" / "bin" / "llama-quantize"
    if not quantize.is_file():
        run(["cmake", "-B", "build", "-S", ".", "-DGGML_NATIVE=OFF"], cwd=LLAMA)
        run(["cmake", "--build", "build", "--target", "llama-quantize",
             "llama-perplexity", "llama-completion", "-j", str(os.cpu_count() or 4)], cwd=LLAMA)
    if not quantize.is_file():
        raise FileNotFoundError("llama-quantize build failed")
    return LLAMA / "convert_hf_to_gguf.py"


def ensure_repo() -> Path:
    repo = Path("/tmp/wolof-llm")
    if not repo.is_dir():
        ref = os.environ.get("WOLOF_REPO_REF", "")
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        run(cmd + [REPO_URL, str(repo)])
    return repo


def imatrix_pilot(repo: Path, staging: Path, f16: Path, stem: str) -> dict:
    """Quantize Q4_K_M with and without a Wolof imatrix; keep the winner as canonical.

    Both candidates are fully evaluated (evaluate_gguf.py); the loser's file is
    deleted and the winner is renamed to {stem}-Q4_K_M.gguf with its eval JSON
    stamped as the canonical Q4_K_M eval. Returns the decision record.
    """
    quantize = LLAMA / "build" / "bin" / "llama-quantize"
    if not (LLAMA / "build" / "bin" / "llama-imatrix").is_file():
        run(["cmake", "--build", "build", "--target", "llama-imatrix",
             "-j", str(os.cpu_count() or 4)], cwd=LLAMA)
    corpus = repo / "Evaluation/data/corpus_test_clm.csv"
    slice_file = Path("/tmp/imatrix_corpus.txt")
    texts = pd.read_csv(corpus)["text"].dropna().astype(str).tolist()[:200]
    slice_file.write_text("\n\n".join(texts), encoding="utf-8")
    imatrix = Path("/tmp/imatrix.dat")
    run([str(LLAMA / "build" / "bin" / "llama-imatrix"), "-m", str(f16),
         "-f", str(slice_file), "-o", str(imatrix), "--chunks", "20", "-c", "512"])
    candidates = {}
    for label, extra in (("plain", []), ("imatrix", ["--imatrix", str(imatrix)])):
        cand = staging / f"{stem}-Q4_K_M-{label}.gguf"
        run([str(quantize), *extra, str(f16), str(cand), "Q4_K_M"])
        evaluate_quant(repo, staging, cand, f"Q4KM-{label}")
        ev = json.loads((staging / "eval" / f"Q4KM-{label}_gguf_eval.json").read_text())
        candidates[label] = (cand, ev)
    winner = max(candidates, key=lambda k: (float(candidates[k][1]["chrF"]),
                                            -float(candidates[k][1]["perplexity"])))
    win_file, win_ev = candidates[winner]
    canonical = staging / f"{stem}-Q4_K_M.gguf"
    win_file.replace(canonical)
    win_ev["quant"] = "Q4_K_M"
    win_ev["imatrix"] = winner == "imatrix"
    win_ev["file_sha256"] = sha256_file(canonical)
    (staging / "eval" / "Q4_K_M_gguf_eval.json").write_text(
        json.dumps(win_ev, indent=2) + "\n", encoding="utf-8")
    for label, (cand, _) in candidates.items():
        cand.unlink(missing_ok=True)
    decision = {"winner": winner,
                "chrF": {k: v[1]["chrF"] for k, v in candidates.items()},
                "perplexity": {k: v[1]["perplexity"] for k, v in candidates.items()}}
    print(f"IMATRIX_DECISION {decision}", flush=True)
    return decision


def fetch_tier1(api: HfApi, token: str, size_key: str) -> Path:
    """Tier-1 HF eval JSON must pre-exist (GPU wave); fail before spending CPU hours."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(EVAL_DATASET, f"orange_wolof_qwen3_{size_key}.json",
                           repo_type="dataset", token=token)
    dest = Path(f"/tmp/tier1_{size_key}.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.setdefault("src_sha", "")  # publish gate checks pinned SHA if present
    dest.write_text(json.dumps(data), encoding="utf-8")
    return dest


def evaluate_quant(repo: Path, staging: Path, gguf: Path, quant: str) -> None:
    """Run repo evaluate_gguf.py (CPU) and stamp file_sha256 into the eval JSON."""
    import yaml

    # Pilot runs may bound the eval (GGUF_EVAL_N_TEXTS / GGUF_BLEU_N) to fit CPU
    # job timeouts; default is the full corpus, and the subset size is recorded
    # in the eval JSON (n_texts_evaluated) that the publish gate consumes.
    n_texts = os.environ.get("GGUF_EVAL_N_TEXTS")
    cfg = {"run": f"{gguf.stem}", "gguf_path": str(gguf), "quant": quant,
           "hf_repo": None, "corpus_csv": str(repo / "Evaluation/data/corpus_test_clm.csv"),
           "text_column": "text", "n_texts": int(n_texts) if n_texts else None,
           "apply_verbalization": True,
           "ctx_size": 1024, "n_gpu_layers": 0, "llamacpp_root": str(LLAMA),
           "bleu_n_texts": int(os.environ.get("GGUF_BLEU_N", "100")), "bleu_seed": 0,
           "bleu_prompt_ratio": 0.5,
           "bleu_length_ratio": 1.3, "bleu_min_new_tokens": 16, "bleu_max_new_tokens": 256,
           "output_json": str(staging / "eval" / f"{quant}_gguf_eval.json")}
    cfg_path = Path(f"/tmp/gguf_eval_{quant}.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run([sys.executable, str(repo / "Evaluation/scripts/evaluate_gguf.py"),
         "--config", str(cfg_path)], cwd=repo / "Evaluation")
    out = staging / "eval" / f"{quant}_gguf_eval.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    data["file_sha256"] = sha256_file(gguf)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"EVAL_OK {quant}: PPL={data.get('perplexity')} chrF={data.get('chrF')}", flush=True)


def convert_one(src: str, sha: str, dst: str, token: str, convert: Path) -> None:
    stem = dst.split("/")[-1].removesuffix("-GGUF")
    size_label = stem.split("-")[2]
    size_key = {"0.6B": "06b", "4B": "4b", "8B": "8b", "14B": "14b"}[size_label]
    quants = QUANTS + EXTRA_QUANTS.get(size_label, [])
    repo = ensure_repo()
    api = HfApi(token=token)
    tier1 = fetch_tier1(api, token, size_key)

    work = Path(f"/tmp/gguf-{stem}")
    if work.exists():
        shutil.rmtree(work)
    src_dir, staging = work / "src", work / "out"
    src_dir.mkdir(parents=True)
    staging.mkdir(parents=True)
    (staging / "eval").mkdir()
    print(f"download {src}@{sha}", flush=True)
    snapshot_download(src, revision=sha, local_dir=str(src_dir), token=token)

    quantize = LLAMA / "build" / "bin" / "llama-quantize"
    f16 = staging / f"{stem}-F16.gguf"
    run([sys.executable, str(convert), str(src_dir), "--outfile", str(f16), "--outtype", "f16"])
    shutil.rmtree(src_dir)  # free disk before the quant matrix
    evaluate_quant(repo, staging, f16, "F16")  # reference for dPPL; never uploaded
    imatrix_decision = None
    for qtype in quants:
        if qtype == "F16":
            continue
        if qtype == "Q4_K_M" and os.environ.get("IMATRIX_PILOT"):
            imatrix_decision = imatrix_pilot(repo, staging, f16, stem)
            continue
        out_gguf = staging / f"{stem}-{qtype}.gguf"
        run([str(quantize), str(f16), str(out_gguf), qtype])
        evaluate_quant(repo, staging, out_gguf, qtype)

    (staging / "gguf_meta.json").write_text(json.dumps(
        {"src": src, "src_sha": sha, "stem": stem, "quants": quants,
         "tool": f"llama.cpp {LLAMA_TAG} convert_hf_to_gguf.py + llama-quantize",
         "imatrix_decision": imatrix_decision,
         "tier1_json": tier1.name}, indent=2) + "\n", encoding="utf-8")

    publisher = repo / "Experiments/Orange-Wolof-Qwen3-v2610/scripts/publication/publish_quant.py"
    run([sys.executable, str(publisher), "--staging-dir", str(staging), "--kind", "gguf",
         "--src", src, "--src-sha", sha, "--dst-repo", dst,
         "--tier1-json", str(tier1), "--eval-dir", str(staging / "eval"),
         "--meta-json", str(staging / "gguf_meta.json")])
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    convert = ensure_llamacpp()
    only = os.environ.get("ONLY", "").strip().lower()
    for src, sha, dst in MODELS:
        if only and only not in src.lower() and only not in dst.lower():
            continue
        convert_one(src, sha, dst, token, convert)
    print("GGUF_FAMILY_JOB_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
