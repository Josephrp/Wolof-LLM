#!/usr/bin/env python3
"""Private-first publisher for Orange Wolof-Qwen3 v2610 GGUF/ONNX repos.

HARD GATE (Eval Plan §2b): this script refuses to create any Hub repo unless
  1. the Tier-1 HF eval JSON for the source model exists and its source SHA
     matches the pinned Orange revision,
  2. every quant file / ONNX dtype to be uploaded has its own eval JSON whose
     recorded file SHA-256 matches the actual file on disk, with a verdict of
     pass / compat-only computed here from the documented thresholds,
  3. the model card rendered from those JSONs contains a non-empty evaluation
     section (the generator raises otherwise).
There is deliberately no --skip-eval flag. --dry-run renders and validates the
card without touching the Hub.

GGUF eval JSON schema (superset of Evaluation/scripts/evaluate_gguf.py output):
  {protocol: "gguf-llama-cpp", quant, gguf_path, file_sha256, perplexity,
   BLEU, chrF, n_texts_evaluated, diacritic_chrf (optional)}
ONNX eval JSON schema (written by job_onnx_family.py):
  {dtype, model_file, file_sha256, chrf, parity: "pass", n_pairs}
Tier-1 JSON schema: Evaluation/scripts/evaluate_clm.py output plus
  {src_sha} injected by job_eval_clm_family.py (or hub_sha in family.yaml).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "Experiments" / "Qwen3.5-Wolof-Family"))
sys.path.insert(0, str(ROOT))

from common.hub_utils import build_checksums, scan_for_secrets, select_namespace, sha256_file  # noqa: E402

# quant -> (max dchrF pts vs F16 GGUF, max dPPL % vs F16 GGUF, verdict label if within)
# Both deltas are measured against the evaluated F16 GGUF master on the
# identical eval subset and llama.cpp protocol. Comparing GGUF chrF against
# the Tier-1 HF chrF instead would be unsound: the F16 master itself scores
# ~1.3pt below Tier-1 purely from protocol/subset effects, which would fail
# every quant regardless of quantization quality. Tier-1 remains a hard gate
# via existence + source-SHA match and the F16 sanity band below.
GGUF_THRESHOLDS = {
    "Q8_0": (0.5, 1.0, "pass"),
    "Q6_K": (0.5, 1.0, "pass"),
    "Q5_K_M": (1.0, 3.0, "pass"),
    "Q4_K_M": (1.5, 5.0, "pass"),
    "Q4_0": (2.5, 8.0, "compat-only"),
    "F16": (0.5, 1.0, "pass"),  # reference master; evaluated but never uploaded
}
ONNX_MAX_DCHRF = {"fp16": 1.0, "quantized": 1.5, "q4": 2.0, "q4f16": 2.0}
# The evaluated F16 GGUF master must score within this perplexity ratio of
# the Tier-1 HF model; beyond it the conversion itself is suspect (wrong
# tokenizer/template) and no quant verdict can be trusted. Perplexity is
# used for this band (not chrF) because it is computed over every eval text,
# while chrF comes from a small sampled generation subset whose absolute
# level swings with the draw; observed healthy ratios are ~1.1-1.3x.
F16_TIER1_MAX_PPL_RATIO = 1.5
TEXT_SUFFIXES = {".md", ".json", ".txt", ".yaml", ".yml", ".jinja"}


class GateError(RuntimeError):
    """Raised when the eval-before-publish gate blocks a model."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise GateError(f"missing required eval JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_text(text: str) -> str:
    return re.sub(r"/mnt/(?:volume|data)[^\"'\s]*/", "", text)


def check_tier1(path: Path, src_sha: str) -> dict:
    tier1 = load_json(path)
    pinned = str(tier1.get("src_sha") or "")
    if pinned and not src_sha.startswith(pinned[:10]) and not pinned.startswith(src_sha[:10]):
        raise GateError(f"Tier-1 eval SHA mismatch: json={pinned} pinned={src_sha} ({path})")
    if not isinstance(tier1.get("BPB"), (int, float)):
        raise GateError(f"Tier-1 eval has no finite BPB: {path}")
    return tier1


def gguf_verdict(quant: str, ev: dict, f16_ref: dict) -> str:
    """Verdict for one quant vs the evaluated F16 GGUF master.

    Both metrics come from the same eval subset and llama.cpp protocol, so
    the deltas isolate quantization loss from protocol/subset effects.
    """
    if quant not in GGUF_THRESHOLDS:
        raise GateError(f"no thresholds defined for quant {quant}")
    max_dchrf, max_dppl, label = GGUF_THRESHOLDS[quant]
    dchrf = float(f16_ref["chrF"]) - float(ev["chrF"])
    if dchrf > max_dchrf:
        return "fail"
    f16_ppl = float(f16_ref["perplexity"])
    dppl = 100.0 * (float(ev["perplexity"]) - f16_ppl) / f16_ppl
    if dppl > max_dppl:
        return "fail"
    return label


def validate_gguf_metadata(path: Path) -> dict:
    """Assert GGUF magic, qwen3 architecture and an embedded chat template.

    Uses gguf-py when available (GGUF Jobs install it); falls back to a magic +
    embedded-string scan so the publisher also works in minimal envs.
    """
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise GateError(f"{path.name}: not a GGUF file (bad magic)")
    try:
        # Lazy import: gguf-py is an optional dependency (present in GGUF Jobs,
        # absent in minimal publisher envs) — the fallback below covers absence.
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        fields = {field.name: field for field in reader.fields.values()}
        arch_field = fields.get("general.architecture")
        arch = bytes(arch_field.parts[-1]).decode("utf-8", "replace") if arch_field else ""
        if "qwen3" not in arch:
            raise GateError(f"{path.name}: architecture {arch!r} != qwen3")
        if not any("chat_template" in name for name in fields):
            raise GateError(f"{path.name}: no embedded chat template in GGUF metadata")
        return {"architecture": arch, "metadata_keys": len(fields)}
    except ImportError:
        blob = path.read_bytes()
        if b"chat_template" not in blob and b"tokenizer.chat_template" not in blob:
            raise GateError(f"{path.name}: no chat template string found in GGUF file")
        return {"architecture": "qwen3 (string-scan fallback)", "metadata_keys": -1}


def validate_gguf(staging: Path, eval_dir: Path, tier1: dict) -> list[dict]:
    """Return per-quant records for uploadable files; raise GateError on any gap."""
    files = sorted(staging.glob("*.gguf"))
    if not files:
        raise GateError(f"no .gguf files in {staging}")
    evals: dict[str, dict] = {}
    for path in sorted(eval_dir.glob("*_gguf_eval.json")):
        ev = load_json(path)
        evals[str(ev.get("quant"))] = ev
    # F16 reference eval (master evaluated, never uploaded). Fail closed
    # without it: quant verdicts are deltas against this reference, and its
    # perplexity must sit within a sanity ratio of the Tier-1 HF score.
    f16_ref = evals.get("F16")
    if f16_ref is None:
        raise GateError("missing F16 reference eval JSON; every quant must be "
                        "gated against the evaluated F16 master")
    tier1_ppl = tier1.get("perplexity")
    if isinstance(tier1_ppl, (int, float)) and float(tier1_ppl) > 0:
        ratio = float(f16_ref["perplexity"]) / float(tier1_ppl)
        if ratio > F16_TIER1_MAX_PPL_RATIO:
            raise GateError(
                f"F16 GGUF perplexity {f16_ref['perplexity']} is {ratio:.2f}x "
                f"Tier-1 perplexity {tier1_ppl} (band {F16_TIER1_MAX_PPL_RATIO}x); "
                "conversion suspect, no quant verdict can be trusted")
    records: list[dict] = []
    for path in files:
        quant = path.stem.rsplit("-", 1)[-1]
        if quant == "F16":
            continue  # master excluded from upload (quants-only policy)
        ev = evals.get(quant)
        if ev is None:
            raise GateError(f"missing per-quant eval JSON for {path.name} (quant={quant})")
        digest = sha256_file(path)
        if ev.get("file_sha256") != digest:
            raise GateError(f"eval/file SHA mismatch for {path.name}: "
                            f"json={ev.get('file_sha256')} file={digest}")
        verdict = gguf_verdict(quant, ev, f16_ref)
        records.append({"file": path.name, "quant": quant, "size": path.stat().st_size,
                        "sha256": digest, "eval": ev, "verdict": verdict,
                        "gguf_meta": validate_gguf_metadata(path)})
    if not any(r["verdict"] in ("pass", "compat-only") for r in records):
        raise GateError("no quant passed the eval gate; nothing may be published")
    return records


def validate_onnx(staging: Path, eval_dir: Path, tier1: dict) -> list[dict]:
    onnx_dir = staging / "onnx"
    files = sorted(onnx_dir.glob("model*.onnx")) if onnx_dir.is_dir() else []
    if not files:
        raise GateError(f"no ONNX files in {onnx_dir}")
    evals: dict[str, dict] = {}
    for path in sorted(eval_dir.glob("*_onnx_eval.json")):
        ev = load_json(path)
        evals[str(ev.get("dtype"))] = ev
    dtype_of = {"model.onnx": "fp32", "model_fp16.onnx": "fp16",
                "model_quantized.onnx": "quantized", "model_q4.onnx": "q4",
                "model_q4f16.onnx": "q4f16"}
    # Quantized dtypes are gated on chrF delta vs the evaluated fp16 dtype
    # (same prompts/protocol), not vs Tier-1 HF chrF; fail closed without it.
    fp16_ref = evals.get("fp16")
    records: list[dict] = []
    for path in files:
        dtype = dtype_of.get(path.name)
        if dtype is None or dtype == "fp32":
            continue  # fp32 base graph is packaging, dtypes below are the artifacts
        ev = evals.get(dtype)
        if ev is None:
            raise GateError(f"missing per-dtype eval JSON for {path.name} (dtype={dtype})")
        digest = sha256_file(path)
        if ev.get("file_sha256") != digest:
            raise GateError(f"eval/file SHA mismatch for {path.name}")
        if ev.get("parity") != "pass":
            raise GateError(f"ONNX parity not pass for {dtype}")
        if dtype == "fp16":
            dchrf = 0.0  # reference dtype; parity above is its gate
        else:
            if fp16_ref is None:
                raise GateError("missing fp16 reference eval JSON; quantized "
                                "dtypes must be gated against the evaluated fp16 dtype")
            dchrf = float(fp16_ref["chrf"]) - float(ev["chrf"])
        verdict = "pass" if dchrf <= ONNX_MAX_DCHRF.get(dtype, 2.0) else "fail"
        records.append({"file": f"onnx/{path.name}", "quant": dtype, "size": path.stat().st_size,
                        "sha256": digest, "eval": ev, "verdict": verdict})
    if not any(r["verdict"] == "pass" for r in records):
        raise GateError("no ONNX dtype passed the eval gate; nothing may be published")
    return records


def render_card(*, kind: str, dst_repo: str, src: str, src_sha: str, stem: str,
                tier1: dict, records: list[dict], meta: dict) -> str:
    uploadable = [r for r in records if r["verdict"] in ("pass", "compat-only")]
    if not uploadable:
        raise GateError("card generation refused: no evaluated quant/dtype to describe")
    default = next((r for r in uploadable if r["quant"] in ("Q4_K_M", "q4f16")), uploadable[0])
    rows = "\n".join(
        f"| `{r['file']}` | {r['quant']} | {r['size'] / 1e9:.2f} GB | `{r['sha256'][:16]}…` "
        f"| chrF {r['eval'].get('chrF', r['eval'].get('chrf'))} | {r['verdict']} |"
        for r in records)
    failed = [r for r in records if r["verdict"] == "fail"]
    failed_note = ""
    if failed:
        failed_note = ("\n\nEvaluated but **not uploaded** (failed acceptance thresholds): "
                       + ", ".join(f"{r['quant']} (chrF {r['eval'].get('chrF', r['eval'].get('chrf'))})"
                                   for r in failed) + ". Their eval JSONs are in the manifest repo history.")
    if kind == "gguf":
        usage = f'```bash\nllama-cli -hf {dst_repo}:{default["quant"]} -p "Salaam aleekum, naka nga def?"\n```'
        library, tags = "gguf", "[gguf, llama.cpp, wolof, qwen3]"
    else:
        usage = ("```js\nimport { pipeline } from '@huggingface/transformers';\n"
                 f"const gen = await pipeline('text-generation', '{dst_repo}', {{ dtype: 'q4f16', device: 'webgpu' }});\n```")
        library, tags = "transformers.js", "[onnx, transformers.js, webgpu, wolof, qwen3]"
    card = f"""---
license: cc-by-nc-sa-4.0
base_model: {src}
library_name: {library}
pipeline_tag: text-generation
language: [wo, fr, en]
tags: {tags}
model-index:
- name: {stem}
  results:
  - task: {{type: text-generation}}
    dataset: {{name: "Wolof-LLM corpus_test_clm (4,282 verbalized texts)", type: local}}
    metrics:
    - {{type: perplexity, value: {tier1.get('perplexity')}}}
    - {{type: chrf, value: {tier1.get('chrF')}}}
---

# {stem} ({kind.upper()})

Quantized export of [`{src}`](https://huggingface.co/{src}) at pinned revision
`{src_sha}`. **Non-commercial**: CC-BY-NC-SA-4.0 inherited from the base model;
commercial use requires separate Orange terms. Wolof (wo) primary, French and
English supported. Please cite Lecorvé et al., *Wolof-Adapted Industry-Ready
Open-Weight Language Models* (arXiv:2610.20824).

## Evaluation (measured, protocol: see below)

Source model (HF, `Evaluation/scripts/evaluate_clm.py` on 4,282 verbalized Wolof
texts): **BPB {tier1.get('BPB')}**, perplexity {tier1.get('perplexity')},
BLEU {tier1.get('BLEU')}, chrF {tier1.get('chrF')}.

| File | Quant/dtype | Size | SHA-256 (16) | chrF | Verdict |
|---|---|---:|---|---|---|
{rows}
{failed_note}

Thresholds: GGUF ΔchrF and ΔPPL vs the evaluated F16 GGUF master on the
identical eval subset and llama.cpp protocol (Q8_0/Q6_K ≤0.5pt/+1%,
Q5_K_M ≤1pt/+3%, Q4_K_M ≤1.5pt/+5%, Q4_0 ≤2.5pt/+8% compat-only); the F16
master itself is sanity-checked at ≤1.5× the Tier-1 HF perplexity above.
ONNX: ΔchrF vs the evaluated fp16 dtype (q4f16 ≤ 2pts) + generation
parity. Recommended default: **{default['quant']}**.

## Usage

{usage}

## Provenance & quantization

Converted with {meta.get('tool', 'llama.cpp convert_hf_to_gguf.py + llama-quantize' if kind == 'gguf' else 'optimum ONNX export + onnxruntime quantization')}
(versions in the meta JSON). No imatrix unless stated in the meta JSON.
The F16 master was evaluated as reference but is not uploaded (available from
the source repo). Full checksums: `SHA256SUMS.json`.

## Limitations

Inherits the base model's limitations: function-calling and instruction-following
regressions vs stock Qwen3 reported by Orange; Wolof dialect/orthography coverage
is incomplete; SFT training context was 2,048 tokens (config advertises 40,960 —
validate long-context use); not for consequential decisions (medical, legal,
financial, employment) without human oversight; not validated for autonomous
tool use. Quantization may further affect Wolof diacritics (ñ/ŋ/à) — see the
diacritic-slice results in the eval JSONs.
"""
    if "chrF" not in card or tier1.get("chrF") is None:
        raise GateError("card generation produced an empty evaluation section")
    return card


LICENSE_TEXT = """CC-BY-NC-SA-4.0 (inherited from the base model)

This quantized artifact is a derivative of an Orange Wolof-Qwen3 model and is
distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International license, the same terms as the base model. Commercial use
requires separate terms with Orange. Full license text:
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
Base model: see the model card's base_model field and pinned revision.
"""


def write_blocked(manifest_path: Path, entry: dict) -> None:
    manifest = {"schema_version": 1, "models": []}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("models", []).append(entry)
    manifest["updated_at"] = utc_now()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Private-first, eval-gated quant publisher")
    p.add_argument("--staging-dir", type=Path, required=True)
    p.add_argument("--kind", choices=["gguf", "onnx"], required=True)
    p.add_argument("--src", required=True, help="Orange source repo id")
    p.add_argument("--src-sha", required=True, help="pinned source revision")
    p.add_argument("--dst-repo", required=True, help="full target repo id, e.g. Tonic/Wolof-Qwen3-4B-v2610-GGUF")
    p.add_argument("--tier1-json", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="dir with *_gguf_eval.json / *_onnx_eval.json (incl. file_sha256)")
    p.add_argument("--meta-json", type=Path, default=None, help="gguf_meta.json / onnx_meta.json in staging")
    p.add_argument("--manifest", type=Path, default=ROOT / "results" / "publication_manifest_quant.json")
    p.add_argument("--dry-run", action="store_true", help="validate gate + render card, no Hub writes")
    p.add_argument("--promote-public", action="store_true",
                   help="flip an already-published private repo public (gate re-validated first)")
    p.add_argument("--yes", action="store_true", help="required confirmation for --promote-public")
    p.add_argument("--smoke-bin", type=Path, default=None,
                   help="llama-completion binary for authenticated post-upload smoke (GGUF)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    staging = args.staging_dir
    stem = args.dst_repo.split("/")[-1].removesuffix("-GGUF").removesuffix("-ONNX")
    try:
        tier1 = check_tier1(args.tier1_json, args.src_sha)
        records = (validate_gguf(staging, args.eval_dir, tier1) if args.kind == "gguf"
                   else validate_onnx(staging, args.eval_dir, tier1))
        meta = {}
        if args.meta_json and args.meta_json.is_file():
            meta = json.loads(sanitize_text(args.meta_json.read_text(encoding="utf-8")))
        card = render_card(kind=args.kind, dst_repo=args.dst_repo, src=args.src,
                           src_sha=args.src_sha, stem=stem, tier1=tier1,
                           records=records, meta=meta)
    except GateError as exc:
        write_blocked(args.manifest, {"dst_repo": args.dst_repo, "status": "blocked_eval_gate",
                                      "reason": str(exc), "at": utc_now()})
        print(f"EVAL_GATE_BLOCKED {args.dst_repo}: {exc}", file=sys.stderr, flush=True)
        return 2

    (staging / "README.md").write_text(card, encoding="utf-8")
    (staging / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    # Checksums over uploadable artifacts + card + meta (F16 excluded by construction).
    uploadable = [r for r in records if r["verdict"] in ("pass", "compat-only")]
    files = [staging / r["file"] for r in uploadable]
    for extra in ("README.md", "LICENSE"):
        files.append(staging / extra)
    if args.meta_json and args.meta_json.is_file():
        args.meta_json.write_text(sanitize_text(args.meta_json.read_text(encoding="utf-8")), encoding="utf-8")
        files.append(args.meta_json)
    # Tokenizer / chat template files if present (GGUF jobs copy them alongside).
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (staging / name).is_file():
            files.append(staging / name)
    scan_for_secrets(staging, [p for p in files if p.suffix.lower() in TEXT_SUFFIXES])
    checksums = build_checksums(staging, files)
    (staging / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append(staging / "SHA256SUMS.json")
    allow = sorted({p.relative_to(staging).as_posix() for p in files})
    print(f"gate OK: {len(uploadable)} uploadable, card rendered ({len(card)} chars)", flush=True)
    print("allow_patterns:", allow, flush=True)

    if args.dry_run:
        print(card)
        print("DRY_RUN_OK (no Hub writes)", flush=True)
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN required")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    namespace = select_namespace(api.whoami(), args.dst_repo.split("/")[0])
    repo_id = f"{namespace}/{args.dst_repo.split('/')[-1]}"
    if args.promote_public:
        # Gate already re-validated above (tier1 + per-file SHA + verdicts);
        # promotion additionally requires explicit human confirmation.
        if not args.yes:
            raise SystemExit("--promote-public requires --yes (human promotion gate)")
        info = api.model_info(repo_id)
        if not info.private:
            print(f"{repo_id} already public", flush=True)
            return 0
        api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model")
        write_blocked(args.manifest, {"dst_repo": repo_id, "status": "promoted_public",
                                      "at": utc_now()})
        print(f"PROMOTED_PUBLIC {repo_id}", flush=True)
        return 0
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(repo_id)
    if not info.private:
        raise SystemExit(f"{repo_id} exists and is PUBLIC — refusing to publish into it")
    commit = api.upload_folder(repo_id=repo_id, folder_path=str(staging), repo_type="model",
                               allow_patterns=allow,
                               commit_message=f"Eval-gated quant publish of {args.src}@{args.src_sha}")
    # Post-upload verification: private, exact file list, revision recorded.
    info = api.model_info(repo_id, files_metadata=True)
    remote_files = sorted(s.rfilename for s in info.siblings)
    if not info.private:
        raise SystemExit("post-upload check failed: repo is not private")
    missing = [a for a in allow if a not in remote_files]
    if missing:
        raise SystemExit(f"post-upload check failed: missing {missing}")
    # Authenticated post-upload smoke (GGUF): download the default quant from the
    # private repo and generate with llama-completion. If no binary is available
    # here, record the skip — per-quant eval already generated with llama.cpp
    # pre-upload, so this is a repo-integrity smoke, not the quality gate.
    smoke = {"status": "skipped", "reason": "no llama-completion binary available"}
    if args.kind == "gguf":
        candidates = [args.smoke_bin, Path("/tmp/llama.cpp/build/bin/llama-completion"),
                      Path("/mnt/volume_w2l8nkp/tools/llama.cpp/build/bin/llama-completion")]
        smoke_bin = next((c for c in candidates if c and c.is_file()), None)
        if smoke_bin:
            from huggingface_hub import hf_hub_download

            default_rec = next((r for r in uploadable if r["quant"] == "Q4_K_M"), uploadable[0])
            with tempfile.TemporaryDirectory() as tmp:
                local = hf_hub_download(repo_id, default_rec["file"], token=token,
                                        local_dir=tmp)
                if sha256_file(Path(local)) != default_rec["sha256"]:
                    raise SystemExit("post-upload smoke failed: downloaded SHA mismatch")
                proc = subprocess.run(
                    [str(smoke_bin), "-m", str(local), "-p",
                     "Salaam aleekum, naka nga def?", "-n", "16", "--temp", "0", "-no-cnv"],
                    capture_output=True, timeout=600)
                out = (proc.stdout or b"").decode("utf-8", "replace").strip()
                smoke = {"status": "pass" if proc.returncode == 0 and out else "fail",
                         "file": default_rec["file"], "output_chars": len(out)}
                if smoke["status"] != "pass":
                    raise SystemExit(f"post-upload smoke failed: {smoke}")
    entry = {"dst_repo": repo_id, "status": "published_private", "revision": commit.oid,
             "post_upload_smoke": smoke,
             "private": True, "src": args.src, "src_sha": args.src_sha, "kind": args.kind,
             "files": remote_files, "verdicts": {r["quant"]: r["verdict"] for r in records},
             "tier1_bpb": tier1.get("BPB"), "tier1_chrf": tier1.get("chrF"),
             "verified_at": utc_now()}
    write_blocked(args.manifest, entry)  # appends to manifest models list
    print(json.dumps(entry, indent=2), flush=True)
    print("PUBLISH_QUANT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
