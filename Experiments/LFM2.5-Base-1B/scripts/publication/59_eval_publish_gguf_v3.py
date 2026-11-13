#!/usr/bin/env python3
"""Evaluate all V3 Q4_K_M GGUFs and publish metrics to the matching Hub sidecars."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT.parents[1] / "Evaluation"
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write  # noqa: E402


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_hubm():
    path = ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"
    spec = importlib.util.spec_from_file_location("publish_hub_metrics_56", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODELS: list[dict] = [
    {"size": "230m", "stage": "CPT", "config": "configs/gguf/lfm_230m_cpt_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-230M-Wolof-CPT-v3-GGUF", "base_hf": "Tonic/LFM2.5-230M-Wolof-CPT-v3", "local_gguf": "results/gguf/v3/230m/cpt/LFM2.5-230M-Wolof-CPT-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_230m_cpt_v3.json", "file_stem": "LFM2.5-230M-Wolof-CPT-v3"},
    {"size": "230m", "stage": "Instruct", "config": "configs/gguf/lfm_230m_instruct_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-230M-Wolof-Instruct-v3-GGUF", "base_hf": "Tonic/LFM2.5-230M-Wolof-Instruct-v3", "local_gguf": "results/gguf/v3/230m/instruct/LFM2.5-230M-Wolof-Instruct-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_230m_instruct_v3.json", "file_stem": "LFM2.5-230M-Wolof-Instruct-v3"},
    {"size": "350m", "stage": "CPT", "config": "configs/gguf/lfm_350m_cpt_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-350M-Wolof-CPT-v3-GGUF", "base_hf": "Tonic/LFM2.5-350M-Wolof-CPT-v3", "local_gguf": "results/gguf/v3/350m/cpt/LFM2.5-350M-Wolof-CPT-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_350m_cpt_v3.json", "file_stem": "LFM2.5-350M-Wolof-CPT-v3"},
    {"size": "350m", "stage": "Instruct", "config": "configs/gguf/lfm_350m_instruct_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-350M-Wolof-Instruct-v3-GGUF", "base_hf": "Tonic/LFM2.5-350M-Wolof-Instruct-v3", "local_gguf": "results/gguf/v3/350m/instruct/LFM2.5-350M-Wolof-Instruct-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_350m_instruct_v3.json", "file_stem": "LFM2.5-350M-Wolof-Instruct-v3"},
    {"size": "1.2b", "stage": "CPT", "config": "configs/gguf/lfm_1.2b_cpt_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-1.2B-Wolof-CPT-v3-GGUF", "base_hf": "Tonic/LFM2.5-1.2B-Wolof-CPT-v3", "local_gguf": "results/gguf/v3/1.2b/cpt/LFM2.5-1.2B-Wolof-CPT-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_1.2b_cpt_v3.json", "file_stem": "LFM2.5-1.2B-Wolof-CPT-v3"},
    {"size": "1.2b", "stage": "Instruct", "config": "configs/gguf/lfm_1.2b_instruct_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-1.2B-Wolof-Instruct-v3-GGUF", "base_hf": "Tonic/LFM2.5-1.2B-Wolof-Instruct-v3", "local_gguf": "results/gguf/v3/1.2b/instruct/LFM2.5-1.2B-Wolof-Instruct-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_1.2b_instruct_v3.json", "file_stem": "LFM2.5-1.2B-Wolof-Instruct-v3"},
    {"size": "2.6b", "stage": "CPT", "config": "configs/gguf/lfm_2.6b_cpt_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-2.6B-Wolof-CPT-v3-GGUF", "base_hf": "Tonic/LFM2.5-2.6B-Wolof-CPT-v3", "local_gguf": "results/gguf/v3/2.6b/cpt/LFM2.5-2.6B-Wolof-CPT-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_2.6b_cpt_v3.json", "file_stem": "LFM2.5-2.6B-Wolof-CPT-v3"},
    {"size": "2.6b", "stage": "Instruct", "config": "configs/gguf/lfm_2.6b_instruct_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-2.6B-Wolof-Instruct-v3-GGUF", "base_hf": "Tonic/LFM2.5-2.6B-Wolof-Instruct-v3", "local_gguf": "results/gguf/v3/2.6b/instruct/LFM2.5-2.6B-Wolof-Instruct-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_2.6b_instruct_v3.json", "file_stem": "LFM2.5-2.6B-Wolof-Instruct-v3"},
    {"size": "8b-a1b", "stage": "CPT", "config": "configs/gguf/lfm_8b_cpt_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-8B-A1B-Wolof-CPT-v3-GGUF", "base_hf": "Tonic/LFM2.5-8B-A1B-Wolof-CPT-v3", "local_gguf": "results/gguf/v3/8b-a1b/cpt/LFM2.5-8B-A1B-Wolof-CPT-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_8b_cpt_v3.json", "file_stem": "LFM2.5-8B-A1B-Wolof-CPT-v3"},
    {"size": "8b-a1b", "stage": "Instruct", "config": "configs/gguf/lfm_8b_instruct_v3_q4_k_m.yaml", "hub_repo": "Tonic/LFM2.5-8B-A1B-Wolof-Instruct-v3-GGUF", "base_hf": "Tonic/LFM2.5-8B-A1B-Wolof-Instruct-v3", "local_gguf": "results/gguf/v3/8b-a1b/instruct/LFM2.5-8B-A1B-Wolof-Instruct-v3-Q4_K_M.gguf", "clm_metrics": "metrics/clm/lfm_8b_instruct_v3.json", "file_stem": "LFM2.5-8B-A1B-Wolof-Instruct-v3"},
]


def job_key(entry: dict) -> str:
    return f"gguf-q4-v3:{entry['size']}:{entry['stage']}"


def resolve_gguf(entry: dict, *, token: str | None) -> Path:
    local = ROOT / entry["local_gguf"]
    if local.is_file() and local.stat().st_size > 1_000_000:
        return local
    filename = f"{entry['file_stem']}-Q4_K_M.gguf"
    print(f"[download] {entry['hub_repo']}/{filename}", flush=True)
    cached = hf_hub_download(repo_id=entry["hub_repo"], filename=filename, repo_type="model", token=token)
    return Path(cached)


def load_clm(entry: dict, *, token: str | None) -> dict | None:
    local = ROOT / entry["clm_metrics"]
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    for name in ("metrics_clm.json", "metrics.json"):
        try:
            path = hf_hub_download(repo_id=entry["base_hf"], filename=name, repo_type="model", token=token)
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if name == "metrics.json" and isinstance(payload.get("clm"), dict):
                return payload["clm"]
            if "BPB" in payload or "perplexity" in payload:
                return payload
        except Exception:
            continue
    return None


def run_eval(entry: dict, gguf_path: Path) -> dict:
    cfg_path = EVAL_ROOT / entry["config"]
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["gguf_path"] = str(gguf_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False, dir=str(cfg_path.parent)) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temp_cfg = Path(handle.name)
    py = ROOT / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EVAL_ROOT}:{EVAL_ROOT / 'scripts'}"
    env["TRACKIO_DISABLED"] = "1"
    cmd = [str(py), str(EVAL_ROOT / "scripts" / "evaluate_gguf.py"), "--config", str(temp_cfg)]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=EVAL_ROOT, check=True, env=env)
    finally:
        temp_cfg.unlink(missing_ok=True)
    out = Path(cfg["output_json"])
    if not out.is_absolute():
        stripped = out.as_posix()
        while stripped.startswith("../"):
            stripped = stripped[3:]
        out = (EVAL_ROOT / stripped).resolve()
    return json.loads(out.read_text(encoding="utf-8"))


def compose_payload(entry: dict, gguf: dict, clm: dict | None) -> dict:
    payload: dict = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "hub_repo": entry["hub_repo"],
        "size": entry["size"],
        "stage": entry["stage"],
        "gguf": gguf,
    }
    if clm:
        payload["clm"] = {
            "BPB": clm.get("BPB"),
            "perplexity": clm.get("perplexity"),
            "BLEU": clm.get("BLEU"),
            "chrF": clm.get("chrF"),
            "n_texts_scored": clm.get("n_texts_scored") or clm.get("n_texts_evaluated"),
            "n_bleu_pairs": clm.get("n_bleu_pairs"),
            "vocab_size": clm.get("vocab_size"),
            "run": clm.get("run"),
            "device": clm.get("device"),
        }
    return payload


def publish_composed(api: HfApi, token: str, entry: dict, raw: dict, hubm, *, dry_run: bool) -> dict:
    info = api.repo_info(repo_id=entry["hub_repo"], repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo {entry['hub_repo']}")
    metrics = hubm.normalize_metrics(raw, hub_repo=entry["hub_repo"], size=entry["size"], stage=entry["stage"], is_gguf=True)
    model_name = entry["hub_repo"].split("/")[-1]
    model_index = metrics["model-index"]
    readme = hubm.download_text(api, entry["hub_repo"], "README.md", token) or f"# {model_name}\n"
    meta, body = hubm.split_readme(readme)
    meta["model-index"] = model_index
    meta.setdefault("language", ["wo"])
    meta.setdefault("pipeline_tag", "text-generation")
    meta.setdefault("library_name", "gguf")
    meta.setdefault("tags", [])
    if isinstance(meta["tags"], list) and "gguf" not in meta["tags"]:
        meta["tags"] = list(meta["tags"]) + ["gguf"]
    section = hubm.build_eval_section(metrics=metrics, is_gguf=True, size=entry["size"])
    body = hubm.upsert_eval_section(body, section)
    new_readme = hubm.dump_readme(meta, body)
    if dry_run:
        return {"repo": entry["hub_repo"], "status": "dry-run", "ppl": (raw.get("gguf") or {}).get("perplexity")}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (tmp_path / "README.md").write_text(new_readme, encoding="utf-8")
        commit = api.upload_folder(folder_path=str(tmp_path), repo_id=entry["hub_repo"], repo_type="model", commit_message="Publish V3 GGUF Q4_K_M evaluation metrics and model-index", allow_patterns=["metrics.json", "README.md"])
    return {"repo": entry["hub_repo"], "status": "ok", "revision": commit.oid, "ppl": (raw.get("gguf") or {}).get("perplexity"), "BLEU": (raw.get("gguf") or {}).get("BLEU"), "chrF": (raw.get("gguf") or {}).get("chrF")}


def upsert_family_eval_row(key: str, ppl, bleu, chrf) -> None:
    path = ROOT / "docs" / "FAMILY_EVAL.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    row = f"| {key} | {ppl} | {bleu} | {chrf} |"
    pattern = re.compile(rf"^\| {re.escape(key)} \|.*$", flags=re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(row, text)
    else:
        marker = "## GGUF Q4_K_M"
        if marker not in text:
            text = text.rstrip() + "\n\n## GGUF Q4_K_M (gguf-llama-cpp protocol; not HF-BPB-equivalent)\n\n| Job | PPL | BLEU | chrF |\n|---|---:|---:|---:|\n"
        parts = text.split(marker, 1)
        head, rest = parts[0], parts[1]
        lines = rest.splitlines()
        insert_at = None
        for i, line in enumerate(lines):
            if line.startswith("|---"):
                j = i + 1
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                insert_at = j
                break
        if insert_at is None:
            rest = rest.rstrip() + "\n" + row + "\n"
        else:
            lines.insert(insert_at, row)
            rest = "\n".join(lines)
            if not rest.endswith("\n"):
                rest += "\n"
        text = head + marker + rest
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state", type=Path, default=ROOT / "results" / "full_pipeline" / "gguf_eval_v3_state.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "full_pipeline" / "gguf_eval_v3_report.json")
    args = parser.parse_args()
    load_trackio_env()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.skip_upload and not args.dry_run:
        raise SystemExit("HF_TOKEN required")
    hubm = load_hubm()
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema_version": 1, "status": "running", "jobs": {}, "uploads": []}
    models = MODELS
    if args.smoke:
        models = [m for m in models if m["size"] == "230m" and m["stage"] == "Instruct"]
    if args.only:
        models = [m for m in models if args.only == m["size"] or args.only == f"{m['size']}:{m['stage']}" or args.only.lower() == m["stage"].lower()]
        if not models:
            raise SystemExit(f"--only matched nothing: {args.only}")
    api = HfApi(token=token)
    uploads = []
    try:
        for entry in models:
            key = job_key(entry)
            rec = state["jobs"].setdefault(key, {"status": "pending"})
            print(f"==== {key} ====", flush=True)
            if args.resume and rec.get("status") == "complete" and rec.get("metrics"):
                print(f"skip complete {key}", flush=True)
                gguf_metrics = rec["metrics"]
            else:
                rec["status"] = "running"
                rec["started_at"] = utc_now()
                atomic_json_write(state_path, state)
                try:
                    if args.skip_eval:
                        cfg = yaml.safe_load((EVAL_ROOT / entry["config"]).read_text(encoding="utf-8"))
                        out = Path(cfg["output_json"])
                        if not out.is_absolute():
                            stripped = out.as_posix()
                            while stripped.startswith("../"):
                                stripped = stripped[3:]
                            out = EVAL_ROOT / stripped
                        gguf_metrics = json.loads(out.read_text(encoding="utf-8"))
                    else:
                        gguf_path = resolve_gguf(entry, token=token)
                        gguf_metrics = run_eval(entry, gguf_path)
                    rec["metrics"] = gguf_metrics
                    rec["status"] = "complete"
                    rec["completed_at"] = utc_now()
                except Exception as exc:
                    rec["status"] = "failed"
                    rec["error"] = str(exc)
                    rec["failed_at"] = utc_now()
                    atomic_json_write(state_path, state)
                    raise
                atomic_json_write(state_path, state)
            clm = load_clm(entry, token=token)
            raw = compose_payload(entry, gguf_metrics, clm)
            local_copy = (ROOT / entry["local_gguf"]).parent / "metrics.json"
            local_copy.parent.mkdir(parents=True, exist_ok=True)
            local_copy.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if not args.skip_upload:
                result = publish_composed(api, token or "", entry, raw, hubm, dry_run=args.dry_run)
                uploads.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            upsert_family_eval_row(key, gguf_metrics.get("perplexity"), gguf_metrics.get("BLEU"), gguf_metrics.get("chrF"))
        state["status"] = "complete"
        state["uploads"] = uploads
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        report = {"updated_at": utc_now(), "status": "complete", "n_jobs": len(models), "uploads": uploads, "jobs": {k: {"status": v.get("status"), "perplexity": (v.get("metrics") or {}).get("perplexity"), "BLEU": (v.get("metrics") or {}).get("BLEU"), "chrF": (v.get("metrics") or {}).get("chrF"), "error": v.get("error")} for k, v in state.get("jobs", {}).items()}}
        report_path = args.report if args.report.is_absolute() else ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2)[:3000])
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
