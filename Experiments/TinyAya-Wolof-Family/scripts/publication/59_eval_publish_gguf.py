#!/usr/bin/env python3
"""Evaluate Tiny Aya Q4_K_M GGUFs and publish metrics to Hub sidecars + FAMILY_EVAL."""

from __future__ import annotations

import argparse
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

EVAL_MARKER_START = "<!-- FAMILY_EVAL_START -->"
EVAL_MARKER_END = "<!-- FAMILY_EVAL_END -->"


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


def split_readme(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, match.group(2)


def dump_readme(meta: dict, body: str) -> str:
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False).strip()
    return f"---\n{front}\n---\n\n{body.lstrip(chr(10))}"


def metric_entry(metric_type: str, value, *, name: str | None = None) -> dict | None:
    if value is None:
        return None
    entry: dict = {"type": metric_type, "value": value}
    if name:
        entry["name"] = name
    return entry


def upsert_eval_section(body: str, section: str) -> str:
    pattern = re.compile(
        re.escape(EVAL_MARKER_START) + r".*?" + re.escape(EVAL_MARKER_END),
        flags=re.DOTALL,
    )
    if pattern.search(body):
        return pattern.sub(section, body)
    return body.rstrip() + "\n\n" + section + "\n"


MODELS: list[dict] = [
    {
        "stage": "CPT",
        "ortho": "ns",
        "config": "configs/gguf/tinyaya_3b35_cpt_ns_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-CPT-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-CPT-v1",
        "local_gguf": "results/gguf/v1/3b35/cpt/ns/TinyAya-3.35B-Wolof-CPT-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_cpt_ns_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-CPT-v1",
    },
    {
        "stage": "CPT",
        "ortho": "std",
        "config": "configs/gguf/tinyaya_3b35_cpt_std_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-CPT-std-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-CPT-std-v1",
        "local_gguf": "results/gguf/v1/3b35/cpt/std/TinyAya-3.35B-Wolof-CPT-std-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_cpt_std_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-CPT-std-v1",
    },
    {
        "stage": "InstructEarth",
        "ortho": "ns",
        "config": "configs/gguf/tinyaya_3b35_instructearth_ns_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-InstructEarth-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-InstructEarth-v1",
        "local_gguf": "results/gguf/v1/3b35/instructearth/ns/TinyAya-3.35B-Wolof-InstructEarth-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_from_earth_ns_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-InstructEarth-v1",
    },
    {
        "stage": "InstructEarth",
        "ortho": "std",
        "config": "configs/gguf/tinyaya_3b35_instructearth_std_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-InstructEarth-std-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-InstructEarth-std-v1",
        "local_gguf": "results/gguf/v1/3b35/instructearth/std/TinyAya-3.35B-Wolof-InstructEarth-std-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_from_earth_std_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-InstructEarth-std-v1",
    },
    {
        "stage": "InstructGlobal",
        "ortho": "ns",
        "config": "configs/gguf/tinyaya_3b35_instructglobal_ns_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-InstructGlobal-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-InstructGlobal-v1",
        "local_gguf": "results/gguf/v1/3b35/instructglobal/ns/TinyAya-3.35B-Wolof-InstructGlobal-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_from_global_ns_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-InstructGlobal-v1",
    },
    {
        "stage": "InstructGlobal",
        "ortho": "std",
        "config": "configs/gguf/tinyaya_3b35_instructglobal_std_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-InstructGlobal-std-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-InstructGlobal-std-v1",
        "local_gguf": "results/gguf/v1/3b35/instructglobal/std/TinyAya-3.35B-Wolof-InstructGlobal-std-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_from_global_std_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-InstructGlobal-std-v1",
    },
    {
        "stage": "DPO",
        "ortho": "ns",
        "config": "configs/gguf/tinyaya_3b35_dpo_ns_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-DPO-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-DPO-v1",
        "local_gguf": "results/gguf/v1/3b35/dpo/ns/TinyAya-3.35B-Wolof-DPO-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_dpo_ns_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-DPO-v1",
    },
    {
        "stage": "DPO",
        "ortho": "std",
        "config": "configs/gguf/tinyaya_3b35_dpo_std_q4_k_m.yaml",
        "hub_repo": "Tonic/TinyAya-3.35B-Wolof-DPO-std-v1-GGUF",
        "base_hf": "Tonic/TinyAya-3.35B-Wolof-DPO-std-v1",
        "local_gguf": "results/gguf/v1/3b35/dpo/std/TinyAya-3.35B-Wolof-DPO-std-v1-Q4_K_M.gguf",
        "clm_metrics": "results/bpb/clm_dpo_std_3b35.json",
        "file_stem": "TinyAya-3.35B-Wolof-DPO-std-v1",
    },
]


def job_key(entry: dict) -> str:
    return f"gguf-q4:{entry['stage']}:{entry['ortho']}"


def resolve_gguf(entry: dict, *, token: str | None) -> Path:
    local = ROOT / entry["local_gguf"]
    if local.is_file() and local.stat().st_size > 1_000_000:
        return local
    filename = f"{entry['file_stem']}-Q4_K_M.gguf"
    print(f"[download] {entry['hub_repo']}/{filename}", flush=True)
    cached = hf_hub_download(
        repo_id=entry["hub_repo"], filename=filename, repo_type="model", token=token
    )
    return Path(cached)


def load_clm(entry: dict, *, token: str | None) -> dict | None:
    local = ROOT / entry["clm_metrics"]
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    for name in ("metrics_clm.json", "metrics.json"):
        try:
            path = hf_hub_download(
                repo_id=entry["base_hf"], filename=name, repo_type="model", token=token
            )
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
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".yaml", delete=False, dir=str(cfg_path.parent)
    ) as handle:
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
        "size": "3b35",
        "stage": entry["stage"],
        "ortho": entry["ortho"],
        "protocol": "gguf-llama-cpp",
        "gguf": gguf,
        "notes": (
            "GGUF metrics are not HF-BPB-equivalent; HF BPB remains the cross-tokenizer gold metric. "
            "K-quants were imatrix-calibrated on the matching ortho NTP test set."
        ),
    }
    if clm:
        payload["clm"] = {
            "BPB": clm.get("BPB"),
            "perplexity": clm.get("perplexity"),
            "BLEU": clm.get("BLEU"),
            "chrF": clm.get("chrF"),
            "n_texts_scored": clm.get("n_texts_scored") or clm.get("n_texts_evaluated"),
            "n_bleu_pairs": clm.get("n_bleu_pairs"),
            "run": clm.get("run"),
        }
    model_name = entry["hub_repo"].split("/")[-1]
    source = {
        "name": "GalsenAI Tiny Aya Wolof GGUF eval",
        "url": f"https://huggingface.co/{entry['hub_repo']}/blob/main/metrics.json",
    }
    mets = [
        metric_entry("perplexity", gguf.get("perplexity"), name="llama.cpp perplexity (Q4_K_M)"),
        metric_entry("bleu", gguf.get("BLEU"), name="BLEU via llama-completion"),
        metric_entry("chrf", gguf.get("chrF"), name="chrF via llama-completion"),
    ]
    payload["model-index"] = [
        {
            "name": model_name,
            "results": [
                {
                    "task": {"type": "text-generation", "name": "Wolof GGUF Q4_K_M"},
                    "dataset": {
                        "type": f"tinyaya-wolof-ntp-ortho-{entry['ortho']}-mixed",
                        "name": f"Tiny Aya Wolof NTP ortho {entry['ortho']}+mixed test",
                        "split": "test",
                    },
                    "metrics": [m for m in mets if m is not None],
                    "source": source,
                }
            ],
        }
    ]
    return payload


def build_eval_section(payload: dict) -> str:
    g = payload.get("gguf") or {}
    clm = payload.get("clm") or {}
    lines = [
        EVAL_MARKER_START,
        "## Evaluation",
        "",
        "- **Protocol**: `gguf-llama-cpp` (not HF-BPB-equivalent).",
        "- K-quants imatrix-calibrated on matching ortho NTP test.",
        "- Machine-readable: [`metrics.json`](./metrics.json).",
        "",
        "### GGUF Q4_K_M",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Perplexity | {g.get('perplexity')} |",
        f"| BLEU | {g.get('BLEU')} |",
        f"| chrF | {g.get('chrF')} |",
        "",
    ]
    if clm.get("BPB") is not None:
        lines.extend(
            [
                "### HF reference (same verbalized corpus)",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| BPB | {clm.get('BPB')} |",
                f"| BLEU | {clm.get('BLEU')} |",
                f"| chrF | {clm.get('chrF')} |",
                "",
            ]
        )
    lines.append(EVAL_MARKER_END)
    return "\n".join(lines)


def publish_composed(api: HfApi, entry: dict, raw: dict, *, dry_run: bool) -> dict:
    info = api.repo_info(repo_id=entry["hub_repo"], repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo {entry['hub_repo']}")
    model_name = entry["hub_repo"].split("/")[-1]
    try:
        readme = hf_hub_download(
            repo_id=entry["hub_repo"], filename="README.md", repo_type="model"
        )
        text = Path(readme).read_text(encoding="utf-8")
    except Exception:
        text = f"# {model_name}\n"
    meta, body = split_readme(text)
    meta["model-index"] = raw["model-index"]
    meta.setdefault("language", ["wo"])
    meta.setdefault("pipeline_tag", "text-generation")
    meta.setdefault("library_name", "gguf")
    meta.setdefault("tags", [])
    if isinstance(meta["tags"], list) and "gguf" not in meta["tags"]:
        meta["tags"] = list(meta["tags"]) + ["gguf"]
    body = upsert_eval_section(body, build_eval_section(raw))
    new_readme = dump_readme(meta, body)
    if dry_run:
        return {
            "repo": entry["hub_repo"],
            "status": "dry-run",
            "ppl": (raw.get("gguf") or {}).get("perplexity"),
        }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "metrics.json").write_text(
            json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (tmp_path / "README.md").write_text(new_readme, encoding="utf-8")
        commit = api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=entry["hub_repo"],
            repo_type="model",
            commit_message="Publish TinyAya GGUF Q4_K_M evaluation metrics and model-index",
            allow_patterns=["metrics.json", "README.md"],
        )
    return {
        "repo": entry["hub_repo"],
        "status": "ok",
        "revision": commit.oid,
        "ppl": (raw.get("gguf") or {}).get("perplexity"),
        "BLEU": (raw.get("gguf") or {}).get("BLEU"),
        "chrF": (raw.get("gguf") or {}).get("chrF"),
    }


def upsert_family_eval_row(key: str, ppl, bleu, chrf) -> None:
    path = ROOT / "docs" / "FAMILY_EVAL.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    # Drop obsolete "GGUF not in scope" note.
    text = re.sub(r"^GGUF not in scope.*$", "", text, flags=re.MULTILINE)
    row = f"| {key} | {ppl} | {bleu} | {chrf} |"
    pattern = re.compile(rf"^\| {re.escape(key)} \|.*$", flags=re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(row, text)
    else:
        marker = "## GGUF Q4_K_M"
        if marker not in text:
            text = (
                text.rstrip()
                + "\n\n## GGUF Q4_K_M (gguf-llama-cpp; imatrix-calibrated; not HF-BPB-equivalent)\n\n"
                + "| Job | PPL | BLEU | chrF |\n|---|---:|---:|---:|\n"
            )
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
    parser.add_argument("--only", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "gguf_eval_state.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "gguf_eval_report.json",
    )
    args = parser.parse_args()
    load_trackio_env()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.skip_upload and not args.dry_run:
        raise SystemExit("HF_TOKEN required")
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema_version": 1, "status": "running", "jobs": {}, "uploads": []}
    models = MODELS
    if args.only:
        needle = args.only.lower()
        models = [
            m
            for m in models
            if needle in job_key(m).lower()
            or needle in m["file_stem"].lower()
            or needle in f"{m['stage']}-{m['ortho']}".lower()
        ]
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
            local_copy.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if not args.skip_upload:
                result = publish_composed(api, entry, raw, dry_run=args.dry_run)
                uploads.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            upsert_family_eval_row(
                key,
                gguf_metrics.get("perplexity"),
                gguf_metrics.get("BLEU"),
                gguf_metrics.get("chrF"),
            )
        state["status"] = "complete"
        state["uploads"] = uploads
        state["updated_at"] = utc_now()
        atomic_json_write(state_path, state)
        report = {
            "updated_at": utc_now(),
            "status": "complete",
            "n_jobs": len(models),
            "uploads": uploads,
            "jobs": {
                k: {
                    "status": v.get("status"),
                    "perplexity": (v.get("metrics") or {}).get("perplexity"),
                    "BLEU": (v.get("metrics") or {}).get("BLEU"),
                    "chrF": (v.get("metrics") or {}).get("chrF"),
                    "error": v.get("error"),
                }
                for k, v in state.get("jobs", {}).items()
            },
        }
        report_path = args.report if args.report.is_absolute() else ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2)[:3000])
        print("TINYAYA_GGUF_EVAL_OK", flush=True)
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
