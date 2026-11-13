#!/usr/bin/env python3
"""Upload HF-compliant metrics.json + README model-index for Tiny Aya Wolof models.

Mirrors LFM 56_publish_hub_metrics.py: compose clm + sft payloads, write model-index,
upsert Evaluation section. Primary BPB = ortho ns+mixed test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write

EVAL_MARKER_START = "<!-- FAMILY_EVAL_START -->"
EVAL_MARKER_END = "<!-- FAMILY_EVAL_END -->"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def normalize_clm(raw: dict | None) -> dict | None:
    """Map Tiny Aya BPB probe JSON (bpb.all) or LFM-style BPB keys into HF clm blob."""
    if not isinstance(raw, dict) or not raw:
        return None
    if raw.get("BPB") is not None:
        return {
            "BPB": raw.get("BPB"),
            "perplexity": raw.get("perplexity"),
            "BLEU": raw.get("BLEU"),
            "chrF": raw.get("chrF"),
            "bpb_by_lang": raw.get("bpb") or raw.get("bpb_by_lang"),
            "source": raw.get("data") or raw.get("source"),
        }
    bpb = raw.get("bpb") or {}
    if isinstance(bpb, dict) and bpb.get("all") is not None:
        return {
            "BPB": bpb.get("all"),
            "BPB_wo": bpb.get("wo"),
            "BPB_fr": bpb.get("fr"),
            "BPB_en": bpb.get("en"),
            "perplexity": raw.get("perplexity"),
            "bpb_by_lang": bpb,
            "counts": raw.get("counts"),
            "source": raw.get("data"),
        }
    return None


def build_model_index(*, model_name: str, metrics: dict) -> list[dict]:
    results: list[dict] = []
    source = {
        "name": "GalsenAI Tiny Aya Wolof family eval",
        "url": f"https://huggingface.co/{metrics.get('hub_repo', model_name)}/blob/main/metrics.json",
    }
    clm = metrics.get("clm") or {}
    if clm:
        mets = [
            metric_entry("bpb", clm.get("BPB"), name="Bits Per Byte (ortho ns+mixed test)"),
            metric_entry("bpb_wo", clm.get("BPB_wo"), name="BPB Wolof subset"),
            metric_entry("perplexity", clm.get("perplexity"), name="Perplexity (same tokenizer)"),
            metric_entry(
                "bleu",
                clm.get("BLEU"),
                name="BLEU (100-pair verbalized continuation)",
            ),
            metric_entry(
                "chrf",
                clm.get("chrF"),
                name="chrF (100-pair verbalized continuation)",
            ),
        ]
        results.append(
            {
                "task": {"type": "text-generation", "name": "Wolof causal LM"},
                "dataset": {
                    "type": "tinyaya-wolof-ntp-ortho-ns-mixed",
                    "name": "Tiny Aya Wolof NTP ortho ns+mixed test",
                    "split": "test",
                },
                "metrics": [m for m in mets if m is not None],
                "source": source,
            }
        )
    sft = metrics.get("sft") or {}
    per_task = sft.get("per_task") or {}
    for task_name, task_metrics in sorted(per_task.items()):
        if not isinstance(task_metrics, dict):
            continue
        mets = [
            metric_entry(
                "loss",
                task_metrics.get("loss")
                or task_metrics.get("eval_loss")
                or task_metrics.get("mean_loss"),
                name="Eval loss",
            ),
            metric_entry(
                "perplexity",
                task_metrics.get("perplexity")
                or task_metrics.get("eval_perplexity")
                or task_metrics.get("ppl"),
                name="Eval perplexity",
            ),
        ]
        filtered = [m for m in mets if m is not None]
        if not filtered:
            continue
        results.append(
            {
                "task": {"type": "text-generation", "name": f"SFT task: {task_name}"},
                "dataset": {
                    "type": "tinyaya-wolof-sft-anchored",
                    "name": f"Tiny Aya Wolof SFT ({task_name})",
                    "split": "eval",
                },
                "metrics": filtered,
                "source": source,
            }
        )
    return [{"name": model_name, "results": results}]


def build_eval_section(*, metrics: dict) -> str:
    lines = [
        EVAL_MARKER_START,
        "## Evaluation",
        "",
        "- **Primary BPB**: ortho non-standard+mixed NTP test (cross-tokenizer gold).",
        "- **Secondary BPB**: original clean test when measured (`metrics_clm_original_test.json`).",
        "- **License**: CC-BY-NC-4.0 + Cohere Labs AUP.",
        "- Machine-readable: [`metrics.json`](./metrics.json), [`metrics_clm.json`](./metrics_clm.json).",
        "",
    ]
    clm = metrics.get("clm") or {}
    if clm:
        lines.extend(
            [
                "### Causal LM (ortho ns+mixed primary)",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| BPB ↓ | {clm.get('BPB')} |",
                f"| BPB wo | {clm.get('BPB_wo')} |",
                f"| Perplexity | {clm.get('perplexity')} |",
                f"| BLEU | {clm.get('BLEU')} |",
                f"| chrF | {clm.get('chrF')} |",
                "",
            ]
        )
        secondary = clm.get("secondary_original_test") or {}
        if isinstance(secondary, dict) and secondary.get("BPB") is not None:
            lines.extend(
                [
                    "### Causal LM (original test secondary)",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| BPB ↓ | {secondary.get('BPB')} |",
                    "",
                    "Secondary file: [`metrics_clm_original_test.json`](./metrics_clm_original_test.json).",
                    "",
                ]
            )
    sft = metrics.get("sft") or {}
    per_task = sft.get("per_task") or {}
    if per_task:
        lines.extend(
            [
                "### Instruct task eval",
                "",
                "| Task | Loss | PPL |",
                "|---|---:|---:|",
            ]
        )
        for task_name, task_metrics in sorted(per_task.items()):
            if not isinstance(task_metrics, dict):
                continue
            loss = (
                task_metrics.get("loss")
                or task_metrics.get("mean_loss")
                or task_metrics.get("eval_loss")
                or "—"
            )
            ppl = (
                task_metrics.get("perplexity")
                or task_metrics.get("ppl")
                or task_metrics.get("eval_perplexity")
                or "—"
            )
            lines.append(f"| {task_name} | {loss} | {ppl} |")
        lines.append("")
    lines.append(EVAL_MARKER_END)
    return "\n".join(lines)


def upsert_eval_section(body: str, section: str) -> str:
    pattern = re.compile(
        re.escape(EVAL_MARKER_START) + r".*?" + re.escape(EVAL_MARKER_END),
        flags=re.DOTALL,
    )
    if pattern.search(body):
        return pattern.sub(section, body)
    legacy = re.compile(r"(?im)^##\s+(Metrics|Evaluation)\s*$.*?(?=^##\s|\Z)", flags=re.DOTALL)
    if legacy.search(body):
        return legacy.sub(section + "\n\n", body, count=1)
    return body.rstrip() + "\n\n" + section + "\n"


def _redact(value):
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"/mnt/(?:volume|data)[^/\s]*/[^\s\"']*", "<REDACTED_REMOTE_PATH>", value)
        value = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "<REDACTED_TOKEN>", value)
        return value
    return value


def discover_targets(cfg: dict) -> list[dict]:
    targets = []
    for group in ("cpt", "cpt_std", "instruct", "instruct_std", "dpo", "dpo_std"):
        for entry in cfg.get(group) or []:
            if not entry.get("hub_repo"):
                continue
            stage_default = (
                "CPT"
                if group.startswith("cpt")
                else "DPO"
                if group.startswith("dpo")
                else "Instruct"
            )
            targets.append(
                {
                    "hub_repo": entry["hub_repo"],
                    "stage": entry.get("stage") or stage_default,
                    "role": entry.get("role"),
                    "group": group,
                    "local_artifact": entry.get("local_artifact"),
                    "local_clm": entry.get("clm_metrics") or entry.get("local_clm"),
                    "local_clm_secondary": entry.get("clm_metrics_secondary")
                    or entry.get("local_clm_secondary"),
                    "local_sft": entry.get("sft_eval") or entry.get("local_sft"),
                }
            )
    return targets


def load_local_json(path: Path | None) -> dict | None:
    if not path:
        return None
    p = path if path.is_absolute() else ROOT / path
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def download_json(token: str, repo_id: str, path: str) -> dict | None:
    try:
        local = hf_hub_download(repo_id, path, repo_type="model", token=token)
        return json.loads(Path(local).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def download_text(token: str, repo_id: str, path: str) -> str | None:
    try:
        local = hf_hub_download(repo_id, path, repo_type="model", token=token)
        return Path(local).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def compose_payload(token: str, target: dict) -> dict:
    clm = normalize_clm(load_local_json(Path(target["local_clm"]) if target.get("local_clm") else None))
    if clm is None:
        clm = normalize_clm(download_json(token, target["hub_repo"], "metrics_clm.json"))
    secondary = normalize_clm(
        load_local_json(
            Path(target["local_clm_secondary"]) if target.get("local_clm_secondary") else None
        )
    )
    if secondary is None:
        secondary = normalize_clm(
            download_json(token, target["hub_repo"], "metrics_clm_original_test.json")
        )
    if clm and secondary:
        clm["secondary_original_test"] = secondary
    sft_raw = load_local_json(Path(target["local_sft"]) if target.get("local_sft") else None)
    if sft_raw is None:
        sft_raw = download_json(token, target["hub_repo"], "metrics_sft_evaluation.json")
    if sft_raw is None:
        sft_raw = download_json(token, target["hub_repo"], "metrics.json")
    sft = None
    if isinstance(sft_raw, dict):
        if isinstance(sft_raw.get("per_task"), dict):
            sft = {
                "per_task": sft_raw["per_task"],
                "qualitative_count": sft_raw.get("qualitative_count"),
                "mean_loss": sft_raw.get("mean_loss"),
            }
        elif isinstance((sft_raw.get("sft") or {}).get("per_task"), dict):
            sft = sft_raw["sft"]
    composed: dict = {}
    if clm:
        composed["clm"] = _redact(clm)
    if sft:
        composed["sft"] = _redact(sft)
    if not composed:
        raise RuntimeError(f"no clm/sft metrics available for {target['hub_repo']}")
    return composed


def publish_one(api: HfApi, token: str, target: dict, *, dry_run: bool) -> dict:
    repo_id = target["hub_repo"]
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo {repo_id}")

    raw = compose_payload(token, target)
    model_name = repo_id.split("/")[-1]
    metrics = {
        **raw,
        "schema_version": 2,
        "updated_at": utc_now(),
        "hub_repo": repo_id,
        "stage": target.get("stage"),
        "size": "3.35B",
        "protocol": "evaluate_clm-bpb-bleu-chrf",
        "notes": (
            "Gold CLM via Evaluation/evaluate_clm.py (BPB+PPL+BLEU+chrF) on "
            "ortho NTP test; CC-BY-NC-4.0 + Cohere Labs AUP."
        ),
        "license_note": "cc-by-nc-4.0 + Cohere Labs AUP",
    }
    metrics["model-index"] = build_model_index(model_name=model_name, metrics=metrics)

    readme = download_text(token, repo_id, "README.md") or f"# {model_name}\n"
    meta, body = split_readme(readme)
    meta["model-index"] = metrics["model-index"]
    meta.setdefault("language", ["wo"])
    meta.setdefault("pipeline_tag", "text-generation")
    meta.setdefault("library_name", "transformers")
    meta.setdefault("license", "cc-by-nc-4.0")
    section = build_eval_section(metrics=metrics)
    body = upsert_eval_section(body, section)
    new_readme = dump_readme(meta, body)

    result = {"repo": repo_id, "status": "dry-run" if dry_run else "pending"}
    if dry_run:
        result["n_model_index_results"] = len(metrics["model-index"][0]["results"])
        return result

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (tmp_path / "README.md").write_text(new_readme, encoding="utf-8")
        # Also drop normalized metrics_clm.json when clm present.
        if metrics.get("clm"):
            (tmp_path / "metrics_clm.json").write_text(
                json.dumps(metrics["clm"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        commit = api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish HF-compliant evaluation metrics and model-index",
        )
        result["status"] = "ok"
        result["revision"] = commit.oid
        result["n_model_index_results"] = len(metrics["model-index"][0]["results"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish Tiny Aya Hub metrics + model-index")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "publication" / "eval_family.yaml",
    )
    parser.add_argument("--only-repo", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "hub_metrics_publish.json",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN required")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = discover_targets(cfg)
    if args.only_repo:
        wanted = set(args.only_repo)
        targets = [t for t in targets if t["hub_repo"] in wanted]
    # Enrich local metric paths from convention if missing in yaml.
    for t in targets:
        role = t.get("role") or ""
        group = t.get("group") or ""
        ortho = "std" if group.endswith("_std") else "ns"
        prefix = "3b35_std" if ortho == "std" else "3b35"
        if t.get("local_clm"):
            clm_p = ROOT / t["local_clm"] if not Path(t["local_clm"]).is_absolute() else Path(t["local_clm"])
            if not clm_p.is_file():
                t["local_clm"] = None
        if not t.get("local_clm"):
            candidates = [
                ROOT / "results" / "bpb" / f"cpt_{ortho}_3b35.json"
                if role == "cpt" or (t.get("stage") or "").upper() == "CPT"
                else None,
                ROOT / "results" / "bpb" / f"instruct_{role}_{ortho}_3b35.json",
                ROOT / "results" / "bpb" / f"dpo_{ortho}_3b35.json" if role == "dpo" else None,
                ROOT / "results" / "bpb" / f"{role}_3b35.json",
                ROOT / "results" / "bpb" / "cpt_3b35.json",
            ]
            for c in candidates:
                if c and c.is_file():
                    t["local_clm"] = str(c)
                    break
        if t.get("local_sft"):
            sft_p = ROOT / t["local_sft"] if not Path(t["local_sft"]).is_absolute() else Path(t["local_sft"])
            if not sft_p.is_file():
                t["local_sft"] = None
        if not t.get("local_sft"):
            sft_candidates = []
            if role.startswith("from_"):
                sft_candidates.append(
                    ROOT / "results" / "sft" / f"eval_{prefix}_{role}" / "evaluation.json"
                )
                sft_candidates.append(ROOT / "results" / "sft" / f"eval_{role}" / "evaluation.json")
            if role == "dpo":
                sft_candidates.append(
                    ROOT / "results" / "sft" / f"eval_{prefix}_dpo" / "evaluation.json"
                )
                sft_candidates.append(ROOT / "results" / "sft" / "eval_dpo" / "evaluation.json")
            for p in sft_candidates:
                if p.is_file():
                    t["local_sft"] = str(p)
                    break

    api = HfApi(token=token)
    results = []
    for target in targets:
        try:
            results.append(publish_one(api, token, target, dry_run=args.dry_run))
            print(json.dumps(results[-1], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            err = {"repo": target["hub_repo"], "status": "error", "error": str(exc)}
            results.append(err)
            print(json.dumps(err, indent=2), flush=True)

    report = {
        "updated_at": utc_now(),
        "dry_run": args.dry_run,
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_failed": sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }
    out = args.report if args.report.is_absolute() else ROOT / args.report
    atomic_json_write(out, report)
    print(f"report={out} ok={report['n_ok']} failed={report['n_failed']}", flush=True)
    return 0 if report["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
