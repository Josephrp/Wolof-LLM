#!/usr/bin/env python3
"""Upload HF-compliant metrics.json + README model-index for each Tonic Wolof model."""

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
from huggingface_hub import HfApi, hf_hub_download, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write  # noqa: E402

EVAL_MARKER_START = "<!-- FAMILY_EVAL_START -->"
EVAL_MARKER_END = "<!-- FAMILY_EVAL_END -->"

SIZE_HUB = {
    "230m": "230M",
    "350m": "350M",
    "1.2b": "1.2B",
    "2.6b": "2.6B",
    "8b-a1b": "8B-A1B",
}

TOK_FAMILY = {
    "230m": "65k-ext",
    "350m": "65k-ext",
    "1.2b": "65k-ext",
    "2.6b": "128k-ext",
    "8b-a1b": "128k-ext",
}


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
    # Keep key order stable-ish; YAML dump without sorting for readability.
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False).strip()
    body = body.lstrip("\n")
    return f"---\n{front}\n---\n\n{body}"


def metric_entry(metric_type: str, value, *, name: str | None = None) -> dict | None:
    if value is None:
        return None
    entry: dict = {"type": metric_type, "value": value}
    if name:
        entry["name"] = name
    return entry


def build_model_index(*, model_name: str, metrics: dict, is_gguf: bool) -> list[dict]:
    results: list[dict] = []
    source = {
        "name": "GalsenAI LFM2.5 Wolof family eval",
        "url": f"https://huggingface.co/{metrics.get('hub_repo', model_name)}/blob/main/metrics.json",
    }

    clm = metrics.get("clm") or {}
    if clm and not is_gguf:
        mets = [
            metric_entry("bpb", clm.get("BPB"), name="Bits Per Byte (cross-tokenizer gold)"),
            metric_entry("perplexity", clm.get("perplexity"), name="Perplexity (same tokenizer only)"),
            metric_entry("bleu", clm.get("BLEU"), name="BLEU (100-pair verbalized)"),
            metric_entry("chrf", clm.get("chrF"), name="chrF (100-pair verbalized)"),
        ]
        results.append(
            {
                "task": {"type": "text-generation", "name": "Wolof causal LM"},
                "dataset": {
                    "type": "wolof-clm-corpus-test",
                    "name": "Wolof CLM corpus test (verbalized)",
                    "split": "test",
                },
                "metrics": [m for m in mets if m is not None],
                "source": source,
            }
        )

    sft = metrics.get("sft") or {}
    per_task = sft.get("per_task") or {}
    if per_task and not is_gguf:
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
                        "type": "wolof-sft-mixture",
                        "name": f"Wolof SFT mixture ({task_name})",
                        "split": "eval",
                    },
                    "metrics": filtered,
                    "source": source,
                }
            )

    gguf = metrics.get("gguf") or {}
    if is_gguf and gguf:
        mets = [
            metric_entry("perplexity", gguf.get("perplexity"), name="llama.cpp perplexity (Q4_K_M)"),
            metric_entry("bleu", gguf.get("BLEU"), name="BLEU via llama-completion"),
            metric_entry("chrf", gguf.get("chrF"), name="chrF via llama-completion"),
        ]
        results.append(
            {
                "task": {"type": "text-generation", "name": "Wolof GGUF Q4_K_M"},
                "dataset": {
                    "type": "wolof-clm-corpus-test",
                    "name": "Wolof CLM corpus test (verbalized, gguf-llama-cpp)",
                    "split": "test",
                },
                "metrics": [m for m in mets if m is not None],
                "source": source,
            }
        )

    return [{"name": model_name, "results": results}]


def build_eval_section(*, metrics: dict, is_gguf: bool, size: str) -> str:
    lines = [
        EVAL_MARKER_START,
        "## Evaluation",
        "",
        f"- **Tokenizer family**: `{TOK_FAMILY.get(size, 'unknown')}`",
        "- **Protocol**: HF `evaluate_clm.py` BPB is the cross-tokenizer gold metric.",
        "- Machine-readable copy: [`metrics.json`](./metrics.json).",
        "",
    ]
    if is_gguf:
        g = metrics.get("gguf") or {}
        lines.extend(
            [
                "### GGUF Q4_K_M (`gguf-llama-cpp`; not HF-BPB-equivalent)",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Perplexity | {g.get('perplexity')} |",
                f"| BLEU | {g.get('BLEU')} |",
                f"| chrF | {g.get('chrF')} |",
                "",
                "Compare GGUF PPL only within the same GGUF/tokenizer family. Use HF BPB for cross-model ranking.",
                "",
            ]
        )
        clm = metrics.get("clm") or {}
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
    else:
        clm = metrics.get("clm") or {}
        lines.extend(
            [
                "### Causal LM (verbalized corpus test)",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| BPB ↓ | {clm.get('BPB')} |",
                f"| Perplexity | {clm.get('perplexity')} |",
                f"| BLEU | {clm.get('BLEU')} |",
                f"| chrF | {clm.get('chrF')} |",
                "",
            ]
        )
        sft = metrics.get("sft") or {}
        per_task = sft.get("per_task") or {}
        if per_task:
            lines.extend(
                [
                    "### Instruct task eval (`12_evaluate_sft.py`)",
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
                    or task_metrics.get("eval_loss")
                    or task_metrics.get("mean_loss")
                    or "—"
                )
                ppl = (
                    task_metrics.get("perplexity")
                    or task_metrics.get("eval_perplexity")
                    or task_metrics.get("ppl")
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

    # Replace legacy "## Metrics" / "## Evaluation" blocks if present.
    legacy = re.compile(r"(?im)^##\s+(Metrics|Evaluation)\s*$.*?(?=^##\s|\Z)", flags=re.DOTALL)
    if legacy.search(body):
        return legacy.sub(section + "\n\n", body, count=1)

    return body.rstrip() + "\n\n" + section + "\n"


def normalize_metrics(raw: dict, *, hub_repo: str, size: str, stage: str, is_gguf: bool) -> dict:
    payload = dict(raw)
    payload["schema_version"] = 2
    payload["updated_at"] = utc_now()
    payload["hub_repo"] = hub_repo
    payload["size"] = size
    payload["stage"] = stage
    model_name = hub_repo.split("/")[-1]
    payload["model-index"] = build_model_index(model_name=model_name, metrics=payload, is_gguf=is_gguf)
    payload["protocol"] = "gguf-llama-cpp" if is_gguf else "hf-evaluate-clm"
    payload["notes"] = (
        "GGUF metrics are not HF-BPB-equivalent; BPB is the cross-tokenizer gold metric."
        if is_gguf
        else "BPB is the cross-tokenizer gold metric; do not compare PPL across tokenizer families."
    )
    return payload


def discover_targets(cfg: dict) -> list[dict]:
    targets = []
    for key in (
        "clm_cpt",
        "clm_instruct",
        "clm_instruct_v2",
        "clm_cpt_v3",
        "clm_instruct_v3",
    ):
        for entry in cfg.get(key, []):
            targets.append(
                {
                    "hub_repo": entry["hub_repo"],
                    "size": entry["size"],
                    "stage": entry["stage"],
                    "is_gguf": False,
                }
            )
    for key in ("gguf_q4", "gguf_q4_v3"):
        for entry in cfg.get(key, []):
            targets.append(
                {
                    "hub_repo": entry["hub_repo"],
                    "size": entry["size"],
                    "stage": entry["stage"],
                    "is_gguf": True,
                }
            )
    return targets


def download_json(api: HfApi, repo_id: str, path: str, token: str) -> dict | None:
    try:
        local = hf_hub_download(repo_id, path, repo_type="model", token=token)
        return json.loads(Path(local).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def download_text(api: HfApi, repo_id: str, path: str, token: str) -> str | None:
    try:
        local = hf_hub_download(repo_id, path, repo_type="model", token=token)
        return Path(local).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _redact_paths(value):
    if isinstance(value, dict):
        return {key: _redact_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"/mnt/(?:volume|data)[^/\s]*/[^\s\"']*", "<REDACTED_REMOTE_PATH>", value)
        value = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "<REDACTED_TOKEN>", value)
        value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<REDACTED_IP>", value)
        return value
    return value


def compose_raw_metrics(api: HfApi, token: str, repo_id: str) -> dict:
    """Build schema-v2 payload from Hub files (supports Instruct-v2 split files)."""
    raw = download_json(api, repo_id, "metrics.json", token)
    if not raw:
        raise RuntimeError(f"missing metrics.json on {repo_id}; run family eval first")

    # Already composed by a prior 56 publish.
    if isinstance(raw.get("clm"), dict) or isinstance(raw.get("sft"), dict) or isinstance(raw.get("gguf"), dict):
        return raw

    composed: dict = {}
    clm = download_json(api, repo_id, "metrics_clm.json", token)
    if isinstance(clm, dict) and clm:
        composed["clm"] = _redact_paths(clm)
    if isinstance(raw.get("per_task"), dict):
        composed["sft"] = {
            "per_task": raw["per_task"],
            "qualitative_count": raw.get("qualitative_count"),
            "model": raw.get("model"),
        }
    elif isinstance(raw.get("gguf"), dict):
        composed["gguf"] = raw["gguf"]
    else:
        # Unknown shape — keep as sft blob so publish still surfaces something.
        composed["sft"] = _redact_paths(raw)

    if not composed.get("clm") and not composed.get("sft") and not composed.get("gguf"):
        raise RuntimeError(f"unable to compose evaluation payload for {repo_id}")
    return composed


def publish_one(api: HfApi, token: str, target: dict, *, dry_run: bool) -> dict:
    repo_id = target["hub_repo"]
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing to publish metrics to non-private repo: {repo_id}")

    raw = compose_raw_metrics(api, token, repo_id)

    metrics = normalize_metrics(
        raw,
        hub_repo=repo_id,
        size=target["size"],
        stage=target["stage"],
        is_gguf=target["is_gguf"],
    )
    model_name = repo_id.split("/")[-1]
    model_index = metrics["model-index"]

    readme = download_text(api, repo_id, "README.md", token) or f"# {model_name}\n"
    meta, body = split_readme(readme)
    meta["model-index"] = model_index
    meta.setdefault("language", ["wo"])
    meta.setdefault("pipeline_tag", "text-generation")
    if target["is_gguf"]:
        meta.setdefault("library_name", "llama.cpp")
        meta.setdefault("tags", [])
        if isinstance(meta["tags"], list) and "gguf" not in meta["tags"]:
            meta["tags"] = list(meta["tags"]) + ["gguf"]
    else:
        meta.setdefault("library_name", "transformers")

    section = build_eval_section(metrics=metrics, is_gguf=target["is_gguf"], size=target["size"])
    body = upsert_eval_section(body, section)
    # Drop stale placeholder Metrics blurb if Evaluation already inserted.
    body = re.sub(
        r"(?im)^##\s+Metrics\s*$\nOnly measured outputs are included in `metrics\.json`\..*?(?=^##\s|\Z)",
        "",
        body,
        count=1,
        flags=re.DOTALL,
    )
    new_readme = dump_readme(meta, body)

    result = {"repo": repo_id, "status": "dry-run" if dry_run else "pending"}
    if dry_run:
        result["metrics_keys"] = sorted(metrics.keys())
        result["n_model_index_results"] = len(model_index[0]["results"])
        return result

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        metrics_path = tmp_path / "metrics.json"
        readme_path = tmp_path / "README.md"
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        readme_path.write_text(new_readme, encoding="utf-8")
        commit = api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish HF-compliant evaluation metrics and model-index",
            allow_patterns=["metrics.json", "README.md"],
        )
        result["status"] = "ok"
        result["revision"] = commit.oid
        result["n_model_index_results"] = len(model_index[0]["results"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish HF-compliant metrics to Hub model cards")
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

    load_trackio_env()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN required")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = discover_targets(cfg)
    if args.only_repo:
        wanted = set(args.only_repo)
        targets = [t for t in targets if t["hub_repo"] in wanted]

    api = HfApi(token=token)
    results = []
    failures = []
    for target in targets:
        try:
            rec = publish_one(api, token or "", target, dry_run=args.dry_run)
            results.append(rec)
            print(json.dumps(rec, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            fail = {"repo": target["hub_repo"], "status": "failed", "error": str(exc)}
            failures.append(fail)
            print(json.dumps(fail, ensure_ascii=False), file=sys.stderr)

    report = {
        "updated_at": utc_now(),
        "dry_run": args.dry_run,
        "n_ok": sum(1 for r in results if r.get("status") in {"ok", "dry-run"}),
        "n_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(args.report, report)
    print(f"report={args.report} ok={report['n_ok']} failed={report['n_failed']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
