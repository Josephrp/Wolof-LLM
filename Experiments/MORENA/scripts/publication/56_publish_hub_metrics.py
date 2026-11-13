#!/usr/bin/env python3
"""Publish HF-compliant metrics.json + README model-index for MORENA Wolof repos."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.pipeline import atomic_json_write, redact, utc_now  # noqa: E402

EVAL_MARKER_START = "<!-- FAMILY_EVAL_START -->"
EVAL_MARKER_END = "<!-- FAMILY_EVAL_END -->"


def split_readme(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    return (meta if isinstance(meta, dict) else {}), match.group(2)


def dump_readme(meta: dict, body: str) -> str:
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{body.lstrip()}"


def build_model_index(model_name: str, metrics: dict) -> list[dict]:
    clm = metrics.get("clm") or metrics
    results = [
        {
            "task": {"type": "text-generation", "name": "Wolof causal LM"},
            "dataset": {
                "type": "wolof-clm-corpus-test",
                "name": "Wolof CLM corpus test (verbalized)",
                "split": "test",
            },
            "metrics": [
                {"type": "bpb", "name": "Bits Per Byte", "value": clm.get("BPB")},
                {"type": "bleu", "name": "BLEU", "value": clm.get("BLEU")},
                {"type": "chrf", "name": "chrF", "value": clm.get("chrF")},
            ],
            "source": {
                "name": "GalsenAI MORENA Wolof family eval",
                "url": f"https://huggingface.co/{metrics.get('hub_repo', model_name)}/blob/main/metrics.json",
            },
        }
    ]
    # drop null metrics
    for result in results:
        result["metrics"] = [m for m in result["metrics"] if m.get("value") is not None]
    return [{"name": model_name, "results": results}]


def eval_section(metrics: dict) -> str:
    clm = metrics.get("clm") or metrics
    lines = [
        EVAL_MARKER_START,
        "## Evaluation",
        "",
        "- Protocol: `morena-native` (BPB gold). Tokenizer: MORENA 65k frozen.",
        "- Machine-readable: [`metrics.json`](./metrics.json).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| BPB ↓ | {clm.get('BPB')} |",
        f"| Perplexity | {clm.get('perplexity')} |",
        f"| BLEU | {clm.get('BLEU')} |",
        f"| chrF | {clm.get('chrF')} |",
        "",
        EVAL_MARKER_END,
    ]
    return "\n".join(lines)


def upsert_section(body: str, section: str) -> str:
    pattern = re.compile(
        re.escape(EVAL_MARKER_START) + r".*?" + re.escape(EVAL_MARKER_END),
        flags=re.DOTALL,
    )
    if pattern.search(body):
        return pattern.sub(section, body)
    return body.rstrip() + "\n\n" + section + "\n"


def publish_one(api: HfApi, token: str, repo_id: str, metrics_path: Path, *, dry_run: bool) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = redact(metrics)
    metrics["schema_version"] = 2
    metrics["updated_at"] = utc_now()
    metrics["hub_repo"] = repo_id
    metrics["protocol"] = metrics.get("protocol") or "morena-native"
    model_name = repo_id.split("/")[-1]
    metrics["model-index"] = build_model_index(model_name, metrics)

    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if info.private is not True:
        raise RuntimeError(f"refusing non-private repo: {repo_id}")

    try:
        readme = Path(hf_hub_download(repo_id, "README.md", token=token)).read_text(encoding="utf-8")
    except Exception:
        readme = f"# {model_name}\n\nWolof adaptation of MORENA.\n"

    meta, body = split_readme(readme)
    meta["model-index"] = metrics["model-index"]
    meta.setdefault("language", ["wo"])
    meta.setdefault("pipeline_tag", "text-generation")
    meta.setdefault("library_name", "pytorch")
    meta.setdefault("tags", ["morena", "wolof", "african-languages"])
    body = upsert_section(body, eval_section(metrics))
    new_readme = dump_readme(meta, body)

    if dry_run:
        return {"repo": repo_id, "status": "dry-run"}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (tmp_path / "README.md").write_text(new_readme, encoding="utf-8")
        commit = api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish MORENA Wolof evaluation metrics",
            allow_patterns=["metrics.json", "README.md"],
        )
    return {"repo": repo_id, "status": "ok", "revision": commit.oid}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
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
    api = HfApi(token=token)
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    results = []
    failures = []
    for entry in manifest.get("uploads") or []:
        try:
            rec = publish_one(
                api,
                token or "",
                entry["hub_repo"],
                Path(entry["metrics"]),
                dry_run=args.dry_run,
            )
            results.append(rec)
            print(json.dumps(rec))
        except Exception as exc:  # noqa: BLE001
            fail = {"repo": entry["hub_repo"], "status": "failed", "error": str(exc)}
            failures.append(fail)
            print(json.dumps(fail), file=sys.stderr)
    report = {
        "updated_at": utc_now(),
        "n_ok": len(results),
        "n_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    atomic_json_write(args.report, report)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
