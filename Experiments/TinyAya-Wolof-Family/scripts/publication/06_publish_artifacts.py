#!/usr/bin/env python3
"""Publish Tiny Aya artifacts to private Hub as soon as they exist (incremental).

Model uploads mirror LFM2.5 Hub publish: attach metrics_clm.json (+ optional
original-test secondary), compose metrics.json via 43_prepare_artifact, then
run 56_publish_hub_metrics for HF model-index + Evaluation README section.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.eval_gate import is_full_clm
from common.pipeline import utc_now

LOCAL_PATH = re.compile(
    r"(?:/mnt/[^\s\"']+|/home/[^\s\"']+|[A-Za-z]:\\[^\s\"']+|<REDACTED_PRIVATE_ROOT>/[^\s\"']+)"
)
TEXT_SUFFIXES = {".md", ".json", ".txt", ".yaml", ".yml", ".jinja"}

WHICH = [
    "datasets",
    "datasets_std",
    "prefs",
    "prefs_std",
    "prepared",
    "prepared_std",
    "fertility",
    "tokenizer",
    "ext_init",
    "earth_ext_init",
    "global_ext_init",
    "cpt_adapter",
    "cpt_adapter_std",
    "cpt_merged",
    "cpt_merged_std",
    "instruct_earth",
    "instruct_earth_std",
    "instruct_global",
    "instruct_global_std",
    "instruct_cpt",
    "instruct_cpt_std",
    "dpo",
    "dpo_std",
    "bpb",
    "sft_eval",
    "state",
    "family_eval",
    "all",
]

# Map --which model keys → eval_family.yaml section + role.
MODEL_WHICH_TO_EVAL: dict[str, tuple[str, str]] = {
    "cpt_merged": ("cpt", "cpt"),
    "instruct_cpt": ("instruct", "from_cpt"),
    "instruct_earth": ("instruct", "from_earth"),
    "instruct_global": ("instruct", "from_global"),
    "dpo": ("dpo", "dpo"),
    "cpt_merged_std": ("cpt_std", "cpt"),
    "instruct_cpt_std": ("instruct_std", "from_cpt"),
    "instruct_earth_std": ("instruct_std", "from_earth"),
    "instruct_global_std": ("instruct_std", "from_global"),
    "dpo_std": ("dpo_std", "dpo"),
}

STAGE_FOR_WHICH = {
    "cpt_merged": "CPT",
    "cpt_merged_std": "CPT",
    "instruct_cpt": "Instruct",
    "instruct_cpt_std": "Instruct",
    "instruct_earth": "Instruct",
    "instruct_earth_std": "Instruct",
    "instruct_global": "Instruct",
    "instruct_global_std": "Instruct",
    "dpo": "DPO",
    "dpo_std": "DPO",
    "ext_init": "Init",
    "earth_ext_init": "Init",
    "global_ext_init": "Init",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish Tiny Aya Wolof artifacts")
    parser.add_argument("--namespace", default="Tonic")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--which", nargs="+", default=["state"], choices=WHICH)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--allow-partial-clm",
        action="store_true",
        help="Allow Hub metrics attach without BLEU/chrF (legacy lightweight BPB only).",
    )
    return parser


def load_eval_cfg() -> dict:
    path = ROOT / "configs" / "publication" / "eval_family.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_eval_entry(cfg: dict, section: str, role: str) -> dict | None:
    for entry in cfg.get(section, []) or []:
        if entry.get("role") == role:
            return entry
    return None


def resolve_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def first_existing(*candidates: Path | None) -> Path | None:
    for cand in candidates:
        if cand is not None and cand.exists():
            return cand
    return None


def hub_base_for_local_path(path: str, *, namespace: str) -> str:
    """Map local base_model paths to Hub ids Hub metadata validation accepts."""
    lower = path.replace("\\", "/").lower()
    if "earth-ext-init" in lower or "earth_ext_init" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-earth-ext-init-v1"
    if "global-ext-init" in lower or "global_ext_init" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-global-ext-init-v1"
    if "ext-init" in lower or "ext_init" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-ext-init-v1"
    if "cpt-std-v1-merged" in lower or "cpt_std" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-CPT-std-v1"
    if "cpt-v1-merged" in lower or "cpt_merged" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-CPT-v1"
    if "instruct-from-earth" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-InstructEarth-v1"
    if "instruct-from-global" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-InstructGlobal-v1"
    if "instruct-from-cpt" in lower:
        return f"{namespace}/TinyAya-3.35B-Wolof-Instruct-v1"
    return "CohereLabs/tiny-aya-base"


def rewrite_local_base_models(text: str, *, namespace: str) -> str:
    local = LOCAL_PATH.pattern
    text = re.sub(
        rf"^base_model:\s*({local})$",
        lambda m: f"base_model: {hub_base_for_local_path(m.group(1), namespace=namespace)}",
        text,
        flags=re.M,
    )
    text = re.sub(
        rf"base_model:adapter:({local})",
        lambda m: (
            "base_model:adapter:"
            + hub_base_for_local_path(m.group(1), namespace=namespace)
        ),
        text,
    )
    text = re.sub(
        rf'"base_model_name_or_path"\s*:\s*"({local})"',
        lambda m: (
            '"base_model_name_or_path": "'
            + hub_base_for_local_path(m.group(1), namespace=namespace)
            + '"'
        ),
        text,
    )
    return text


@contextlib.contextmanager
def sanitized_staging(source: Path, *, namespace: str):
    """Hard-link weights; rewrite Hub-invalid local base_model paths in text metadata."""
    staging_root = ROOT / "results" / ".publish_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
        staged = Path(temporary) / source.name
        shutil.copytree(source, staged, copy_function=os.link)
        for path in staged.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.stat().st_size > 10 * 1024 * 1024:
                continue
            original = path.read_text(encoding="utf-8", errors="replace")
            redacted = rewrite_local_base_models(original, namespace=namespace)
            if redacted != original:
                path.unlink()
                path.write_text(redacted, encoding="utf-8")
        yield staged


def publish_folder(
    api: HfApi,
    local: Path,
    repo_id: str,
    *,
    repo_type: str,
    private: bool,
    namespace: str = "Tonic",
) -> None:
    if not local.exists():
        print(f"SKIP missing {local}", flush=True)
        return
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    info = api.repo_info(repo_id, repo_type=repo_type)
    if private and getattr(info, "private", None) is not True:
        raise RuntimeError(f"refusing non-private repo {repo_id}")
    if local.is_file():
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=local.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {local.name} ({utc_now()})",
        )
    elif repo_type == "model":
        with sanitized_staging(local, namespace=namespace) as staged:
            api.upload_folder(
                folder_path=str(staged),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload {local.name} ({utc_now()})",
            )
    else:
        api.upload_folder(
            folder_path=str(local),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {local.name} ({utc_now()})",
        )
    print(f"published {local} -> {repo_id} ({repo_type})", flush=True)


def _py() -> list[str]:
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.is_file():
        return [str(venv)]
    return [sys.executable]


def compose_stub_metrics(*, clm: dict | None, sft: dict | None, hub_repo: str, stage: str) -> dict:
    payload: dict = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "hub_repo": hub_repo,
        "stage": stage,
        "size": "3b35",
        "license_note": "cc-by-nc-4.0 + Cohere Labs AUP",
    }
    if clm:
        payload["clm"] = clm
    if sft:
        payload["sft"] = {
            "per_task": sft.get("per_task"),
            "mean_loss": sft.get("mean_loss"),
            "qualitative_count": sft.get("qualitative_count"),
            "model": sft.get("model"),
        }
    return payload


def attach_and_publish_model_metrics(
    api: HfApi,
    *,
    which_key: str,
    local: Path,
    repo_id: str,
    private: bool,
    eval_cfg: dict,
    skip_metrics: bool,
    namespace: str = "Tonic",
) -> None:
    """Upload model folder, then attach CLM/SFT metrics + model-index when present."""
    if not local.is_dir():
        print(f"SKIP missing model dir {local}", flush=True)
        return

    section_role = MODEL_WHICH_TO_EVAL.get(which_key)
    entry = find_eval_entry(eval_cfg, *section_role) if section_role else None
    stage = (entry or {}).get("stage") or STAGE_FOR_WHICH.get(which_key, "Instruct")
    base_model = (entry or {}).get("base_model") or "CohereLabs/tiny-aya-base"

    clm_path = resolve_path((entry or {}).get("clm_metrics"))
    clm_secondary = resolve_path((entry or {}).get("clm_metrics_secondary"))
    sft_path = resolve_path((entry or {}).get("sft_eval"))
    manifest = resolve_path((entry or {}).get("run_manifest"))

    # Fallbacks for journals / smoke runs.
    # Canonical on-disk names use ortho: cpt_{ns|std}_3b35.json,
    # instruct_{role}_{ns|std}_3b35.json, eval_{3b35|3b35_std}_{role}/.
    if which_key in {"cpt_merged", "cpt_merged_std"}:
        std = which_key.endswith("_std")
        ortho = "std" if std else "ns"
        manifest = first_existing(
            manifest,
            ROOT
            / "runs"
            / "clm"
            / "3b35"
            / ("v1_ntp70_ortho_std_lora" if std else "v1_ntp70_ortho_lora")
            / "journal_cpt.json",
            ROOT
            / "results"
            / "smoke"
            / ("tinyaya_3b35_cpt_std_smoke-cpt" if std else "tinyaya_3b35_cpt_smoke-cpt")
            / "journal_cpt.json",
        )
        clm_path = first_existing(
            clm_path,
            ROOT / "results" / "bpb" / f"clm_cpt_{ortho}_3b35.json",
            ROOT / "results" / "bpb" / f"cpt_{ortho}_3b35.json",
            ROOT / "results" / "bpb" / "cpt_3b35.json",
        )
        clm_secondary = first_existing(
            clm_secondary,
            ROOT / "results" / "bpb" / f"cpt_{ortho}_3b35_original_test.json",
            ROOT / "results" / "bpb" / "cpt_3b35_original_test.json",
        )
    elif which_key.startswith("instruct_"):
        # Strip ortho suffix so instruct_earth_std → instruct_earth.
        base_key = which_key[:-4] if which_key.endswith("_std") else which_key
        std = which_key.endswith("_std")
        ortho = "std" if std else "ns"
        role = {
            "instruct_cpt": "from_cpt",
            "instruct_earth": "from_earth",
            "instruct_global": "from_global",
        }.get(base_key)
        if role is None:
            raise KeyError(f"unknown instruct which_key={which_key!r}")
        run_dirs = {
            "from_cpt": "3b35_from_cpt_std_v1" if std else "3b35_from_cpt_v1",
            "from_earth": "3b35_from_earth_std_v1" if std else "3b35_from_earth_v1",
            "from_global": "3b35_from_global_std_v1" if std else "3b35_from_global_v1",
        }
        smoke_run = f"tinyaya_3b35_sft_{role}{'_std' if std else ''}_v1-sft"
        prefix = "3b35_std" if std else "3b35"
        manifest = first_existing(
            manifest,
            ROOT / "runs" / "sft" / run_dirs[role] / "journal_sft.json",
            ROOT / "results" / "smoke" / smoke_run / "journal_sft.json",
        )
        sft_path = first_existing(
            sft_path,
            ROOT / "results" / "sft" / f"eval_{prefix}_{role}" / "evaluation.json",
            ROOT / "results" / "sft" / f"eval_{role}{'_std' if std else ''}" / "evaluation.json",
            ROOT / "results" / "sft" / f"eval_{role}" / "evaluation.json",
        )
        clm_path = first_existing(
            clm_path,
            ROOT / "results" / "bpb" / f"clm_{role}_{ortho}_3b35.json",
            ROOT / "results" / "bpb" / f"instruct_{role}_{ortho}_3b35.json",
            ROOT / "results" / "bpb" / f"instruct_{role}_3b35.json",
        )
    elif which_key in {"dpo", "dpo_std"}:
        std = which_key.endswith("_std")
        ortho = "std" if std else "ns"
        prefix = "3b35_std" if std else "3b35"
        run_dir = "3b35_std_dpo_v1" if std else "3b35_dpo_v1"
        manifest = first_existing(
            manifest,
            ROOT / "runs" / "sft" / run_dir / "journal_dpo.json",
            ROOT
            / "results"
            / "smoke"
            / (f"tinyaya_{run_dir}-dpo")
            / "journal_dpo.json",
        )
        sft_path = first_existing(
            sft_path,
            ROOT / "results" / "sft" / f"eval_{prefix}_dpo" / "evaluation.json",
            ROOT / "results" / "sft" / f"eval_dpo{'_std' if std else ''}" / "evaluation.json",
            ROOT / "results" / "sft" / "eval_dpo" / "evaluation.json",
        )
        clm_path = first_existing(
            clm_path,
            ROOT / "results" / "bpb" / f"clm_dpo_{ortho}_3b35.json",
            ROOT / "results" / "bpb" / f"dpo_{ortho}_3b35.json",
            ROOT / "results" / "bpb" / "dpo_3b35.json",
        )

    clm_raw = None
    if clm_path and clm_path.is_file():
        clm_raw = json.loads(clm_path.read_text(encoding="utf-8"))
    sft_raw = None
    if sft_path and sft_path.is_file():
        sft_raw = json.loads(sft_path.read_text(encoding="utf-8"))

    require_full = os.environ.get("TINYAYA_REQUIRE_FULL_CLM", "1") not in {
        "0",
        "false",
        "False",
    }
    # Model which-keys that must carry gold evaluate_clm metrics before Hub metrics attach.
    needs_full_clm = which_key in {
        "cpt_merged",
        "cpt_merged_std",
        "instruct_earth",
        "instruct_earth_std",
        "instruct_global",
        "instruct_global_std",
        "instruct_cpt",
        "instruct_cpt_std",
        "dpo",
        "dpo_std",
    }
    if (
        require_full
        and needs_full_clm
        and clm_raw is not None
        and not is_full_clm(clm_raw)
        and not skip_metrics
    ):
        print(
            f"METRICS_BLOCK {repo_id}: CLM incomplete "
            f"(need BPB+BLEU+chrF+perplexity; got keys={sorted(clm_raw)}). "
            "Run scripts/publication/55_run_eval_family.py before Hub metrics.",
            flush=True,
        )
        clm_raw = None

    has_metrics = bool(clm_raw or sft_raw)
    prepared_ok = False

    if not skip_metrics and has_metrics:
        stub_dir = ROOT / "results" / "full_pipeline" / "publish_stubs"
        stub_dir.mkdir(parents=True, exist_ok=True)
        metrics_stub = stub_dir / f"{repo_id.replace('/', '__')}_metrics.json"
        metrics_stub.write_text(
            json.dumps(
                compose_stub_metrics(
                    clm=clm_raw, sft=sft_raw, hub_repo=repo_id, stage=stage
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if manifest is None or not manifest.is_file():
            # Minimal training_manifest so 43 can still attach license card.
            manifest = stub_dir / f"{repo_id.replace('/', '__')}_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "created_at": utc_now(),
                        "hub_repo": repo_id,
                        "stage": stage,
                        "note": "synthetic manifest; training journal not found yet",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        try:
            subprocess.run(
                [
                    *_py(),
                    str(ROOT / "scripts" / "publication" / "43_prepare_artifact.py"),
                    "--artifact",
                    str(local),
                    "--metrics",
                    str(metrics_stub),
                    "--run-manifest",
                    str(manifest),
                    "--model-name",
                    repo_id.split("/")[-1],
                    "--base-model",
                    base_model,
                    "--training-stage",
                    stage
                    if stage
                    in {
                        "CPT",
                        "Instruct",
                        "InstructEarth",
                        "InstructGlobal",
                        "DPO",
                        "Base",
                        "Init",
                    }
                    else "Instruct",
                ],

                cwd=ROOT,
                check=True,
            )
            prepared_ok = True
        except subprocess.CalledProcessError as exc:
            print(f"PREPARE_ARTIFACT_FAIL {repo_id}: {exc}", flush=True)

    # Always upload weights (and any prepared card/metrics already in folder).
    publish_folder(
        api, local, repo_id, repo_type="model", private=private, namespace=namespace
    )

    if skip_metrics or not has_metrics:
        print(f"METRICS_SKIP {repo_id} (no local BPB/SFT yet or --skip-metrics)", flush=True)
        return

    # Upload dedicated CLM metric files (LFM naming).
    if clm_path and clm_path.is_file():
        api.upload_file(
            path_or_fileobj=str(clm_path),
            path_in_repo="metrics_clm.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add primary (ortho_ns_mixed) CLM metrics",
        )
        print(f"uploaded metrics_clm.json -> {repo_id}", flush=True)
    if clm_secondary and clm_secondary.is_file():
        api.upload_file(
            path_or_fileobj=str(clm_secondary),
            path_in_repo="metrics_clm_original_test.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add secondary (original clean test) CLM metrics",
        )
        print(f"uploaded metrics_clm_original_test.json -> {repo_id}", flush=True)
    if sft_path and sft_path.is_file():
        # Keep raw SFT evaluation alongside composed metrics.json.
        api.upload_file(
            path_or_fileobj=str(sft_path),
            path_in_repo="metrics_sft_evaluation.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add SFT evaluation.json",
        )

    try:
        subprocess.run(
            [
                *_py(),
                str(ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"),
                "--only-repo",
                repo_id,
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"HUB_METRICS_OK {repo_id} prepared={prepared_ok}", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"HUB_METRICS_FAIL {repo_id}: {exc}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.allow_partial_clm:
        os.environ["TINYAYA_REQUIRE_FULL_CLM"] = "0"
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("HF_TOKEN required")
    api = HfApi(token=token)
    private = not args.public
    ns = args.namespace
    which = set(args.which)
    if "all" in which:
        which = set(WHICH) - {"all"}

    eval_cfg = load_eval_cfg()
    catalog: list[tuple[Path, str, str]] = []
    model_jobs: list[tuple[str, Path, str]] = []

    if "datasets" in which:
        catalog.extend(
            [
                (
                    ROOT / "data" / "ortho_ns_mixed_v1" / "cpt",
                    f"{ns}/TinyAya-Wolof-NTP-ortho_ns_mixed_v1",
                    "dataset",
                ),
                (
                    ROOT / "data" / "ortho_ns_mixed_v1" / "sft_stratified",
                    f"{ns}/TinyAya-Wolof-SFT-ortho_ns_mixed_stratified_v1",
                    "dataset",
                ),
            ]
        )
    if "datasets_std" in which:
        catalog.extend(
            [
                (
                    ROOT / "data" / "ortho_std_mixed_v1" / "cpt",
                    f"{ns}/TinyAya-Wolof-NTP-ortho_std_mixed_v1",
                    "dataset",
                ),
                (
                    ROOT / "data" / "ortho_std_mixed_v1" / "sft_stratified",
                    f"{ns}/TinyAya-Wolof-SFT-ortho_std_mixed_stratified_v1",
                    "dataset",
                ),
            ]
        )
    if "prepared" in which:
        catalog.append(
            (
                ROOT / "data" / "sft" / "prepared_anchored_v1",
                f"{ns}/TinyAya-Wolof-SFT-prepared_anchored_v1",
                "dataset",
            )
        )
    if "prepared_std" in which:
        catalog.append(
            (
                ROOT / "data" / "sft" / "prepared_anchored_std_v1",
                f"{ns}/TinyAya-Wolof-SFT-prepared_anchored_std_v1",
                "dataset",
            )
        )
    if "prefs" in which:
        catalog.append(
            (
                ROOT / "data" / "sft" / "preferences_v1.parquet",
                f"{ns}/TinyAya-Wolof-prefs_v1",
                "dataset",
            )
        )
    if "prefs_std" in which:
        catalog.append(
            (
                ROOT / "data" / "sft" / "preferences_std_v1.parquet",
                f"{ns}/TinyAya-Wolof-prefs_std_v1",
                "dataset",
            )
        )
    if "fertility" in which:
        fert = ROOT / "results" / "tokenizer" / "fertility_probe.json"
        if not fert.is_file():
            for cand in (
                ROOT / "results" / "fertility" / "stock_tinyaya.json",
                ROOT / "results" / "fertility_probe.json",
            ):
                if cand.is_file():
                    fert = cand
                    break
        catalog.append((fert, f"{ns}/TinyAya-Wolof-pipeline-artifacts-v1", "dataset"))
    if "tokenizer" in which:
        catalog.append(
            (
                ROOT / "tokenizers" / "tinyaya-wolof-ext-anchors",
                f"{ns}/TinyAya-Wolof-tokenizer-ext-anchors-v1",
                "model",
            )
        )
    if "ext_init" in which:
        # Init checkpoint: upload weights only (metrics optional / usually absent).
        catalog.append(
            (
                ROOT / "models" / "tinyaya-3.35b-ext-init",
                f"{ns}/TinyAya-3.35B-Wolof-ext-init-v1",
                "model",
            )
        )
    if "earth_ext_init" in which:
        catalog.append(
            (
                ROOT / "models" / "tinyaya-3.35b-earth-ext-init",
                f"{ns}/TinyAya-3.35B-Wolof-earth-ext-init-v1",
                "model",
            )
        )
    if "global_ext_init" in which:
        catalog.append(
            (
                ROOT / "models" / "tinyaya-3.35b-global-ext-init",
                f"{ns}/TinyAya-3.35B-Wolof-global-ext-init-v1",
                "model",
            )
        )
    if "cpt_adapter" in which:
        for cand in (
            ROOT / "results" / "smoke" / "tinyaya_3b35_cpt_smoke-cpt" / "adapter",
            ROOT / "runs" / "clm" / "3b35" / "v1_ntp70_ortho_lora" / "adapter",
        ):
            if cand.is_dir():
                catalog.append(
                    (cand, f"{ns}/TinyAya-3.35B-Wolof-CPT-adapter-v1", "model")
                )
                break
    if "cpt_adapter_std" in which:
        for cand in (
            ROOT / "results" / "smoke" / "tinyaya_3b35_cpt_std_smoke-cpt" / "adapter",
            ROOT / "runs" / "clm" / "3b35" / "v1_ntp70_ortho_std_lora" / "adapter",
        ):
            if cand.is_dir():
                catalog.append(
                    (cand, f"{ns}/TinyAya-3.35B-Wolof-CPT-adapter-std-v1", "model")
                )
                break
    if "cpt_merged" in which:
        model_jobs.append(
            (
                "cpt_merged",
                ROOT / "models" / "tinyaya-3.35b-cpt-v1-merged",
                f"{ns}/TinyAya-3.35B-Wolof-CPT-v1",
            )
        )
    if "cpt_merged_std" in which:
        model_jobs.append(
            (
                "cpt_merged_std",
                ROOT / "models" / "tinyaya-3.35b-cpt-std-v1-merged",
                f"{ns}/TinyAya-3.35B-Wolof-CPT-std-v1",
            )
        )
    if "instruct_earth" in which:
        model_jobs.append(
            (
                "instruct_earth",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-earth-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructEarth-v1",
            )
        )
    if "instruct_earth_std" in which:
        model_jobs.append(
            (
                "instruct_earth_std",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-earth-std-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructEarth-std-v1",
            )
        )
    if "instruct_global" in which:
        model_jobs.append(
            (
                "instruct_global",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-global-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructGlobal-v1",
            )
        )
    if "instruct_global_std" in which:
        model_jobs.append(
            (
                "instruct_global_std",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-global-std-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructGlobal-std-v1",
            )
        )
    if "instruct_cpt" in which:
        model_jobs.append(
            (
                "instruct_cpt",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-cpt-merged",
                f"{ns}/TinyAya-3.35B-Wolof-Instruct-v1",
            )
        )
    if "instruct_cpt_std" in which:
        model_jobs.append(
            (
                "instruct_cpt_std",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-cpt-std-merged",
                f"{ns}/TinyAya-3.35B-Wolof-Instruct-std-v1",
            )
        )
    if "dpo" in which:
        for cand in (
            ROOT / "models" / "tinyaya-3.35b-dpo-v1-merged",
            ROOT / "runs" / "sft" / "3b35_dpo_v1" / "adapter",
        ):
            if cand.exists():
                model_jobs.append(("dpo", cand, f"{ns}/TinyAya-3.35B-Wolof-DPO-v1"))
                break
    if "dpo_std" in which:
        for cand in (
            ROOT / "models" / "tinyaya-3.35b-dpo-std-v1-merged",
            ROOT / "runs" / "sft" / "3b35_std_dpo_v1" / "adapter",
        ):
            if cand.exists():
                model_jobs.append(("dpo_std", cand, f"{ns}/TinyAya-3.35B-Wolof-DPO-std-v1"))
                break
    if "bpb" in which:
        catalog.append(
            (
                ROOT / "results" / "bpb",
                f"{ns}/TinyAya-Wolof-pipeline-artifacts-v1",
                "dataset",
            )
        )
    if "sft_eval" in which:
        catalog.append(
            (
                ROOT / "results" / "sft",
                f"{ns}/TinyAya-Wolof-pipeline-artifacts-v1",
                "dataset",
            )
        )
    if "state" in which:
        catalog.append(
            (
                ROOT / "results" / "full_pipeline" / "tinyaya_state.json",
                f"{ns}/TinyAya-Wolof-pipeline-artifacts-v1",
                "dataset",
            )
        )
    if "family_eval" in which:
        catalog.append(
            (
                ROOT / "docs" / "FAMILY_EVAL.md",
                f"{ns}/TinyAya-Wolof-pipeline-artifacts-v1",
                "dataset",
            )
        )

    for local, repo_id, repo_type in catalog:
        publish_folder(
            api,
            local,
            repo_id,
            repo_type=repo_type,
            private=private,
            namespace=ns,
        )

    for which_key, local, repo_id in model_jobs:
        attach_and_publish_model_metrics(
            api,
            which_key=which_key,
            local=local,
            repo_id=repo_id,
            private=private,
            eval_cfg=eval_cfg,
            skip_metrics=args.skip_metrics,
            namespace=ns,
        )

    # After BPB/SFT artifact dataset upload, refresh model-index on any model that now has metrics.
    if not args.skip_metrics and ({"bpb", "sft_eval", "family_eval"} & which):
        for which_key, local, repo_id in (
            (
                "cpt_merged",
                ROOT / "models" / "tinyaya-3.35b-cpt-v1-merged",
                f"{ns}/TinyAya-3.35B-Wolof-CPT-v1",
            ),
            (
                "cpt_merged_std",
                ROOT / "models" / "tinyaya-3.35b-cpt-std-v1-merged",
                f"{ns}/TinyAya-3.35B-Wolof-CPT-std-v1",
            ),
            (
                "instruct_earth",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-earth-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructEarth-v1",
            ),
            (
                "instruct_earth_std",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-earth-std-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructEarth-std-v1",
            ),
            (
                "instruct_global",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-global-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructGlobal-v1",
            ),
            (
                "instruct_global_std",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-global-std-merged",
                f"{ns}/TinyAya-3.35B-Wolof-InstructGlobal-std-v1",
            ),
            (
                "instruct_cpt",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-cpt-merged",
                f"{ns}/TinyAya-3.35B-Wolof-Instruct-v1",
            ),
            (
                "instruct_cpt_std",
                ROOT / "models" / "tinyaya-3.35b-instruct-from-cpt-std-merged",
                f"{ns}/TinyAya-3.35B-Wolof-Instruct-std-v1",
            ),
            (
                "dpo",
                ROOT / "models" / "tinyaya-3.35b-dpo-v1-merged",
                f"{ns}/TinyAya-3.35B-Wolof-DPO-v1",
            ),
            (
                "dpo_std",
                ROOT / "models" / "tinyaya-3.35b-dpo-std-v1-merged",
                f"{ns}/TinyAya-3.35B-Wolof-DPO-std-v1",
            ),
        ):
            if which_key in {j[0] for j in model_jobs}:
                continue
            if not local.is_dir():
                continue
            # Only re-run 56 if metrics files exist (no full re-upload of weights).
            entry = None
            section_role = MODEL_WHICH_TO_EVAL.get(which_key)
            if section_role:
                entry = find_eval_entry(eval_cfg, *section_role)
            clm = resolve_path((entry or {}).get("clm_metrics"))
            sft = resolve_path((entry or {}).get("sft_eval"))
            if not ((clm and clm.is_file()) or (sft and sft.is_file())):
                continue
            if clm and clm.is_file():
                api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
                api.upload_file(
                    path_or_fileobj=str(clm),
                    path_in_repo="metrics_clm.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Refresh primary (ortho_ns_mixed) CLM metrics",
                )
            sec = resolve_path((entry or {}).get("clm_metrics_secondary"))
            if sec and sec.is_file():
                api.upload_file(
                    path_or_fileobj=str(sec),
                    path_in_repo="metrics_clm_original_test.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Refresh secondary (original clean test) CLM metrics",
                )
            if sft and sft.is_file():
                api.upload_file(
                    path_or_fileobj=str(sft),
                    path_in_repo="metrics_sft_evaluation.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Refresh SFT evaluation.json",
                )
            # Ensure metrics.json exists for 56 compose (copy stub if missing on Hub).
            stub = compose_stub_metrics(
                clm=json.loads(clm.read_text(encoding="utf-8")) if clm and clm.is_file() else None,
                sft=json.loads(sft.read_text(encoding="utf-8")) if sft and sft.is_file() else None,
                hub_repo=repo_id,
                stage=(entry or {}).get("stage") or STAGE_FOR_WHICH.get(which_key, "Instruct"),
            )
            tmp = ROOT / "results" / "full_pipeline" / "publish_stubs" / f"{repo_id.replace('/', '__')}_refresh.json"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(stub, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            api.upload_file(
                path_or_fileobj=str(tmp),
                path_in_repo="metrics.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Seed metrics.json for model-index compose",
            )
            try:
                subprocess.run(
                    [
                        *_py(),
                        str(ROOT / "scripts" / "publication" / "56_publish_hub_metrics.py"),
                        "--only-repo",
                        repo_id,
                    ],
                    cwd=ROOT,
                    check=True,
                )
                print(f"HUB_METRICS_REFRESH_OK {repo_id}", flush=True)
            except subprocess.CalledProcessError as exc:
                print(f"HUB_METRICS_REFRESH_FAIL {repo_id}: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
