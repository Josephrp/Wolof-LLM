#!/usr/bin/env python3
"""Durable gated GGUF family runner (G1–G6) over all published Wolof models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.pipeline import atomic_json_write

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoTokenizer = None


def load_trackio_env() -> None:
    env_path = ROOT / ".env.trackio"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gated GGUF conversion family")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "publication" / "gguf_family.yaml",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "gguf_state.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "full_pipeline" / "gguf_report.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Optional size/stage filter like 230m:Instruct or 8b-a1b",
    )
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--cleanup-after-publish",
        action="store_true",
        help="Delete local *.gguf after Hub publish (keeps meta JSON). "
        "Also enabled when config cleanup_after_publish: true.",
    )
    return parser


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_py(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "publication" / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_state(path: Path, resume: bool) -> dict:
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "pending",
        "updated_at": utc_now(),
        "models": {},
        "gates": {},
    }


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, state)


def model_key(model: dict) -> str:
    return f"{model['size']}:{model['stage']}"


def selected(models: list[dict], only: str | None) -> list[dict]:
    if not only:
        return models
    out = []
    for model in models:
        key = model_key(model)
        if only == model["size"] or only == key or only.lower() == model["stage"].lower():
            out.append(model)
    if not out:
        raise ValueError(f"--only matched no models: {only}")
    return out


def detect_actual_tie(artifact: Path) -> bool:
    weight = artifact / "model.safetensors"
    if safe_open is None or not weight.is_file():
        return True
    with safe_open(weight, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        emb = next(k for k in keys if k.endswith("embed_tokens.weight"))
        lm = next((k for k in keys if k.endswith("lm_head.weight")), None)
        if lm is None:
            return True
        emb_t = handle.get_tensor(emb)
        lm_t = handle.get_tensor(lm)
        mid = emb_t.shape[0] // 2
        return bool(
            (lm_t[0] == emb_t[0]).all()
            and (lm_t[-1] == emb_t[-1]).all()
            and (lm_t[mid] == emb_t[mid]).all()
        )


def correct_tie_flag(artifact: Path) -> bool:
    """Return True if config was changed."""
    if AutoConfig is None:
        raise RuntimeError("transformers is required for tie-flag preflight")
    cfg_path = artifact / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    actual = detect_actual_tie(artifact)
    configured = bool(cfg.get("tie_word_embeddings", True))
    if configured == actual:
        return False
    conf = AutoConfig.from_pretrained(artifact, trust_remote_code=True)
    conf.tie_word_embeddings = actual
    conf.save_pretrained(artifact)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tie_word_embeddings"] = actual
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    meta = artifact / "sft_merge_meta.json"
    if meta.is_file():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        payload["tie_after_merge"] = actual
        payload["tie_config_corrected"] = True
        meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def preflight_model(model: dict) -> dict:
    if AutoTokenizer is None or AutoConfig is None:
        raise RuntimeError("transformers is required for preflight")
    artifact = ROOT / model["artifact"]
    if not artifact.is_dir():
        raise FileNotFoundError(f"missing artifact: {artifact}")
    for required in ("config.json", "tokenizer.json", "model.safetensors"):
        if not (artifact / required).is_file():
            raise FileNotFoundError(f"{artifact} missing {required}")
    tie_fixed = False
    if model["stage"] == "Instruct":
        tie_fixed = correct_tie_flag(artifact)
    tok = AutoTokenizer.from_pretrained(artifact, trust_remote_code=True)
    cfg = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    eos = tok.eos_token_id
    expected_eos = int(model["expected_eos"])
    if eos != expected_eos:
        raise ValueError(f"{artifact}: eos {eos} != expected {expected_eos}")
    vocab = int(cfg.get("vocab_size", 0))
    if vocab < len(tok):
        raise ValueError(f"{artifact}: vocab_size {vocab} < tokenizer {len(tok)}")
    return {
        "artifact": str(artifact),
        "tok_len": len(tok),
        "vocab_size": vocab,
        "eos_token_id": eos,
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "tie_fixed": tie_fixed,
        "arch": cfg.get("model_type") or cfg.get("architectures"),
    }


def write_card(output_dir: Path, model: dict) -> None:
    text = f"""# {model['file_stem']} GGUF

Private production GGUF export.

- Base Hub model: `{model['base_repo']}`
- Stage: `{model['stage']}`
- Tokenizer family: `{model['tokenizer_family']}`
- Ongoing/experimental private production artifact
- Exact base, data caveat, limitations: see `{model['base_repo']}`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def cleanup_local_ggufs(output_dir: Path) -> list[str]:
    """Remove heavy local GGUF blobs after Hub publish; keep sidecars."""
    removed: list[str] = []
    for path in sorted(output_dir.glob("*.gguf")):
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def process_model(
    model: dict,
    *,
    cfg: dict,
    state: dict,
    skip_publish: bool,
    cleanup_after_publish: bool,
) -> dict:
    key = model_key(model)
    record = state["models"].setdefault(key, {"status": "pending"})
    output_dir = ROOT / cfg["output_root"] / model["size"] / model["stage"].lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_card(output_dir, model)

    if record.get("status") == "complete" and (output_dir / "gguf_meta.json").is_file():
        # Resume: if publish already done, allow cleanup of leftover blobs.
        if cleanup_after_publish and not skip_publish:
            removed = cleanup_local_ggufs(output_dir)
            if removed:
                record["cleaned_ggufs"] = removed
                print(f"cleaned leftover {key}: {removed}", flush=True)
        print(f"skip complete {key}", flush=True)
        return record

    record["status"] = "running"
    record["started_at"] = utc_now()
    try:
        pre = preflight_model(model)
        record["preflight"] = pre
        quants = ",".join(cfg["quant_set"])
        run_py(
            "51_convert_gguf.py",
            "--artifact",
            model["artifact"],
            "--output-dir",
            str(output_dir.relative_to(ROOT)),
            "--file-stem",
            model["file_stem"],
            "--llama-root",
            cfg["llamacpp_root"],
            "--quants",
            quants,
        )
        verify_quant = cfg.get("verify_quant", "Q4_K_M")
        gguf_path = output_dir / f"{model['file_stem']}-{verify_quant}.gguf"
        run_py(
            "52_verify_gguf.py",
            "--gguf",
            str(gguf_path.relative_to(ROOT)),
            "--prompt",
            cfg.get("smoke_prompt", "Salaam aleekum"),
            "--n-predict",
            str(cfg.get("smoke_n_predict", 48)),
            "--out",
            str((output_dir / "verify.json").relative_to(ROOT)),
        )
        if not skip_publish:
            publish_args = [
                "53_publish_gguf.py",
                "--gguf-dir",
                str(output_dir.relative_to(ROOT)),
                "--size",
                model["size"],
                "--stage",
                model["stage"],
                "--repo-name",
                model["gguf_repo_name"],
                "--base-repo",
                model["base_repo"],
                "--file-stem",
                model["file_stem"],
                "--preferred-namespace",
                cfg.get("namespace_preferred", "Tonic"),
            ]
            if cfg.get("publication_manifest"):
                publish_args.extend(["--manifest", str(cfg["publication_manifest"])])
            run_py(*publish_args)
            if cleanup_after_publish:
                removed = cleanup_local_ggufs(output_dir)
                record["cleaned_ggufs"] = removed
                print(f"cleaned after publish {key}: {removed}", flush=True)
        record["status"] = "complete"
        record["completed_at"] = utc_now()
        record["output_dir"] = str(output_dir.relative_to(ROOT))
        record["verify_quant"] = verify_quant
    except Exception as exc:  # noqa: BLE001 - durable stage failure capture
        record["status"] = "failed"
        record["error"] = str(exc)
        record["failed_at"] = utc_now()
        raise
    return record


def resolve_manifest_path(cfg: dict) -> Path:
    raw = cfg.get("publication_manifest") or "results/full_pipeline/publication_manifest.json"
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def build_report(cfg: dict, state: dict) -> dict:
    manifest_path = resolve_manifest_path(cfg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    rows = []
    for model in cfg["models"]:
        key = model_key(model)
        rec = state["models"].get(key, {})
        hub = (
            (manifest.get("models") or {})
            .get(model["size"], {})
            .get(model["stage"], {})
            .get("gguf")
        )
        rows.append(
            {
                "key": key,
                "status": rec.get("status"),
                "output_dir": rec.get("output_dir"),
                "gguf_repo": None if not hub else hub.get("repo_id"),
                "revision": None if not hub else hub.get("revision"),
                "gguf_files": None if not hub else hub.get("gguf_files"),
                "error": rec.get("error"),
            }
        )
    return {
        "updated_at": utc_now(),
        "gates": state.get("gates", {}),
        "models": rows,
        "complete": all(row.get("status") == "complete" for row in rows),
    }


def main() -> int:
    args = build_parser().parse_args()
    load_trackio_env()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    state = load_state(state_path, args.resume)
    state["status"] = "running"
    save_state(state_path, state)

    try:
        # G0 assumed already executed; assert setup report exists
        setup_path = ROOT / "results" / "full_pipeline" / "llamacpp_setup.json"
        if not setup_path.is_file():
            run_py("50_setup_llamacpp.py", "--llama-root", cfg["llamacpp_root"])
        state["gates"]["G0"] = {"status": "pass", "at": utc_now()}

        models = selected(cfg["models"], args.only)
        if args.smoke_only:
            models = [m for m in models if m["size"] == "230m" and m["stage"] == "Instruct"]
            if not models:
                models = [cfg["models"][1]]  # Instruct 230m in config order

        # G1 preflight all selected
        preflight = {}
        for model in models:
            preflight[model_key(model)] = preflight_model(model)
        state["gates"]["G1"] = {"status": "pass", "at": utc_now(), "models": preflight}
        save_state(state_path, state)

        cleanup = bool(args.cleanup_after_publish or cfg.get("cleanup_after_publish"))
        for model in models:
            key = model_key(model)
            print(f"==== {key} ====", flush=True)
            process_model(
                model,
                cfg=cfg,
                state=state,
                skip_publish=args.skip_publish,
                cleanup_after_publish=cleanup,
            )
            save_state(state_path, state)
            if args.smoke_only:
                state["gates"]["G2"] = {"status": "pass", "at": utc_now(), "model": key}
                break

        if not args.smoke_only:
            dense_keys = [
                model_key(m)
                for m in models
                if m["size"] != "8b-a1b" and state["models"].get(model_key(m), {}).get("status") == "complete"
            ]
            moe_keys = [
                model_key(m)
                for m in models
                if m["size"] == "8b-a1b" and state["models"].get(model_key(m), {}).get("status") == "complete"
            ]
            state["gates"]["G3"] = {"status": "pass", "at": utc_now(), "models": dense_keys}
            state["gates"]["G4"] = {"status": "pass", "at": utc_now(), "models": moe_keys}
            if not args.skip_publish:
                state["gates"]["G5"] = {"status": "pass", "at": utc_now()}
                # G6: authenticated Hub Q4_K_M smokes for every completed model
                verify_quant = cfg.get("verify_quant", "Q4_K_M")
                hub_smokes: dict[str, dict] = {}
                manifest_path = resolve_manifest_path(cfg)
                manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file()
                    else {}
                )
                for model in models:
                    key = model_key(model)
                    if state["models"].get(key, {}).get("status") != "complete":
                        continue
                    hub = (
                        (manifest.get("models") or {})
                        .get(model["size"], {})
                        .get(model["stage"], {})
                        .get("gguf")
                    )
                    if not hub or not hub.get("repo_id"):
                        raise RuntimeError(f"G6 missing published gguf repo for {key}")
                    repo_id = hub["repo_id"]
                    hf_ref = f"{repo_id}:{verify_quant}"
                    out = ROOT / cfg["output_root"] / model["size"] / model["stage"].lower() / "hub_verify.json"
                    run_py(
                        "52_verify_gguf.py",
                        "--hf",
                        hf_ref,
                        "--prompt",
                        cfg.get("smoke_prompt", "Salaam aleekum"),
                        "--n-predict",
                        str(cfg.get("smoke_n_predict", 48)),
                        "--out",
                        str(out.relative_to(ROOT)),
                    )
                    hub_smokes[key] = {"hf": hf_ref, "verify": str(out.relative_to(ROOT))}
                state["gates"]["G6"] = {
                    "status": "pass",
                    "at": utc_now(),
                    "smokes": hub_smokes,
                }

        state["status"] = "complete"
        save_state(state_path, state)
        report = build_report(cfg, state)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2)[:3000])
        return 0
    except Exception as exc:  # noqa: BLE001
        state["status"] = "failed"
        state["error"] = str(exc)
        save_state(state_path, state)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
