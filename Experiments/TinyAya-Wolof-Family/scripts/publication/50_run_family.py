#!/usr/bin/env python3
"""Durable Tiny Aya Wolof family orchestrator with dual orthography + Hub uploads.

Shared: datasets (ns+std) → fertility → tokenizer → resize_instructs
Per ortho (ns, std): cpt → merge → prepare → sft×3 → pick_best → dpo → eval
Final: publish

Orthographies:
  ns  — informal non_standard+mixed, <|ortho_ns|>
  std — school/CLAD standard+mixed, <|ortho_std|>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data_gate import assert_production_data
from common.ortho_variants import get_variant, resolve_variants
from common.pipeline import atomic_json_write, utc_now


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


STATE = ROOT / "results" / "full_pipeline" / "tinyaya_state.json"

SHARED_STAGES = [
    "datasets",
    "fertility",
    "tokenizer",
    "resize_instructs",
]
ORTHO_STAGES = [
    "cpt",
    "merge_cpt",
    "prepare",
    "sft_earth",
    "sft_global",
    "sft_cpt",
    "pick_best",
    "dpo",
    "eval",
]
FINAL_STAGES = ["publish"]


def stage_name(base: str, ortho: str) -> str:
    suffix = get_variant(ortho)["stage_suffix"]
    return f"{base}{suffix}"


def expand_stages(orthos: list[str]) -> list[str]:
    stages = list(SHARED_STAGES)
    for ortho in orthos:
        for base in ORTHO_STAGES:
            stages.append(stage_name(base, ortho))
    stages.extend(FINAL_STAGES)
    return stages


def uploads_for(stage: str, ortho: str | None = None) -> list[str]:
    """Hub --which keys after a stage. Ortho-aware aliases handled in 06."""
    shared = {
        "datasets": ["datasets", "datasets_std", "prefs", "prefs_std", "state"],
        "fertility": ["fertility", "state"],
        "tokenizer": ["tokenizer", "ext_init", "state"],
        "resize_instructs": ["earth_ext_init", "global_ext_init", "state"],
        "publish": ["all"],
    }
    if stage in shared:
        return shared[stage]
    base = stage
    for key in ("ns", "std"):
        suf = get_variant(key)["stage_suffix"]
        if suf and stage.endswith(suf):
            base = stage[: -len(suf)]
            ortho = key
            break
    else:
        ortho = ortho or "ns"
    tag = "" if ortho == "ns" else "_std"
    mapping = {
        "cpt": [f"cpt_adapter{tag}", "state"],
        "merge_cpt": [f"cpt_merged{tag}", "state"],
        "prepare": [f"prepared{tag}", "state"],
        "sft_earth": [f"instruct_earth{tag}", "state"],
        "sft_global": [f"instruct_global{tag}", "state"],
        "sft_cpt": [f"instruct_cpt{tag}", "state"],
        "pick_best": [
            "sft_eval",
            f"instruct_earth{tag}",
            f"instruct_global{tag}",
            f"instruct_cpt{tag}",
            "state",
        ],
        "dpo": [f"dpo{tag}", "state"],
        "eval": [
            "bpb",
            "sft_eval",
            f"cpt_merged{tag}",
            f"instruct_earth{tag}",
            f"instruct_global{tag}",
            f"instruct_cpt{tag}",
            f"dpo{tag}",
            "family_eval",
            "state",
        ],
    }
    return mapping.get(base, ["state"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Tiny Aya Wolof family pipeline")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ortho",
        default="both",
        help="ns | std | both (default both — dual orthography family)",
    )
    parser.add_argument("--only-stage", default=None)
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    return parser


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": {}, "updated_at": None, "best_instruct": {}, "uploads": {}}


def mark(state: dict, key: str, **extra) -> None:
    state["completed"][key] = utc_now()
    state["updated_at"] = utc_now()
    for k, v in extra.items():
        state[k] = v
    atomic_json_write(STATE, state)


def done(state: dict, key: str, resume: bool) -> bool:
    return resume and key in state.get("completed", {})


def unmark(state: dict, key: str, reason: str) -> None:
    if key in (state.get("completed") or {}):
        del state["completed"][key]
        state["updated_at"] = utc_now()
        atomic_json_write(STATE, state)
        print(f"REOPEN {key}: {reason}", flush=True)


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_full_clm_eval(*, smoke: bool, skip_publish: bool) -> None:
    """Gold CLM (BPB+PPL+BLEU+chrF) via Evaluation/evaluate_clm.py; resumes missing only."""
    from common.eval_gate import family_eval_complete

    ok, missing = family_eval_complete()
    if ok:
        print("FULL_CLM_OK all required evaluate_clm metrics present", flush=True)
        return
    print(f"FULL_CLM_RESUME missing={missing}", flush=True)
    cmd = uv(
        "scripts/publication/55_run_eval_family.py",
        "--bleu-n-texts",
        "20" if smoke else "100",
        "--n-texts",
        "50" if smoke else "all",
    )
    if skip_publish:
        cmd.append("--skip-publish")
    run_cmd(cmd)
    ok2, missing2 = family_eval_complete()
    if not ok2:
        raise SystemExit(f"FULL_CLM_INCOMPLETE after eval: {missing2}")


def uv(*args: str) -> list[str]:
    py = ROOT / ".venv" / "bin" / "python"
    if py.is_file():
        return [str(py), *args]
    return ["uv", "run", "python", *args]


def publish_now(which: list[str], *, skip: bool) -> None:
    if skip or not which:
        return
    try:
        run_cmd(uv("scripts/publication/06_publish_artifacts.py", "--which", *which))
        print(f"HUB_UPLOAD_OK {' '.join(which)}", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"HUB_UPLOAD_FAIL {' '.join(which)}: {exc}", flush=True)


def ensure_trackio() -> None:
    load_trackio_env()
    space = os.environ.get("TRACKIO_SPACE_ID")
    bucket = os.environ.get("TRACKIO_BUCKET_ID")
    if not space or not bucket:
        print(
            "TRACKIO_WARN missing .env.trackio — run "
            "scripts/publication/44_setup_trackio_space.py before production",
            flush=True,
        )
        return
    print(
        f"TRACKIO_READY space={space} bucket={bucket} "
        f"dashboard=https://huggingface.co/spaces/{space}",
        flush=True,
    )


def pick_best_instruct(smoke: bool, ortho: str) -> dict:
    variant = get_variant(ortho)
    prefix = variant["sft_run_prefix"]
    candidates = [
        (
            "from_cpt",
            ROOT / "models" / variant["merged_instruct"],
            ROOT / "results" / "sft" / f"eval_{prefix}_from_cpt" / "evaluation.json",
            f"configs/sft/3b35/from_cpt{'_std' if ortho == 'std' else ''}_v1.yaml",
        ),
        (
            "from_earth",
            ROOT / "models" / variant["merged_earth"],
            ROOT / "results" / "sft" / f"eval_{prefix}_from_earth" / "evaluation.json",
            f"configs/sft/3b35/from_earth{'_std' if ortho == 'std' else ''}_v1.yaml",
        ),
        (
            "from_global",
            ROOT / "models" / variant["merged_global"],
            ROOT / "results" / "sft" / f"eval_{prefix}_from_global" / "evaluation.json",
            f"configs/sft/3b35/from_global{'_std' if ortho == 'std' else ''}_v1.yaml",
        ),
    ]
    scored = []
    for role, path, eval_path, cfg in candidates:
        if not path.is_dir() and not smoke:
            continue
        loss = None
        if eval_path.is_file():
            loss = json.loads(eval_path.read_text(encoding="utf-8")).get("mean_loss")
        scored.append({"role": role, "path": str(path), "mean_loss": loss, "config": cfg})
    scored_with = [s for s in scored if s["mean_loss"] is not None]
    if scored_with:
        best = min(scored_with, key=lambda s: s["mean_loss"])
    elif scored:
        best = next((s for s in scored if s["role"] == "from_earth"), scored[0])
    else:
        best = {
            "role": "from_earth",
            "path": str(ROOT / "models" / variant["merged_earth"]),
            "mean_loss": None,
            "config": candidates[1][3],
        }
    dest = ROOT / "models" / variant["merged_best"]
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json_write(
        dest / "best_pointer.json",
        {
            "created_at": utc_now(),
            "ortho": ortho,
            "best": best,
            "smoke": smoke,
            "missing_weights": not Path(best["path"]).is_dir(),
        },
    )
    return best


def run_ortho_stage(
    *,
    base: str,
    ortho: str,
    smoke: bool,
    family: dict,
    state: dict,
    skip_publish: bool,
    resume: bool = False,
) -> None:
    variant = get_variant(ortho)
    size = family["size"]
    smoke_flags = ["--smoke"] if smoke else []
    resume_flags = ["--resume"] if resume else []
    key = stage_name(base, ortho)

    if base == "cpt":
        cfg = (
            f"configs/clm/3b35/smoke{'_std' if ortho == 'std' else ''}.yaml"
            if smoke
            else (
                "configs/clm/3b35/v1_ntp70_ortho_std_lora.yaml"
                if ortho == "std"
                else "configs/clm/3b35/v1_ntp70_ortho_lora.yaml"
            )
        )
        run_cmd(
            uv("scripts/clm/03_train_cpt.py", "--config", cfg, *smoke_flags, *resume_flags)
        )

    elif base == "merge_cpt":
        adapter = (
            ROOT / "results" / "smoke" / f"{variant['cpt_smoke_run']}-cpt" / "adapter"
            if smoke
            else ROOT / "runs" / "clm" / "3b35" / variant["cpt_run"] / "adapter"
        )
        if smoke and not adapter.is_dir():
            # smoke.yaml may redirect output via run name
            for cand in (ROOT / "results" / "smoke").glob(f"*cpt*{ortho}*"):
                if (cand / "adapter").is_dir():
                    adapter = cand / "adapter"
                    break
            if not adapter.is_dir():
                adapter = ROOT / "results" / "smoke" / "tinyaya_3b35_cpt_smoke-cpt" / "adapter"
                if ortho == "std":
                    adapter = ROOT / "results" / "smoke" / "tinyaya_3b35_cpt_std_smoke-cpt" / "adapter"
        merged = ROOT / "models" / variant["merged_cpt"]
        base_model = ROOT / "models" / "tinyaya-3.35b-ext-init"
        run_cmd(
            uv(
                "scripts/clm/05_merge_adapter.py",
                "--base",
                str(base_model),
                "--adapter",
                str(adapter),
                "--output",
                str(merged),
            )
        )

    elif base == "prepare":
        # Prefer Instruct-extended tokenizer (chat_template + START/END_RESPONSE).
        # Base ext-anchors alone cannot encode Cohere chat turns for SFT.
        tok_candidates = (
            ROOT / "models" / "tinyaya-3.35b-earth-ext-init",
            ROOT / "models" / "tinyaya-3.35b-global-ext-init",
            ROOT / "tokenizers" / "tinyaya-wolof-ext-anchors",
        )
        tok = next((p for p in tok_candidates if p.is_dir()), None)
        extra = list(smoke_flags)
        if tok is not None:
            extra.extend(["--tokenizer-model", str(tok)])
        if not any(p.name.endswith("ext-init") and p.is_dir() for p in tok_candidates[:2]):
            print(
                "WARN: earth/global-ext-init missing; prepare may fail without "
                "Instruct chat specials",
                flush=True,
            )
        run_cmd(uv("scripts/data/03_anchor_and_prepare_sft.py", "--ortho", ortho, *extra))

    elif base == "sft_earth":
        cfg = f"configs/sft/3b35/from_earth{'_std' if ortho == 'std' else ''}_v1.yaml"
        run_cmd(
            uv("scripts/sft/11_train_sft.py", "--config", cfg, *smoke_flags, *resume_flags)
        )
        adapter = (
            ROOT
            / "results"
            / "smoke"
            / f"tinyaya_3b35_sft_from_earth{'_std' if ortho == 'std' else ''}_v1-sft"
            / "adapter"
            if smoke
            else ROOT / "runs" / "sft" / f"{variant['sft_run_prefix']}_from_earth_v1" / "adapter"
        )
        # smoke redirects using config run name
        if smoke:
            run_name = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))["run"]
            adapter = ROOT / "results" / "smoke" / f"{run_name}-sft" / "adapter"
        run_cmd(
            uv(
                "scripts/sft/13_merge_sft.py",
                "--base",
                "models/tinyaya-3.35b-earth-ext-init",
                "--adapter",
                str(adapter),
                "--output",
                f"models/{variant['merged_earth']}",
            )
        )

    elif base == "sft_global":
        cfg = f"configs/sft/3b35/from_global{'_std' if ortho == 'std' else ''}_v1.yaml"
        run_cmd(
            uv("scripts/sft/11_train_sft.py", "--config", cfg, *smoke_flags, *resume_flags)
        )
        run_name = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))["run"]
        adapter = (
            ROOT / "results" / "smoke" / f"{run_name}-sft" / "adapter"
            if smoke
            else ROOT / "runs" / "sft" / f"{variant['sft_run_prefix']}_from_global_v1" / "adapter"
        )
        run_cmd(
            uv(
                "scripts/sft/13_merge_sft.py",
                "--base",
                "models/tinyaya-3.35b-global-ext-init",
                "--adapter",
                str(adapter),
                "--output",
                f"models/{variant['merged_global']}",
            )
        )

    elif base == "sft_cpt":
        # Base CPT vocab puts Wolof chunks at 261000+ where Instruct keeps
        # START_RESPONSE/… specials — prepared Instruct ids do not match CPT.
        # Skip until CPT is rebuilt on an Instruct-compatible tokenizer layout.
        cpt_merged = ROOT / "models" / variant["merged_cpt"]
        skip_cpt = False
        if cpt_merged.is_dir():
            try:
                from transformers import AutoTokenizer as _AT

                _tok = _AT.from_pretrained(str(cpt_merged), trust_remote_code=True)
                _sr = _tok.convert_tokens_to_ids("<|START_RESPONSE|>")
                if _sr is None or _sr == getattr(_tok, "unk_token_id", None):
                    skip_cpt = True
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: could not probe CPT tokenizer ({exc})", flush=True)
                skip_cpt = True
        else:
            skip_cpt = True
        if skip_cpt:
            print(
                f"SKIP sft_cpt/{ortho}: CPT merged lacks Instruct chat specials "
                "(vocab layout incompatible with prepared_anchored); "
                "earth/global SFT still run",
                flush=True,
            )
            mark(state, key)
            publish_now(uploads_for(key, ortho), skip=skip_publish)
            print(f"STAGE_OK {key} (skipped)", flush=True)
            return
        cfg = f"configs/sft/3b35/from_cpt{'_std' if ortho == 'std' else ''}_v1.yaml"
        run_cmd(
            uv("scripts/sft/11_train_sft.py", "--config", cfg, *smoke_flags, *resume_flags)
        )
        run_name = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))["run"]
        adapter = (
            ROOT / "results" / "smoke" / f"{run_name}-sft" / "adapter"
            if smoke
            else ROOT / "runs" / "sft" / f"{variant['sft_run_prefix']}_from_cpt_v1" / "adapter"
        )
        run_cmd(
            uv(
                "scripts/sft/13_merge_sft.py",
                "--base",
                f"models/{variant['merged_cpt']}",
                "--adapter",
                str(adapter),
                "--output",
                f"models/{variant['merged_instruct']}",
            )
        )

    elif base == "pick_best":
        prefix = variant["sft_run_prefix"]
        for role, model_path, out_dir in (
            (
                "from_cpt",
                f"models/{variant['merged_instruct']}",
                f"results/sft/eval_{prefix}_from_cpt",
            ),
            (
                "from_earth",
                f"models/{variant['merged_earth']}",
                f"results/sft/eval_{prefix}_from_earth",
            ),
            (
                "from_global",
                f"models/{variant['merged_global']}",
                f"results/sft/eval_{prefix}_from_global",
            ),
        ):
            if (ROOT / model_path).is_dir():
                try:
                    prepared = f"data/sft/{variant['prepared_dirname']}"
                    run_cmd(
                        uv(
                            "scripts/sft/12_evaluate_sft.py",
                            "--model",
                            model_path,
                            "--data",
                            prepared,
                            "--output",
                            out_dir,
                            "--max-rows",
                            "50" if smoke else "200",
                        )
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"eval {ortho}/{role} failed ({exc}); continuing pick", flush=True)
        best = pick_best_instruct(smoke, ortho)
        bests = state.get("best_instruct") or {}
        if not isinstance(bests, dict):
            bests = {"ns": bests}
        bests[ortho] = best
        mark(state, key, best_instruct=bests)
        publish_now(uploads_for(key, ortho), skip=skip_publish)
        print(f"STAGE_OK {key} best={best}", flush=True)
        return

    elif base == "dpo":
        cfg = f"configs/sft/3b35/dpo{'_std' if ortho == 'std' else ''}_v1.yaml"
        # Prefer pick_best merged weights (merged_best/ is often only a pointer stub).
        dpo_extra = list(smoke_flags)
        best_ptr = ROOT / "models" / variant["merged_best"] / "best_pointer.json"
        if best_ptr.is_file():
            best = json.loads(best_ptr.read_text(encoding="utf-8")).get("best") or {}
            best_path = best.get("path")
            if best_path and Path(best_path).is_dir() and (Path(best_path) / "config.json").is_file():
                dpo_extra.extend(["--model", str(best_path)])
                print(f"DPO model from pick_best: {best_path}", flush=True)
        run_cmd(uv("scripts/sft/14_train_dpo.py", "--config", cfg, *dpo_extra))

    elif base == "eval":
        # Gold protocol: evaluate_clm.py (BPB+PPL+BLEU+chrF). Resumes only missing.
        # Run once per family pass (both orthos); second ortho call is a no-op if complete.
        run_full_clm_eval(smoke=smoke, skip_publish=True)

    else:
        raise SystemExit(f"unknown ortho stage base {base}")

    mark(state, key)
    # Publish model weights+metrics only after full CLM for this ortho is present.
    if base == "eval":
        from common.eval_gate import ortho_eval_complete

        ok, missing = ortho_eval_complete(ortho)
        if not ok:
            raise SystemExit(f"eval stage refused publish; missing full CLM: {missing}")
    publish_now(uploads_for(key, ortho), skip=skip_publish)
    print(f"STAGE_OK {key}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    ensure_trackio()
    orthos = resolve_variants(args.ortho)
    print(f"ORTHO_VARIANTS {orthos}", flush=True)
    if args.smoke:
        os.environ.setdefault("TINYAYA_PIPELINE_RUN_ID", "smoke")
    else:
        os.environ.setdefault("TINYAYA_PRODUCTION", "1")
        os.environ.setdefault("TINYAYA_PIPELINE_RUN_ID", "prod")

    state = load_state()
    family = yaml.safe_load((ROOT / "configs" / "family.yaml").read_text(encoding="utf-8"))
    size = family["size"]
    smoke_flags = ["--smoke"] if args.smoke else []
    if not args.smoke:
        meta = ROOT / "data" / "raw" / "materialize_meta.json"
        # Resume must never skip rematerialize while smoke caches remain.
        if args.resume and "datasets" in state.get("completed", {}):
            assert_production_data(ROOT)
        elif meta.is_file():
            try:
                assert_production_data(ROOT)
            except SystemExit as exc:
                # Fresh production run will rematerialize in the datasets stage.
                print(f"DATA_GATE_WILL_REMATERIALIZE {exc}", flush=True)

    stages = expand_stages(orthos)
    if args.only_stage:
        if args.only_stage not in stages:
            # allow bare names like cpt → expand for each ortho
            expanded = []
            for ortho in orthos:
                name = stage_name(args.only_stage, ortho)
                if name in stages:
                    expanded.append(name)
            if args.only_stage in SHARED_STAGES + FINAL_STAGES:
                stages = [args.only_stage]
            elif expanded:
                stages = expanded
            else:
                raise SystemExit(f"unknown --only-stage {args.only_stage}")
        else:
            stages = [args.only_stage]
    if args.skip_publish:
        stages = [s for s in stages if s != "publish"]
    if args.skip_dpo:
        stages = [s for s in stages if not s.startswith("dpo")]

    for stage in stages:
        # On resume: reopen eval/publish if gold CLM (BPB+BLEU+chrF+PPL) is incomplete.
        if done(state, stage, args.resume):
            reopen = False
            if stage.startswith("eval"):
                from common.eval_gate import ortho_eval_complete

                ortho = "std" if stage.endswith("_std") else "ns"
                ok, missing = ortho_eval_complete(ortho)
                if not ok:
                    unmark(state, stage, f"missing full CLM {missing}")
                    reopen = True
            elif stage == "publish":
                from common.eval_gate import family_eval_complete

                ok, missing = family_eval_complete()
                if not ok:
                    unmark(state, stage, f"missing full CLM {missing}")
                    # Also reopen eval stages so they re-run before publish.
                    for key in ("eval", "eval_std"):
                        if key in (state.get("completed") or {}):
                            unmark(state, key, "publish requires full CLM")
                    reopen = True
            if not reopen:
                print(f"skip completed {stage}", flush=True)
                publish_now(uploads_for(stage), skip=args.skip_publish)
                continue

        if stage == "datasets":
            run_cmd(uv("scripts/data/01_materialize_sources.py", *smoke_flags))
            ortho_flag = "both" if set(orthos) == {"ns", "std"} else orthos[0]
            run_cmd(
                uv(
                    "scripts/data/02_ortho_label_and_filter.py",
                    "--ortho",
                    ortho_flag,
                    *smoke_flags,
                )
            )
            run_cmd(
                uv("scripts/data/05_build_prefs.py", "--ortho", ortho_flag, *smoke_flags)
            )
            if not args.smoke:
                assert_production_data(ROOT)

        elif stage == "fertility":
            run_cmd(
                uv(
                    "scripts/clm/02c_fertility_probe.py",
                    "--base-id",
                    size["base_id"],
                    "--max-rows",
                    "2000" if args.smoke else "5000",
                )
            )

        elif stage == "tokenizer":
            tok_cfg = "configs/tokenizer/ext_wolof_anchors.yaml"
            extra = ["--max-rows", "20000"] if args.smoke else []
            run_cmd(uv("scripts/clm/02_build_tokenizer.py", "--config", tok_cfg, *extra))

        elif stage == "resize_instructs":
            for role, instruct_id, out_name in (
                (
                    "earth",
                    size["instruct_ids"]["earth"],
                    "models/tinyaya-3.35b-earth-ext-init",
                ),
                (
                    "global",
                    size["instruct_ids"]["global"],
                    "models/tinyaya-3.35b-global-ext-init",
                ),
            ):
                run_cmd(
                    uv(
                        "scripts/clm/02b_resize_ext_to_instruct.py",
                        "--instruct-id",
                        instruct_id,
                        "--tokenizer",
                        "tokenizers/tinyaya-wolof-ext-anchors",
                        "--source-model",
                        "models/tinyaya-3.35b-ext-init",
                        "--output",
                        out_name,
                    )
                )
                print(f"resized {role} -> {out_name}", flush=True)

        elif stage == "publish":
            # Always fill any missing gold CLM before Hub push.
            run_full_clm_eval(smoke=args.smoke, skip_publish=True)
            from common.eval_gate import family_eval_complete

            ok, missing = family_eval_complete()
            if not ok:
                raise SystemExit(f"publish blocked; missing full CLM: {missing}")
            publish_now(["all"], skip=False)
            mark(state, stage)
            print(f"STAGE_OK {stage}", flush=True)
            continue

        else:
            # Per-ortho stage
            matched = False
            for ortho in orthos:
                for base in ORTHO_STAGES:
                    if stage == stage_name(base, ortho):
                        run_ortho_stage(
                            base=base,
                            ortho=ortho,
                            smoke=args.smoke,
                            family=family,
                            state=state,
                            skip_publish=args.skip_publish,
                            resume=bool(args.resume),
                        )
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                raise SystemExit(f"unknown stage {stage}")
            continue

        mark(state, stage)
        publish_now(uploads_for(stage), skip=args.skip_publish)
        print(f"STAGE_OK {stage}", flush=True)

    print("TINYAYA_FAMILY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
