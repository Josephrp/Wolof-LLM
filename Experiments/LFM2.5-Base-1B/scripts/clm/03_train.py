#!/usr/bin/env python3
"""Continual pretraining for LFM2.5 Wolof — one code path for the whole grid.

Port of Experiments/SmolLM3-3B-Base/scripts/03_train.py with LFM adaptations:
  * LiquidAI/ Hub ids + local output_root resolution
  * Hub charger_splits_hub when no local parquet
  * Default EOS 7 for packing (65k family)
  * --smoke -> 10 steps under results/smoke/
  * Journal: size, tokenizer_family, stage, EOS id, uv/pkg versions

Heavy deps are imported lazily so ``--help`` works without train group:

  uv run python scripts/clm/03_train.py --help
  uv run python scripts/clm/03_train.py --config configs/scaling/smoke_350m_default_lora.yaml --smoke
"""

from __future__ import annotations

import argparse
import atexit
import importlib.metadata as importlib_metadata
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.config import sha256_file, validate_clm_config
from common.tracking import TrackingRun, TrackioAlertCallback


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LFM2.5 Wolof CLM train (Trainer, dual-LR, smoke, phases)"
    )
    p.add_argument("--config", required=True, help="YAML under configs/scaling or configs/clm")
    p.add_argument("--racine", default=".", help="Experiment root (default: .)")
    p.add_argument("--smoke", action="store_true", help="10-step validation run")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs or resume from the latest Trainer checkpoint",
    )
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--dataset-id", default=None,
                   help="Override Hub CLM dataset id")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Optional Hub row cap (smoke defaults apply if unset)")
    p.add_argument("--eos-token-id", type=int, default=None,
                   help="Packing EOS override (default: 7 or tokenizer.eos)")
    return p


def _is_hub_id(s: str) -> bool:
    if not s or s in ("null", "None"):
        return False
    # org/name Hub repos (LiquidAI/..., galsenai/..., HuggingFaceTB/...)
    if s.startswith(("LiquidAI/", "HuggingFace", "galsenai/", "http://", "https://")):
        return True
    if "/" in s and not Path(s).exists() and not s.startswith((".", "/", "\\")):
        # treat as Hub id when it looks like owner/repo and is not a local path
        parts = s.split("/")
        if len(parts) == 2 and all(parts):
            return True
    return False


def resoudre_artefact(chemin, racine: Path, output_root: Path):
    """Resolve local relative paths; pass through Hub ids unchanged."""
    if chemin is None or str(chemin).lower() in ("none", "null", ""):
        return None
    s = str(chemin)
    if _is_hub_id(s) or Path(s).is_absolute():
        return s
    dans_output = output_root / s
    if dans_output.exists():
        return str(dans_output)
    dans_racine = racine / s
    if dans_racine.exists():
        return str(dans_racine)
    # Prefer output_root for not-yet-built artefacts (ext tokenizer/model)
    return str(dans_output if (output_root / s).parent.exists() or output_root.exists()
               else dans_racine)


def _pkg_versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = None
    try:
        out["galsenai-lfm25-wolof"] = importlib_metadata.version(
            "galsenai-lfm25-wolof"
        )
    except importlib_metadata.PackageNotFoundError:
        out["galsenai-lfm25-wolof"] = None
    out["uv"] = os.environ.get("UV") or os.environ.get("UV_PYTHON") or None
    return out


def _load_texts(racine, data_root, dataset_id, token, max_rows):
    from common.data import charger_splits_hub, charger_splits_local

    if data_root is not None:
        try:
            train, ev = charger_splits_local(racine, data_root=data_root)
            if max_rows is not None:
                train, ev = train[:max_rows], ev[:max_rows]
            return train, ev, "local_parquet"
        except FileNotFoundError:
            print("  local parquet missing — falling back to Hub")

    if not token:
        print(
            "ERROR: no local parquet and HF_TOKEN unset.\n"
            "  Set HF_TOKEN or point data_root at wolof_bpe_train_v1.parquet.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    train, ev = charger_splits_hub(
        dataset_id=dataset_id, max_rows=max_rows, token=token
    )
    return train, ev, dataset_id


def _infer_tokenizer_family(cfg, chemin_tok, len_tok) -> str:
    fam = cfg.get("tokenizer_family") or cfg.get("family")
    if fam:
        return str(fam)
    if chemin_tok and "ext" in str(chemin_tok).lower():
        return "65k-ext"
    if len_tok and len_tok > 70_000:
        return "128k"
    return "65k"


def _dataloader_training_args(training: dict, accepted: set[str]) -> dict:
    """Build supported Trainer dataloader options without changing batch semantics."""
    workers = int(training.get("dataloader_num_workers", 0))
    if workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    configured = {
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": bool(training.get("dataloader_pin_memory", True)),
    }
    if workers:
        configured.update(
            {
                "dataloader_persistent_workers": bool(
                    training.get("dataloader_persistent_workers", True)
                ),
                "dataloader_prefetch_factor": int(
                    training.get("dataloader_prefetch_factor", 2)
                ),
            }
        )
    return {key: value for key, value in configured.items() if key in accepted}


def run(args: argparse.Namespace) -> int:
    import gc
    import inspect
    import time

    import torch
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    sys.path.insert(0, str(ROOT))
    from common.data import paqueter
    from common.journal import ecrire, extraire_courbes, tracer
    from common.metrics import compute_bpb
    from common.model import (
        charger,
        degeler_backbone,
        geler_backbone,
        groupes_parametres,
        preparer_full,
        preparer_lora,
    )

    racine = Path(args.racine).resolve()
    if not (racine / "pyproject.toml").exists() and (ROOT / "pyproject.toml").exists():
        racine = ROOT

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (racine / cfg_path) if (racine / cfg_path).exists() else (ROOT / cfg_path)
    with open(cfg_path, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    validate_clm_config(cfg, cfg_path)

    nom = cfg["run"]
    methode = cfg["methode"]
    e = dict(cfg["entrainement"])  # copy — smoke mutates

    # output_root: config may already be runs/smoke/... ; --smoke forces results/smoke/
    raw_out = cfg.get("output_root")
    if args.smoke:
        nom = nom if nom.endswith("-smoke") or "smoke" in nom else f"{nom}-smoke"
        e["max_steps"] = 10
        sortie = racine / "results" / "smoke" / nom
    elif raw_out:
        sortie = Path(raw_out)
        if not sortie.is_absolute():
            sortie = racine / sortie
    else:
        sortie = racine / "runs" / nom

    # Artefact resolution prefers experiment root (tokenizers/, models/)
    output_root_resolve = racine
    sortie.mkdir(parents=True, exist_ok=True)
    if args.resume and (sortie / "journal.json").is_file():
        print(f"Run already complete, skipping: {sortie}")
        return 0
    tracking = TrackingRun(
        size=str(cfg.get("size", "unknown")),
        stage=str(cfg.get("stage", "cpt")),
        tokenizer_family=str(cfg.get("tokenizer_family", "65k")),
        method=str(methode),
        seed=int(e["seed"]),
        run_id=os.environ.get("LFM_PIPELINE_RUN_ID", nom),
        config=cfg,
        production=os.environ.get("LFM_PRODUCTION") == "1",
        disabled=os.environ.get("TRACKIO_DISABLED") == "1",
        state_path=sortie / "trackio.json",
    ).start()
    atexit.register(tracking.finish, success=False)

    tokens_par_pas = e["micro_bs"] * e["grad_acc"] * e["seq_len"]
    print(f"{'=' * 70}\n  {nom}  [{methode}]\n{'=' * 70}")
    print(
        f"  {tokens_par_pas:,} tokens/pas | {e['max_steps']} pas = "
        f"{e['max_steps'] * tokens_par_pas / 1e6:.1f} M tokens"
    )
    print(f"  learning rate : {e['lr']}")
    print(f"  sortie : {sortie}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        libre, total = torch.cuda.mem_get_info()
        print(
            f"  GPU : {torch.cuda.get_device_name(0)} "
            f"({libre / 1e9:.1f}/{total / 1e9:.1f} Go libres)"
        )
        memoire_fraction = cfg.get("memoire_fraction")
        if memoire_fraction:
            torch.cuda.set_per_process_memory_fraction(float(memoire_fraction), 0)
            print(f"  memoire plafonnee a {float(memoire_fraction):.0%}")
        budget = total / 1e9 * (float(memoire_fraction) if memoire_fraction else 1.0)
        if methode == "full" and budget < 45:
            print(
                f"  ATTENTION : budget {budget:.0f} Go — envisager "
                "optim: adamw_bnb_8bit ou reduire micro_bs"
            )

    # --- model / tokenizer ------------------------------------------------
    base_id = cfg.get("base_id") or "LiquidAI/LFM2.5-1.2B-Base"
    configured_model = cfg.get("model")
    configured_tokenizer = cfg.get("tokenizer")
    chemin_modele = resoudre_artefact(
        configured_model or base_id, racine, output_root_resolve
    )
    chemin_tok = resoudre_artefact(
        configured_tokenizer or base_id, racine, output_root_resolve
    )
    if chemin_modele is None:
        chemin_modele = base_id
    if chemin_tok is None:
        chemin_tok = base_id
    # Explicit extended artefacts must exist. Falling back to the stock base
    # would silently turn an extended-tokenizer run into a different recipe.
    if not _is_hub_id(chemin_modele) and not Path(chemin_modele).exists():
        raise SystemExit(f"ERROR: configured model path does not exist: {chemin_modele}")
    if not _is_hub_id(chemin_tok) and not Path(chemin_tok).exists():
        raise SystemExit(
            f"ERROR: configured tokenizer path does not exist: {chemin_tok}"
        )

    tok, model = charger(
        chemin_modele, chemin_tok, methode, attention=cfg.get("attention", "sdpa")
    )
    eos = tok.eos_token_id
    eos = eos[0] if isinstance(eos, (list, tuple)) else eos
    eos_pack = args.eos_token_id
    if eos_pack is None:
        eos_pack = cfg.get("eos_token_id", 7 if eos == 7 else eos)
    print(f"  packing eos_token_id={eos_pack}")

    # --- data -------------------------------------------------------------
    data_root = Path(cfg["data_root"]) if cfg.get("data_root") else None
    token = os.environ.get(args.hf_token_env) or os.environ.get("HF_TOKEN")
    dataset_id = (
        args.dataset_id
        or cfg.get("dataset_id")
        or "galsenai/cleaned_data_for_clm"
    )
    max_rows = args.max_rows
    if args.smoke and max_rows is None:
        max_rows = 20_000
    textes_train, textes_eval, data_src = _load_texts(
        racine, data_root, dataset_id, token, max_rows
    )
    limite = 20_000 if args.smoke else None
    ds_train = paqueter(
        textes_train, tok, e["seq_len"], limite, eos_token_id=eos_pack
    ).shuffle(seed=e["seed"])
    ds_eval = paqueter(textes_eval, tok, e["seq_len"], eos_token_id=eos_pack)
    print(f"  data_src={data_src} | blocs : train {len(ds_train):,} | eval {len(ds_eval):,}")

    # --- BPB before -------------------------------------------------------
    n_eval = 100 if args.smoke else 500
    bpb_avant = compute_bpb(model, tok, textes_eval[:n_eval])
    print(f"  BPB avant : {bpb_avant['bpb']:.4f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- prepare ----------------------------------------------------------
    gc_actif = e.get("gradient_checkpointing", True)
    if methode == "lora":
        model = preparer_lora(model, cfg.get("lora") or {}, gc_actif)
    else:
        model = preparer_full(model, gc_actif)

    lr_emb = e.get("lr_embeddings")
    pas_echauffement = int(e.get("warmup_embeddings_steps", 0) or 0)

    class TrainerLRDifferencie(Trainer):
        groupes = None

        def create_optimizer(self):
            if self.optimizer is None and self.groupes:
                cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
                kwargs.pop("lr", None)
                kwargs.pop("weight_decay", None)
                self.optimizer = cls(
                    self.groupes, lr=self.args.learning_rate, **kwargs
                )
                return self.optimizer
            return super().create_optimizer()

    def arguments_entrainement(
        out, ee, gc_flag, max_steps=None, lr=None, warmup_ratio=None, arret_anticipe=False
    ):
        params = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = {
            "output_dir": str(out),
            "max_steps": max_steps if max_steps is not None else ee["max_steps"],
            "per_device_train_batch_size": ee["micro_bs"],
            "gradient_accumulation_steps": ee["grad_acc"],
            "learning_rate": float(lr if lr is not None else ee["lr"]),
            "lr_scheduler_type": ee.get("scheduler", "cosine"),
            "logging_steps": ee.get("logging_steps", 25),
            "eval_steps": ee.get("eval_steps", 50),
            "save_steps": ee.get("save_steps", ee["max_steps"]),
            "save_total_limit": ee.get("save_total_limit", 2),
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": gc_flag,
            "optim": ee.get("optim", "adamw_torch"),
            "report_to": "trackio" if not tracking.disabled else "none",
            "seed": ee["seed"],
        }
        kwargs.update(_dataloader_training_args(ee, set(params)))
        for key, value in {
            "trackio_space_id": tracking.space_id,
            "trackio_bucket_id": tracking.bucket_id,
            "run_name": tracking.name,
        }.items():
            if key in params and value:
                kwargs[key] = value
        ratio = warmup_ratio if warmup_ratio is not None else ee.get("warmup_ratio", 0.03)
        if "warmup_ratio" in params:
            kwargs["warmup_ratio"] = ratio
        else:
            kwargs["warmup_steps"] = int(kwargs["max_steps"] * ratio)
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "steps"
        else:
            kwargs["evaluation_strategy"] = "steps"
        if arret_anticipe:
            kwargs["save_steps"] = kwargs["eval_steps"]
            kwargs["load_best_model_at_end"] = True
            kwargs["metric_for_best_model"] = "eval_loss"
            kwargs["greater_is_better"] = False
            kwargs["save_total_limit"] = max(kwargs.get("save_total_limit", 2), 2)
        return TrainingArguments(**kwargs)

    # --- Phase A ----------------------------------------------------------
    duree_a = 0.0
    if pas_echauffement and methode == "full" and not args.smoke:
        print(
            f"\n  --- Phase A : echauffement embeddings ({pas_echauffement} pas) ---"
        )
        geler_backbone(model)
        args_a = arguments_entrainement(
            sortie / "phase_a",
            e,
            gc_actif,
            max_steps=pas_echauffement,
            lr=lr_emb or e["lr"],
            warmup_ratio=0.1,
        )
        if hasattr(args_a, "eval_strategy"):
            args_a.eval_strategy = "no"
        else:
            args_a.evaluation_strategy = "no"
        trainer_a = Trainer(
            model=model,
            args=args_a,
            train_dataset=ds_train,
            data_collator=default_data_collator,
        )
        t_a = time.time()
        trainer_a.train()
        duree_a = (time.time() - t_a) / 3600
        bpb_phase_a = compute_bpb(model, tok, textes_eval[:n_eval])
        print(f"  BPB apres echauffement : {bpb_phase_a['bpb']:.4f}")
        degeler_backbone(model)
        del trainer_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        bpb_phase_a = None

    # --- Phase B ----------------------------------------------------------
    patience = int(e.get("patience", 0) or 0)
    arret = patience > 0 and not args.smoke
    if arret:
        print(
            f"  arret anticipe : patience {patience} "
            f"(plafond {e['max_steps']} pas)"
        )

    ta = arguments_entrainement(sortie, e, gc_actif, arret_anticipe=arret)
    groupes = groupes_parametres(
        model, float(e["lr"]), lr_emb, weight_decay=ta.weight_decay
    )
    callbacks = (
        [
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=float(e.get("seuil_amelioration", 0.0)),
            )
        ]
        if arret
        else []
    )
    callbacks.append(TrackioAlertCallback(tracking))

    commun = {
        "model": model,
        "args": ta,
        "train_dataset": ds_train,
        "eval_dataset": ds_eval,
        "data_collator": default_data_collator,
        "callbacks": callbacks,
    }
    if groupes:
        trainer = TrainerLRDifferencie(**commun)
        trainer.groupes = groupes
    else:
        trainer = Trainer(**commun)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    resume_checkpoint = bool(args.resume and any(sortie.glob("checkpoint-*")))
    res = trainer.train(resume_from_checkpoint=resume_checkpoint)
    duree = (time.time() - t0) / 3600 + duree_a

    bpb_apres = compute_bpb(model, tok, textes_eval[:n_eval])
    gain = 100 * (1 - bpb_apres["bpb"] / bpb_avant["bpb"])
    print(
        f"  BPB : {bpb_avant['bpb']:.4f} -> {bpb_apres['bpb']:.4f} "
        f"({gain:.1f} % de gain)"
    )

    ltr, lval, llr = extraire_courbes(trainer)
    tracer(ltr, lval, llr, nom, sortie / "courbes.png")
    peak_vram = (
        {
            "allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        }
        if torch.cuda.is_available()
        else None
    )

    tok_family = _infer_tokenizer_family(cfg, chemin_tok, len(tok))
    ecrire(
        sortie / "journal.json",
        {
            "run": nom,
            "methode": methode,
            "config": str(cfg_path),
            "size": cfg.get("size"),
            "stage": cfg.get("stage") or ("smoke" if args.smoke else None),
            "tokenizer_family": tok_family,
            "eos_token_id": eos_pack,
            "base_id": base_id,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_vram": peak_vram,
            "modele": str(chemin_modele),
            "tokenizer": str(chemin_tok),
            "vocab": len(tok),
            "smoke": args.smoke,
            "versions": _pkg_versions(),
            "uv_lock_sha256": sha256_file(racine / "uv.lock"),
            "hyperparametres": {
                **e,
                "lora": cfg.get("lora") if methode == "lora" else None,
            },
            "budget": {
                "pas_plafond": e["max_steps"],
                "pas_effectues": int(trainer.state.global_step),
                "arret_anticipe": arret
                and trainer.state.global_step < e["max_steps"],
                "patience": patience,
                "tokens_vus_M": int(trainer.state.global_step) * tokens_par_pas / 1e6,
                "duree_heures": round(duree, 3),
                "epochs_equivalent": round(
                    int(trainer.state.global_step)
                    * e["micro_bs"]
                    * e["grad_acc"]
                    / max(len(ds_train), 1),
                    4,
                ),
            },
            "loss": {
                "train_premiere": ltr[0][1] if ltr else None,
                "train_derniere": ltr[-1][1] if ltr else None,
                "val_derniere": lval[-1][1] if lval else None,
                "moyenne_run": res.metrics.get("train_loss"),
                "courbe_train": ltr,
                "courbe_val": lval,
            },
            "eval": {
                "split": data_src,
                "n_textes": n_eval,
                "bpb_avant": bpb_avant["bpb"],
                "bpb_apres": bpb_apres["bpb"],
                "ppl_avant": bpb_avant["ppl"],
                "ppl_apres": bpb_apres["ppl"],
                "bpb_apres_echauffement": bpb_phase_a["bpb"] if bpb_phase_a else None,
            },
            "phase_a": {
                "pas": pas_echauffement if bpb_phase_a else 0,
                "duree_heures": round(duree_a, 3),
                "lr": lr_emb or e["lr"],
            }
            if bpb_phase_a
            else None,
            "lr_differencies": bool(groupes),
        },
    )

    if methode == "lora":
        model.save_pretrained(str(sortie / "adapter"))
        tok.save_pretrained(str(sortie / "adapter"))
    else:
        model.save_pretrained(str(sortie / "model"), safe_serialization=True)
        tok.save_pretrained(str(sortie / "model"))

    if arret and trainer.state.global_step < e["max_steps"]:
        print(
            f"  arret anticipe au pas {trainer.state.global_step} "
            f"/ {e['max_steps']}"
        )
    if peak_vram:
        print(
            "  peak VRAM : "
            f"{peak_vram['allocated_mib']:.2f} MiB allocated, "
            f"{peak_vram['reserved_mib']:.2f} MiB reserved"
        )
    print(f"  duree : {duree:.2f} h | sauvegarde : {sortie}")

    tracking.log(
        {
            "bpb_before": bpb_avant["bpb"],
            "bpb_after": bpb_apres["bpb"],
            "bpb_gain_pct": gain,
            "tokens_total": int(trainer.state.global_step) * tokens_par_pas,
            "throughput_tokens_per_second": (
                int(trainer.state.global_step) * tokens_par_pas / max(duree * 3600, 1)
            ),
            "peak_vram_mib": peak_vram["allocated_mib"] if peak_vram else 0,
            "gate_status": 1,
        }
    )
    tracking.finish(success=True)
    del model, trainer, ds_train, ds_eval
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
