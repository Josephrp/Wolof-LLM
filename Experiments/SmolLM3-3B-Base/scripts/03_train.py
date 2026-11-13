#!/usr/bin/env python3
"""Execute UN run de continual pretraining, pilote par un fichier de config.

Un seul chemin de code sert les quatre experiences de la grille : c'est ce qui
garantit qu'aucune divergence involontaire ne s'introduit entre deux runs.

    python3 scripts/03_train.py --config configs/wolof-ext4000-lora.yaml
    python3 scripts/03_train.py --config configs/wolof-default-lora.yaml --smoke

L'option --smoke reduit a 10 pas sur 20 000 textes : elle valide la chaine
complete (chargement, packing, entrainement, evaluation, sauvegarde) en
quelques minutes, sans attendre un run complet.
"""

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                          default_data_collator)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import charger_splits, paqueter
from common.journal import ecrire, extraire_courbes, tracer
from common.metrics import compute_bpb
from common.model import (charger, degeler_backbone, geler_backbone,
                          groupes_parametres, preparer_full, preparer_lora)


def resoudre_artefact(chemin, racine, output_root):
    """Résout les chemins locaux relatifs en privilégiant output_root."""
    s = str(chemin)
    if s.startswith(("HuggingFace", "/")):
        return s
    dans_output = output_root / s
    if dans_output.exists():
        return str(dans_output)
    return str(racine / s)


class TrainerLRDifferencie(Trainer):
    """Trainer qui applique un learning rate distinct aux embeddings.

    On surcharge create_optimizer plutot que de passer un optimiseur tout fait :
    le Trainer conserve ainsi sa propre construction du scheduler, ce qui evite
    les incompatibilites entre versions de transformers.
    """

    groupes = None   # rempli avant l'appel a train()

    def create_optimizer(self):
        if self.optimizer is None and self.groupes:
            cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            kwargs.pop("lr", None)
            kwargs.pop("weight_decay", None)
            self.optimizer = cls(self.groupes, lr=self.args.learning_rate, **kwargs)
            return self.optimizer
        return super().create_optimizer()


def arguments_entrainement(sortie, e, gc_actif, max_steps=None, lr=None,
                           warmup_ratio=None, arret_anticipe=False):
    """Construit TrainingArguments avec compatibilite transformers ancien/recent.

    max_steps, lr et warmup_ratio permettent de reutiliser la meme fabrique
    pour la phase d'echauffement des embeddings.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(sortie),
        "max_steps": max_steps if max_steps is not None else e["max_steps"],
        "per_device_train_batch_size": e["micro_bs"],
        "gradient_accumulation_steps": e["grad_acc"],
        "learning_rate": float(lr if lr is not None else e["lr"]),
        "lr_scheduler_type": e.get("scheduler", "cosine"),
        "logging_steps": e.get("logging_steps", 25),
        "eval_steps": e.get("eval_steps", 50),
        "save_steps": e.get("save_steps", e["max_steps"]),
        "save_total_limit": e.get("save_total_limit", 2),
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": gc_actif,
        "optim": e.get("optim", "adamw_torch"),
        "report_to": "none",
        "seed": e["seed"],
    }

    ratio = warmup_ratio if warmup_ratio is not None else e.get("warmup_ratio", 0.03)
    if "warmup_ratio" in params:
        kwargs["warmup_ratio"] = ratio
    else:
        kwargs["warmup_steps"] = int(kwargs["max_steps"] * ratio)

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    # Arret anticipe : le Trainer doit pouvoir recharger le MEILLEUR checkpoint,
    # pas le dernier. Cela impose que sauvegarde et evaluation tombent aux memes
    # pas — sinon transformers refuse load_best_model_at_end.
    if arret_anticipe:
        kwargs["save_steps"] = kwargs["eval_steps"]
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
        kwargs["save_total_limit"] = max(kwargs.get("save_total_limit", 2), 2)

    return TrainingArguments(**kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--racine", default=".")
    p.add_argument("--smoke", action="store_true", help="run court de validation")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    racine = Path(a.racine)
    nom = cfg["run"]
    methode = cfg["methode"]
    e = cfg["entrainement"]

    if a.smoke:
        nom += "-smoke"
        e["max_steps"] = 10
    output_root = Path(cfg.get("output_root", racine))
    data_root = Path(cfg.get("data_root")) if cfg.get("data_root") else None
    sortie = output_root / "runs" / nom
    sortie.mkdir(parents=True, exist_ok=True)

    tokens_par_pas = e["micro_bs"] * e["grad_acc"] * e["seq_len"]
    print(f"{'='*70}\n  {nom}  [{methode}]\n{'='*70}")
    print(f"  {tokens_par_pas:,} tokens/pas | {e['max_steps']} pas = "
          f"{e['max_steps']*tokens_par_pas/1e6:.1f} M tokens")
    print(f"  learning rate : {e['lr']}")

    if torch.cuda.is_available():
        libre, total = torch.cuda.mem_get_info()
        print(f"  GPU : {torch.cuda.get_device_name(0)} "
              f"({libre/1e9:.1f}/{total/1e9:.1f} Go libres)")
        memoire_fraction = cfg.get("memoire_fraction")
        if memoire_fraction:
            torch.cuda.set_per_process_memory_fraction(float(memoire_fraction), 0)
            print(f"  memoire plafonnee a {float(memoire_fraction):.0%}")
        # Full fine-tuning d'un 3B en bf16 avec AdamW : ~37 Go hors activations
        # (poids 6 + gradients 6 + etats AdamW fp32 25).
        budget = total / 1e9 * (float(memoire_fraction) if memoire_fraction else 1.0)
        if methode == "full" and budget < 45:
            print(f"  ATTENTION : budget {budget:.0f} Go — envisager "
                  "optim: adamw_bnb_8bit ou reduire micro_bs")

    # --- modele -----------------------------------------------------------
    chemin_modele = resoudre_artefact(cfg.get("model") or cfg["base_id"],
                                      racine, output_root)
    chemin_tok = resoudre_artefact(cfg["tokenizer"], racine, output_root)

    tok, model = charger(chemin_modele, chemin_tok, methode,
                         attention=cfg.get("attention", "sdpa"))

    # --- donnees ----------------------------------------------------------
    textes_train, textes_eval = charger_splits(racine, data_root=data_root)
    limite = 20000 if a.smoke else None
    ds_train = paqueter(textes_train, tok, e["seq_len"], limite).shuffle(seed=e["seed"])
    ds_eval = paqueter(textes_eval, tok, e["seq_len"])
    print(f"  blocs : train {len(ds_train):,} | eval {len(ds_eval):,}")

    # --- BPB avant entrainement ------------------------------------------
    n_eval = 100 if a.smoke else 500
    bpb_avant = compute_bpb(model, tok, textes_eval[:n_eval])
    print(f"  BPB avant : {bpb_avant['bpb']:.4f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- preparation ------------------------------------------------------
    gc_actif = e.get("gradient_checkpointing", True)
    if methode == "lora":
        model = preparer_lora(model, cfg.get("lora", {}), gc_actif)
    else:
        model = preparer_full(model, gc_actif)

    lr_emb = e.get("lr_embeddings")
    pas_echauffement = int(e.get("warmup_embeddings_steps", 0) or 0)

    # --- Phase A (optionnelle) : echauffement des embeddings seuls ---------
    # Backbone gele, seuls les embeddings apprennent. Objectif : amener les
    # nouveaux vecteurs a un etat exploitable AVANT de toucher aux poids
    # pre-entraines, qu'un gradient bruite degraderait.
    # Reservee au full fine-tuning : en LoRA le backbone est deja gele par
    # construction, la phase n'aurait pas d'objet.
    duree_a = 0.0
    if pas_echauffement and methode == "full" and not a.smoke:
        print(f"\n  --- Phase A : echauffement des embeddings "
              f"({pas_echauffement} pas) ---")
        geler_backbone(model)
        args_a = arguments_entrainement(
            sortie / "phase_a", e, gc_actif,
            max_steps=pas_echauffement,
            lr=lr_emb or e["lr"],
            warmup_ratio=0.1)
        # Pas d'evaluation pendant l'echauffement des embeddings ->
        # evite l'erreur lorsque Trainer recoit args.eval_strategy='steps'
        if hasattr(args_a, "eval_strategy"):
            setattr(args_a, "eval_strategy", "no")
        else:
            setattr(args_a, "evaluation_strategy", "no")

        trainer_a = Trainer(model=model, args=args_a, train_dataset=ds_train,
                            data_collator=default_data_collator)
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

    # --- Phase B : entrainement principal ----------------------------------
    # patience > 0 : on pousse le run jusqu'a ce que la loss de validation
    # cesse de s'ameliorer, plutot que de fixer max_steps a l'avance.
    # max_steps devient alors un PLAFOND de securite, pas une cible.
    #
    # L'arret se fonde sur la loss de validation et non sur le BPB : a
    # l'interieur d'un meme run le tokenizer ne change pas, la loss est donc
    # un indicateur valide. Le BPB reste la metrique de comparaison ENTRE
    # runs, calcule en fin de parcours.
    patience = int(e.get("patience", 0) or 0)
    arret = patience > 0 and not a.smoke
    if arret:
        print(f"  arret anticipe actif : patience {patience} evaluations "
              f"(plafond {e['max_steps']} pas)")

    args = arguments_entrainement(sortie, e, gc_actif, arret_anticipe=arret)
    groupes = groupes_parametres(model, float(e["lr"]), lr_emb,
                                 weight_decay=args.weight_decay)
    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=float(e.get("seuil_amelioration", 0.0)))] if arret else None

    commun = dict(model=model, args=args, train_dataset=ds_train,
                  eval_dataset=ds_eval, data_collator=default_data_collator,
                  callbacks=callbacks)
    if groupes:
        trainer = TrainerLRDifferencie(**commun)
        trainer.groupes = groupes
    else:
        trainer = Trainer(**commun)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    res = trainer.train()
    duree = (time.time() - t0) / 3600 + duree_a

    bpb_apres = compute_bpb(model, tok, textes_eval[:n_eval])
    gain = 100 * (1 - bpb_apres["bpb"] / bpb_avant["bpb"])
    print(f"  BPB : {bpb_avant['bpb']:.4f} -> {bpb_apres['bpb']:.4f} ({gain:.1f} % de gain)")

    ltr, lval, llr = extraire_courbes(trainer)
    tracer(ltr, lval, llr, nom, sortie / "courbes.png")

    ecrire(sortie / "journal.json", {
        "run": nom, "methode": methode, "config": str(a.config),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "modele": str(chemin_modele), "tokenizer": str(chemin_tok),
        "vocab": len(tok), "smoke": a.smoke,
        "hyperparametres": {**e, "lora": cfg.get("lora") if methode == "lora" else None},
        "budget": {"pas_plafond": e["max_steps"],
                   "pas_effectues": int(trainer.state.global_step),
                   "arret_anticipe": arret and trainer.state.global_step < e["max_steps"],
                   "patience": patience,
                   "tokens_vus_M": int(trainer.state.global_step) * tokens_par_pas / 1e6,
                   "duree_heures": round(duree, 3),
                   "epochs_equivalent": round(
                       int(trainer.state.global_step) * e["micro_bs"] * e["grad_acc"]
                       / len(ds_train), 4)},
        "loss": {"train_premiere": ltr[0][1] if ltr else None,
                 "train_derniere": ltr[-1][1] if ltr else None,
                 "val_derniere": lval[-1][1] if lval else None,
                 "moyenne_run": res.metrics.get("train_loss"),
                 "courbe_train": ltr, "courbe_val": lval},
        "eval": {"split": "wolof_eval_clean_v1.parquet", "n_textes": n_eval,
                 "bpb_avant": bpb_avant["bpb"], "bpb_apres": bpb_apres["bpb"],
                 "ppl_avant": bpb_avant["ppl"], "ppl_apres": bpb_apres["ppl"],
                 "bpb_apres_echauffement": bpb_phase_a["bpb"] if bpb_phase_a else None},
        "phase_a": {"pas": pas_echauffement if bpb_phase_a else 0,
                    "duree_heures": round(duree_a, 3),
                    "lr": lr_emb or e["lr"]} if bpb_phase_a else None,
        "lr_differencies": bool(groupes),
    })

    if methode == "lora":
        model.save_pretrained(str(sortie / "adapter"))
    else:
        model.save_pretrained(str(sortie / "model"), safe_serialization=True)
    tok.save_pretrained(str(sortie / ("adapter" if methode == "lora" else "model")))

    if arret and trainer.state.global_step < e["max_steps"]:
        print(f"  arret anticipe au pas {trainer.state.global_step} "
              f"/ {e['max_steps']} — meilleur checkpoint recharge")
    print(f"  duree : {duree:.2f} h | sauvegarde : {sortie}")

    del model, trainer, ds_train, ds_eval
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
