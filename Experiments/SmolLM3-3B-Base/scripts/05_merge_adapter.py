#!/usr/bin/env python3
"""Fusionne un adaptateur LoRA dans le modele de base, embeddings tied compris.

    python3 scripts/05_merge_adapter.py --run wolof-ext4000-lora \\
        --output-root /mnt/volume_w2l8nkp/galsenllm/OutputOfSmolLM_Model \\
        --data-root  /mnt/volume_w2l8nkp/galsenllm/data

POURQUOI CE SCRIPT EXISTE

SmolLM3 a `tie_word_embeddings=True` : une seule matrice sert d'entree et de
projection de sortie. Quand `modules_to_save` contient embed_tokens ET lm_head,
PEFT cree deux copies independantes et les entraine separement — il DE-TIE les
poids de fait — mais ne met pas a jour `tie_word_embeddings` dans la config.
D'ou l'avertissement rencontre a l'entrainement :

    Model has `tie_word_embeddings=True` and a tied layer is part of the
    adapter, but `ensure_weight_tying` is not set to True. This can lead to
    complications, for example when merging the adapter [...]

Sans correction, le modele fusionne peut re-tier les deux matrices au
rechargement et perdre la moitie de ce qui a ete appris — silencieusement.

CE QUE FAIT LE SCRIPT

1. charge le modele de base en bf16 (PAS en 4 bits : on fusionne en pleine
   precision, sinon la quantification degrade les poids fusionnes) ;
2. applique l'adaptateur, puis inspecte si entree et sortie ont diverge ;
3. fusionne, et fixe `tie_word_embeddings` a la valeur qui correspond
   reellement aux poids ;
4. VERIFIE que le BPB du modele fusionne egale celui d'avant fusion.

Le point 4 est le garde-fou : une fusion silencieusement fausse se voit
immediatement sur le BPB.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import charger_splits
from common.metrics import compute_bpb


def etat_tying(model):
    """Retourne (tied_en_memoire, poids_identiques)."""
    entree = model.get_input_embeddings().weight
    sortie_mod = model.get_output_embeddings()
    if sortie_mod is None:
        return True, True
    sortie = sortie_mod.weight
    memes_donnees = entree.data_ptr() == sortie.data_ptr()
    if memes_donnees:
        return True, True
    identiques = bool(torch.equal(entree.detach().cpu(), sortie.detach().cpu()))
    return False, identiques


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="nom du run, ex. wolof-ext4000-lora")
    p.add_argument("--racine", default=".")
    p.add_argument("--output-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--n-textes", type=int, default=300,
                   help="textes d'evaluation pour la verification de non-regression")
    p.add_argument("--tolerance", type=float, default=0.002,
                   help="ecart de BPB tolere entre avant et apres fusion")
    a = p.parse_args()

    racine = Path(a.racine)
    output_root = Path(a.output_root) if a.output_root else racine
    dossier_run = output_root / "runs" / a.run
    chemin_adapter = dossier_run / "adapter"

    if not chemin_adapter.is_dir():
        raise SystemExit(f"adaptateur introuvable : {chemin_adapter}")

    journal = json.load(open(dossier_run / "journal.json", encoding="utf-8"))
    if journal.get("methode") != "lora":
        raise SystemExit(f"{a.run} n'est pas un run LoRA — rien a fusionner")
    chemin_base = journal["modele"]

    print(f"{'='*70}\n  fusion : {a.run}\n{'='*70}")
    print(f"  modele de base : {chemin_base}")
    print(f"  adaptateur     : {chemin_adapter}")

    # --- chargement en bf16, pas en 4 bits --------------------------------
    tok = AutoTokenizer.from_pretrained(str(chemin_adapter))
    base = AutoModelForCausalLM.from_pretrained(
        chemin_base, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")

    tied_config = bool(getattr(base.config, "tie_word_embeddings", False))
    print(f"  tie_word_embeddings dans la config de base : {tied_config}")

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(chemin_adapter))
    model.eval()

    tied_avant, identiques_avant = etat_tying(model)
    print(f"  avant fusion — memes donnees : {tied_avant} | "
          f"valeurs identiques : {identiques_avant}")

    # --- BPB de reference AVANT fusion ------------------------------------
    _, textes_eval = charger_splits(
        racine, data_root=Path(a.data_root) if a.data_root else None)
    textes = textes_eval[:a.n_textes]

    bpb_avant = compute_bpb(model, tok, textes)
    print(f"  BPB avant fusion : {bpb_avant['bpb']:.4f}")

    # --- fusion -----------------------------------------------------------
    fusionne = model.merge_and_unload()
    tied_apres, identiques_apres = etat_tying(fusionne)
    print(f"  apres fusion  — memes donnees : {tied_apres} | "
          f"valeurs identiques : {identiques_apres}")

    # La config doit refleter la REALITE des poids, pas l'intention d'origine.
    # Si entree et sortie ont diverge, les laisser declarees tied ferait
    # perdre la matrice de sortie au rechargement.
    doit_etre_tied = tied_apres or identiques_apres
    if fusionne.config.tie_word_embeddings != doit_etre_tied:
        print(f"  correction : tie_word_embeddings "
              f"{fusionne.config.tie_word_embeddings} -> {doit_etre_tied}")
        fusionne.config.tie_word_embeddings = doit_etre_tied
    else:
        print(f"  tie_word_embeddings deja correct ({doit_etre_tied})")

    # --- verification de non-regression -----------------------------------
    bpb_apres = compute_bpb(fusionne, tok, textes)
    ecart = abs(bpb_apres["bpb"] - bpb_avant["bpb"])
    print(f"  BPB apres fusion : {bpb_apres['bpb']:.4f}  (ecart {ecart:+.5f})")

    if ecart > a.tolerance:
        raise SystemExit(
            f"\n  ECHEC : la fusion a change le BPB de {ecart:.5f} "
            f"(tolerance {a.tolerance}).\n"
            f"  Le modele fusionne n'est PAS equivalent a l'adaptateur.\n"
            f"  Ne pas l'utiliser pour le SFT — verifier la gestion des "
            f"embeddings tied.")

    # --- sauvegarde --------------------------------------------------------
    sortie = output_root / "models" / f"{a.run}-merged"
    sortie.mkdir(parents=True, exist_ok=True)
    fusionne.save_pretrained(str(sortie), safe_serialization=True)
    tok.save_pretrained(str(sortie))

    json.dump({
        "run_source": a.run,
        "modele_base": str(chemin_base),
        "adaptateur": str(chemin_adapter),
        "tie_word_embeddings_config_base": tied_config,
        "tied_avant_fusion": tied_avant,
        "tied_apres_fusion": doit_etre_tied,
        "bpb_avant_fusion": bpb_avant["bpb"],
        "bpb_apres_fusion": bpb_apres["bpb"],
        "ecart_bpb": ecart,
        "n_textes_verification": a.n_textes,
    }, open(sortie / "fusion_meta.json", "w"), indent=2, ensure_ascii=False)

    print(f"\n  fusion validee — modele complet : {sortie}")
    print(f"  utilisable directement comme point de depart du SFT")


if __name__ == "__main__":
    main()
