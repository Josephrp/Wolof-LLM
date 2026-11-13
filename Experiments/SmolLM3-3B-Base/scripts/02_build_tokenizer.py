#!/usr/bin/env python3
"""Construit le tokenizer wolof etendu et le modele initialise correspondant.

Taille de vocabulaire fixee a 4 000 (voir README : l'ablation 4k/8k a montre
que les deux sont equivalents apres entrainement ; le 4 000 est retenu car il
atteint le meme BPB avec deux fois moins d'embeddings a entrainer).

    python scripts/02_build_tokenizer.py --n-tokens 4000
    python scripts/02_build_tokenizer.py --config configs/tokenizer-ext4000.yaml

Produit :
  <output_root>/tokenizers/smollm3_wolof_ext_<N>/    tokenizer + extension_meta.json
  <output_root>/models/smollm3-wolof-ext<N>-init/    modele redimensionne et initialise
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import charger_splits
from common.metrics import compute_fertility

MOTS_TEST = ["jefandikoo", "new", "wooral", "cakuteef", "jewrinu", "reew", "waaw"]


def selectionner_candidats(base_tok, textes, n_echantillon, seed):
    """Selectionne les formes de mots entiers les plus rentables.

    Le tokenizer de SmolLM3 a `ignore_merges = True` : il cherche le mot entier
    dans le vocabulaire AVANT d'appliquer les regles de fusion. Consequences :

      1. Ajouter une forme de mot entier suffit a obtenir 1 token, sans toucher
         aux merges.
      2. La monotonicite est garantie PAR CONSTRUCTION : un mot ajoute devient
         1 token, un mot absent suit exactement son chemin d'avant.
      3. Ajouter des sous-mots serait inutile : un sous-mot n'est atteignable
         que par les merges, il resterait une ligne d'embedding morte.

    Le tri se fait par gain attendu = frequence x (nb_tokens - 1), c'est-a-dire
    le nombre de tokens reellement economises sur le corpus.
    """
    import random
    random.seed(seed)
    echantillon = random.sample(textes, min(n_echantillon, len(textes)))

    pre = base_tok.backend_tokenizer.pre_tokenizer
    cnt = Counter()
    for t in echantillon:
        for chunk, _ in pre.pre_tokenize_str(t):
            cnt[chunk] += 1
    print(f"  segments distincts : {len(cnt):,}")

    base_vocab = set(base_tok.get_vocab())
    dec = base_tok.backend_tokenizer.decoder
    candidats = []
    for chunk, freq in cnt.most_common():
        if chunk in base_vocab:
            continue
        texte = dec.decode([chunk])
        n_tok = len(base_tok.encode(texte, add_special_tokens=False))
        if n_tok < 2:
            continue
        brut = texte.strip()
        if len(brut) < 2 or not any(c.isalpha() for c in brut):
            continue
        candidats.append((chunk, freq, n_tok, freq * (n_tok - 1)))
    candidats.sort(key=lambda x: -x[3])
    print(f"  candidats : {len(candidats):,}")
    return candidats


def construire_tokenizer(base_id, candidats, n_add):
    """Ajoute les tokens au vocabulaire avec REINDEXATION des identifiants.

    SmolLM3 range ses tokens ainsi : BPE aux ids 0..V-1, speciaux aux ids
    V..V+255. Injecter naivement de nouveaux tokens dans model.vocab cree soit
    une COLLISION avec les speciaux, soit un TROU de numerotation — la
    bibliotheque `tokenizers` ne supporte ni l'un ni l'autre (symptome observe :
    des mots wolof decodes en <|reserved_special_token_N|>).

    Solution : nouveaux tokens contigus juste apres le vocab BPE (V..V+N-1),
    speciaux decales au-dessus (V+N..V+N+255). Le id_map retourne est
    INDISPENSABLE pour deplacer les lignes d'embedding correspondantes.
    """
    tok = AutoTokenizer.from_pretrained(base_id)
    j = json.loads(tok.backend_tokenizer.to_str())
    v = j["model"]["vocab"]
    V = max(v.values()) + 1
    specials = sorted(j.get("added_tokens", []), key=lambda a: a["id"])

    nouveaux = [c for c, _, _, _ in candidats[:n_add] if c not in v]
    N = len(nouveaux)
    for i, s in enumerate(nouveaux):
        v[s] = V + i

    id_map = {}
    for k, a in enumerate(specials):
        id_map[a["id"]] = V + N + k
        a["id"] = V + N + k

    j["model"]["vocab"] = v
    j["added_tokens"] = specials

    d = tempfile.mkdtemp()
    tok.save_pretrained(d)
    with open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)
    for fn in ("vocab.json", "merges.txt"):   # eviter le reload des fichiers legacy
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)

    return AutoTokenizer.from_pretrained(d), {"V": V, "N": N, "id_map": id_map}


def valider(base_tok, ext_tok, textes_eval):
    """Trois validations bloquantes."""
    ok = True

    # 1) monotonicite : aucun mot ne doit etre degrade
    mots = []
    for t in textes_eval:
        for w in str(t).split():
            c = w.strip(".,;:!?\"'()[]{}<<>>---...0123456789")
            if c and c.isalpha():
                mots.append(c)
    pire = mieux = 0
    for c in mots[:20000]:
        d = len(base_tok.encode(" " + c, add_special_tokens=False))
        e = len(ext_tok.encode(" " + c, add_special_tokens=False))
        if e > d:
            pire += 1
        elif e < d:
            mieux += 1
    print(f"  monotonicite : pire={pire} mieux={mieux}", "OK" if pire == 0 else "ECHEC")
    ok &= (pire == 0)

    # 2) round-trip, compare au tokenizer de base
    #    clean_up_tokenization_spaces=False : avec True, le decodeur de SmolLM3
    #    retouche les espaces et fait echouer le test meme sur la base.
    def rt(tk):
        return sum(tk.decode(tk.encode(t, add_special_tokens=False),
                             clean_up_tokenization_spaces=False) == t
                   for t in textes_eval[:500])
    rb, re_ = rt(base_tok), rt(ext_tok)
    print(f"  round-trip   : base {rb}/500 | etendu {re_}/500",
          "OK" if re_ >= rb else "ECHEC")
    ok &= (re_ >= rb)

    # 3) non-regression FR/EN/code — crucial pour les runs multilingues
    for phrase in ["Le chat dort sur le canape.",
                   "The quick brown fox jumps over the lazy dog.",
                   "def add(a, b): return a + b"]:
        a = base_tok.encode(phrase, add_special_tokens=False)
        b = ext_tok.encode(phrase, add_special_tokens=False)
        if a != b:
            print(f"  REGRESSION sur : {phrase}")
            ok = False
    print("  non-regression FR/EN/code : OK" if ok else "  non-regression : ECHEC")
    return ok


def initialiser_modele(base_id, base_tok, ext_tok, meta, sortie):
    """Redimensionne, deplace les speciaux, initialise en mean-pooling.

    ATTENTION — ne PAS recalibrer la norme des nouveaux embeddings.

    Le mean-pooling produit une norme reduite (ratio ~0,5 : la moyenne de k
    vecteurs independants voit sa norme chuter en racine de k). C'est CORRECT.

    Les embeddings de SmolLM3 sont *tied* : la meme matrice sert d'entree et de
    projection de sortie, ou le logit vaut h . e. Ramener la norme a 1,0 double
    les logits des nouveaux tokens. Mesure sur ce modele : 99,9 % des
    predictions tombaient sur les nouveaux tokens, BPB 5,07 au lieu de 2,16,
    entropie croisee des tokens anciens 21,6 (pire que le hasard).

    La norme reduite agit comme une regularisation implicite.
    """
    V, N, id_map = meta["V"], meta["N"], meta["id_map"]

    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32,
                                                 device_map="cpu")
    emb_old = model.get_input_embeddings().weight.data.clone()
    model.resize_token_embeddings(len(ext_tok))
    emb = model.get_input_embeddings().weight.data

    with torch.no_grad():
        for old, new in id_map.items():          # deplacer les speciaux
            emb[new] = emb_old[old]

        sans = 0
        for tok_str, nid in {t: i for t, i in ext_tok.get_vocab().items()
                             if V <= i < V + N}.items():
            texte = ext_tok.convert_tokens_to_string([tok_str])
            sub = [s for s in base_tok.encode(texte, add_special_tokens=False) if s < V]
            if not sub:
                sans += 1
                continue
            emb[nid] = emb_old[sub].mean(dim=0)  # mean-pooling
    print(f"  tokens sans sous-tokens exploitables : {sans}")

    ratio = emb[V:V + N].norm(dim=1).mean().item() / emb_old[:V].norm(dim=1).mean().item()
    print(f"  ratio de norme : {ratio:.3f}",
          "OK (~0,5 attendu, NE PAS recalibrer)" if ratio < 0.7 else "ANOMALIE")

    def remap(x):
        if isinstance(x, list):
            return [id_map.get(i, i) for i in x]
        return id_map.get(x, x) if isinstance(x, int) else x

    for cfg in [model.config, model.generation_config]:
        if cfg is None:
            continue
        for champ in ["eos_token_id", "bos_token_id", "pad_token_id",
                      "decoder_start_token_id"]:
            a = getattr(cfg, champ, None)
            if a is not None:
                setattr(cfg, champ, remap(a))
    model.config.vocab_size = len(ext_tok)

    Path(sortie).mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16)
    model.save_pretrained(sortie, safe_serialization=True)
    ext_tok.save_pretrained(sortie)
    del model, emb_old
    return ratio


def sous_output_root(chemin, output_root):
    p = Path(chemin)
    return p if p.is_absolute() else output_root / p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Config YAML optionnelle pour lire base_id, data_root et output_root.")
    p.add_argument("--racine", default=".")
    p.add_argument("--data-root", default=None,
                   help="Dossier contenant les splits parquet (défaut: <racine>/data).")
    p.add_argument("--output-root", default=None,
                   help="Dossier où écrire tokenizer et modèle init (défaut: <racine>).")
    p.add_argument("--base-id", default=None)
    p.add_argument("--n-tokens", type=int, default=None)
    p.add_argument("--n-echantillon", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    racine = Path(a.racine)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8")) if a.config else {}
    base_id = a.base_id or cfg.get("base_id") or "HuggingFaceTB/SmolLM3-3B-Base"
    data_root_cfg = a.data_root or cfg.get("data_root")
    output_root_cfg = a.output_root or cfg.get("output_root")
    data_root = Path(data_root_cfg) if data_root_cfg else None
    output_root = Path(output_root_cfg) if output_root_cfg else racine
    n_tokens = a.n_tokens or cfg.get("n_tokens", 4000)
    n_echantillon = a.n_echantillon or cfg.get("n_echantillon", 300000)
    seed = a.seed if a.seed is not None else cfg.get("seed", 42)
    textes_train, textes_eval = charger_splits(racine, data_root=data_root)

    base_tok = AutoTokenizer.from_pretrained(base_id)
    # SmolLM3 : vocab_size annonce 128000 mais len(tokenizer) vaut 128256.
    # Dimensionner sur vocab_size produit des indices hors bornes.
    print(f"tokenizer de base : vocab_size={base_tok.vocab_size} "
          f"len={len(base_tok)} (utiliser len)")

    fert_base = compute_fertility(base_tok, textes_eval)["tokens_par_mot"]
    print(f"fertilite de base : {fert_base:.3f} tokens/mot\n")

    print("selection des candidats")
    candidats = selectionner_candidats(base_tok, textes_train, n_echantillon, seed)

    print(f"\nconstruction du tokenizer etendu ({n_tokens} tokens)")
    ext_tok, meta = construire_tokenizer(base_id, candidats, n_tokens)
    fert_ext = compute_fertility(ext_tok, textes_eval)["tokens_par_mot"]
    print(f"  vocabulaire : {len(base_tok):,} -> {len(ext_tok):,}")
    print(f"  fertilite   : {fert_base:.3f} -> {fert_ext:.3f} "
          f"({100*(1-fert_ext/fert_base):.1f} % de gain)")

    print("\nvalidations")
    if not valider(base_tok, ext_tok, textes_eval):
        raise SystemExit("ECHEC des validations — tokenizer non sauvegarde")

    tok_dir = sous_output_root(
        cfg.get("tokenizer_path", f"tokenizers/smollm3_wolof_ext_{meta['N']}"),
        output_root)
    tok_dir.mkdir(parents=True, exist_ok=True)
    ext_tok.save_pretrained(tok_dir)

    print("\ninitialisation des embeddings")
    model_dir = sous_output_root(
        cfg.get("model_path", f"models/smollm3-wolof-ext{meta['N']}-init"),
        output_root)
    ratio = initialiser_modele(base_id, base_tok, ext_tok, meta, model_dir)

    with open(tok_dir / "extension_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": base_id,
            "methode": "mots entiers par frequence (ignore_merges=True), reindexation",
            "n_added": meta["N"], "vocab_bpe_size": meta["V"],
            "vocab_total_before": len(base_tok), "vocab_total_after": len(ext_tok),
            "id_map_specials": {str(k): v for k, v in meta["id_map"].items()},
            "fertility_default": fert_base, "fertility_extended": fert_ext,
            "ratio_norme": ratio, "seed": seed,
        }, f, indent=2, ensure_ascii=False)

    print(f"\ntokenizer : {tok_dir}")
    print(f"modele    : {model_dir}")


if __name__ == "__main__":
    main()
