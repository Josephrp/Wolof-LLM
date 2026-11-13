"""Chargement du modele et configuration de l'entrainement."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _attention_dispo(demandee):
    """Retourne l'implementation d'attention utilisable.

    FlashAttention-2 apporte un gain important sur H100 mais n'est pas toujours
    installe : on retombe silencieusement sur l'implementation par defaut.
    """
    if demandee != "flash_attention_2":
        return demandee
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        print("  flash_attn absent -> attention par defaut "
              "(installer flash-attn pour accelerer)")
        return "sdpa"


def charger(chemin_modele, chemin_tokenizer, methode, attention="sdpa", verifier=True):
    """Charge le couple (tokenizer, modele) pret a entrainer.

    Le modele etendu est charge DEJA INITIALISE depuis le disque : la chirurgie
    des embeddings (deplacement des tokens speciaux, mean-pooling, remapping des
    identifiants) a ete faite une seule fois par 02_build_tokenizer.py.

    Refaire cette chirurgie ici serait une source de bugs : la selection naive
    des nouveaux tokens exclut les 256 tokens speciaux (ils existent deja dans le
    vocabulaire de base), dont les lignes d'embedding resteraient alors non
    initialisees — `eos` se retrouverait avec un vecteur aleatoire.
    """
    tok = AutoTokenizer.from_pretrained(chemin_tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn = _attention_dispo(attention)
    commun = {"device_map": {"": 0}, "attn_implementation": attn}

    if methode == "lora":
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            chemin_modele, quantization_config=bnb, **commun)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            chemin_modele, dtype=torch.bfloat16, **commun)

    model.config.use_cache = False

    if verifier:
        assert model.config.vocab_size == len(tok), (
            f"incoherence vocab : modele {model.config.vocab_size} "
            f"vs tokenizer {len(tok)}")
        eos = model.config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        print(f"  vocabulaire {len(tok):,} | eos_id={eos} -> {tok.decode([eos])!r}")
        print(f"  attention : {attn}")

    return tok, model


def preparer_lora(model, cfg, gradient_checkpointing=True):
    """Applique LoRA.

    `modules_to_save` est INDISPENSABLE quand le tokenizer est etendu : par
    defaut LoRA gele les embeddings, et les nouveaux tokens wolof ne seraient
    JAMAIS entraines. L'experience echouerait en silence, sans message d'erreur.

    Note : les embeddings de SmolLM3 sont *tied*. Passer embed_tokens et lm_head
    les dissocie — PEFT en cree deux copies independantes, entrainees en pleine
    precision. C'est le poste memoire dominant de ces runs.

    Le gradient checkpointing coute environ 30 % de vitesse. Il est indispensable
    sur petit GPU, mais peut etre desactive quand la memoire est confortable.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.get("r", 32),
        lora_alpha=cfg.get("alpha", 64),
        lora_dropout=cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]),
        modules_to_save=cfg.get("modules_to_save") or None,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def preparer_full(model, gradient_checkpointing=True):
    """Full fine-tuning : tous les parametres sont deja entrainables."""
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    n_e = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_t = sum(p.numel() for p in model.parameters())
    print(f"  parametres entrainables : {n_e:,} / {n_t:,} ({100*n_e/n_t:.0f} %)")
    return model

# ═══════════════════════════════════════════════════════════════════
#  Learning rates differencies et phase d'echauffement des embeddings
# ═══════════════════════════════════════════════════════════════════

NOMS_EMBEDDINGS = ("embed_tokens", "lm_head")


def _est_embedding(nom):
    return any(cle in nom for cle in NOMS_EMBEDDINGS)


def groupes_parametres(model, lr, lr_embeddings, weight_decay=0.0):
    """Separe les parametres en deux groupes : embeddings et backbone.

    POURQUOI. Un embedding neuf, initialise par mean-pooling, est encore
    approximatif et doit bouger BEAUCOUP. Un poids de backbone est deja
    entraine et ne doit bouger QUE PEU. Un learning rate unique force un
    compromis perdant : trop petit, les nouveaux embeddings ne convergent pas ;
    trop grand, le backbone se degrade.

    C'est le diagnostic de l'echec du run full fine-tuning + tokenizer etendu
    (BPB 0,8926 contre 0,7521 pour le tokenizer par defaut).

    Retourne None si lr_embeddings n'est pas renseigne : le comportement par
    defaut (un seul learning rate) est alors strictement inchange.
    """
    if not lr_embeddings:
        return None

    emb, backbone = [], []
    for nom, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (emb if _est_embedding(nom) else backbone).append(param)

    if not emb:
        print("  aucun parametre d'embedding entrainable -> LR unique conserve")
        return None

    n_emb = sum(p.numel() for p in emb)
    n_bb = sum(p.numel() for p in backbone)
    print(f"  LR differencies : embeddings {lr_embeddings:g} ({n_emb:,} params) "
          f"| backbone {lr:g} ({n_bb:,} params)")

    groupes = [{"params": emb, "lr": float(lr_embeddings), "weight_decay": 0.0}]
    if backbone:
        groupes.append({"params": backbone, "lr": float(lr),
                        "weight_decay": float(weight_decay)})
    return groupes


def geler_backbone(model):
    """Ne laisse entrainables que les embeddings (phase d'echauffement).

    Recette documentee par Chinese-LLaMA (arXiv 2304.08177) et Meltemi
    (arXiv 2407.20743) : integrer les nouveaux vecteurs sans perturber le
    modele original, avant de degeler l'ensemble.
    """
    n = 0
    for nom, param in model.named_parameters():
        garde = _est_embedding(nom)
        param.requires_grad = garde
        if garde:
            n += param.numel()
    print(f"  backbone gele — {n:,} parametres d'embedding entrainables")
    return model


def degeler_backbone(model):
    """Rend tous les parametres entrainables (fin de l'echauffement)."""
    for param in model.parameters():
        param.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  backbone degele — {n:,} parametres entrainables")
    return model
