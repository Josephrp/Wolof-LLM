"""Chargement du modele et configuration de l'entrainement."""

from __future__ import annotations

import os

import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def enable_cuda_training_speedups() -> None:
    """Enable TF32 / matmul precision knobs that speed Ampere+ training."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def causal_conv1d_available() -> bool:
    """True when both LFM short-conv CUDA entry points import cleanly."""
    return causal_conv1d_fn is not None and causal_conv1d_update is not None


def require_causal_conv1d(allow_fallback: bool | None = None) -> None:
    """Fail closed on GPU hosts when the optimized LFM short-conv kernel is missing.

    Transformers otherwise prints a fallback warning and trains much more slowly.
    Set ``LFM_ALLOW_SLOW_CONV=1`` only for emergency CPU/debug paths.
    """
    if allow_fallback is None:
        allow_fallback = os.environ.get("LFM_ALLOW_SLOW_CONV") == "1"
    if causal_conv1d_available():
        print("  causal_conv1d: optimized CUDA kernel available")
        return
    message = (
        "causal_conv1d is not installed; LFM2 short-conv would fall back to a "
        "much slower PyTorch reference path. Install the train-group "
        "`causal-conv1d` package on Linux CUDA hosts before restarting training."
    )
    if allow_fallback or not torch.cuda.is_available():
        print(f"  WARNING: {message}")
        return
    raise RuntimeError(message)


def _attention_dispo(demandee):
    """Retourne l'implementation d'attention utilisable.

    FlashAttention-2 apporte un gain important sur H100 mais n'est pas toujours
    installe : on retombe silencieusement sur l'implementation par defaut.
    """
    if demandee != "flash_attention_2":
        return demandee
    if flash_attn is not None:
        return "flash_attention_2"
    print(
        "  flash_attn absent -> attention par defaut "
        "(installer flash-attn pour accelerer)"
    )
    return "sdpa"


def lire_tie_flag(config) -> bool:
    """Lit tie_embedding (LFM 1.2B/350M) ou tie_word_embeddings (2.6B / HF)."""
    if hasattr(config, "tie_embedding") and config.tie_embedding is not None:
        return bool(config.tie_embedding)
    if (
        hasattr(config, "tie_word_embeddings")
        and config.tie_word_embeddings is not None
    ):
        return bool(config.tie_word_embeddings)
    return False


def ecrire_tie_flag(config, value: bool) -> None:
    """Ecrit le flag de tying present sur le config (ou les deux si absents)."""
    wrote = False
    if hasattr(config, "tie_embedding"):
        config.tie_embedding = bool(value)
        wrote = True
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = bool(value)
        wrote = True
    if not wrote:
        # Prefer LFM-native name when neither attribute exists yet
        config.tie_embedding = bool(value)
        config.tie_word_embeddings = bool(value)


# Flexible embedding / head name fragments for LFM and Llama-like modules.
NOMS_EMBEDDINGS = (
    "embed_tokens",
    "wte",
    "tok_embeddings",
    "lm_head",
    "output",
    "embed_in",
    "embed_out",
)


def _est_embedding(nom: str) -> bool:
    """True si le parametre appartient aux embeddings / lm_head (noms flexibles)."""
    lower = nom.lower()
    return any(cle in lower for cle in NOMS_EMBEDDINGS)


def verifier_vocab_compatible(model_vocab_size: int, tokenizer) -> None:
    """Accept padded model vocabularies while rejecting out-of-range token IDs."""
    tokenizer_ids = tokenizer.get_vocab().values()
    max_token_id = max(tokenizer_ids, default=-1)
    if len(tokenizer) > model_vocab_size or max_token_id >= model_vocab_size:
        raise ValueError(
            f"incoherence vocab : modele {model_vocab_size}, "
            f"tokenizer len={len(tokenizer)}, max_id={max_token_id}"
        )


def lora_requires_weight_tying(config, modules_to_save) -> bool:
    """Tell PEFT to preserve tying when adapter artifacts include tied layers."""
    return bool(modules_to_save and lire_tie_flag(config))


def charger(chemin_modele, chemin_tokenizer, methode, attention="sdpa", verifier=True):
    """Charge le couple (tokenizer, modele) pret a entrainer.

    Le modele etendu est charge DEJA INITIALISE depuis le disque : la chirurgie
    des embeddings (mean-pooling, resize) a ete faite une seule fois par
    scripts/clm/02_build_tokenizer.py.
    """
    enable_cuda_training_speedups()
    require_causal_conv1d()
    tok = AutoTokenizer.from_pretrained(chemin_tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn = _attention_dispo(attention)
    cuda = torch.cuda.is_available()
    commun = {
        "device_map": {"": 0} if cuda else "cpu",
        "attn_implementation": attn,
    }

    if methode == "lora":
        if not cuda:
            raise RuntimeError(
                "LoRA training uses bitsandbytes 4-bit loading and requires CUDA. "
                "Run this config on a GPU host."
            )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            chemin_modele, quantization_config=bnb, **commun
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            chemin_modele,
            dtype=torch.bfloat16 if cuda else torch.float32,
            **commun,
        )

    model.config.use_cache = False

    if verifier:
        verifier_vocab_compatible(model.config.vocab_size, tok)
        eos = model.config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        print(f"  vocabulaire {len(tok):,} | eos_id={eos} -> {tok.decode([eos])!r}")
        print(f"  attention : {attn}")
        print(f"  tie_embedding : {lire_tie_flag(model.config)}")

    return tok, model


def preparer_lora(model, cfg, gradient_checkpointing=True):
    """Applique LoRA.

    `modules_to_save` est INDISPENSABLE quand le tokenizer est etendu : par
    defaut LoRA gele les embeddings, et les nouveaux tokens wolof ne seraient
    JAMAIS entraines.
    """
    if (
        LoraConfig is None
        or get_peft_model is None
        or prepare_model_for_kbit_training is None
    ):
        raise RuntimeError("peft is required for LoRA preparation")

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    modules_to_save = cfg.get("modules_to_save") or None
    lora = LoraConfig(
        r=cfg.get("r", 32),
        lora_alpha=cfg.get("alpha", 64),
        lora_dropout=cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        modules_to_save=modules_to_save,
        ensure_weight_tying=lora_requires_weight_tying(model.config, modules_to_save),
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
    print(f"  parametres entrainables : {n_e:,} / {n_t:,} ({100 * n_e / n_t:.0f} %)")
    return model


def groupes_parametres(model, lr, lr_embeddings, weight_decay=0.0):
    """Separe les parametres en deux groupes : embeddings et backbone."""
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
    print(
        f"  LR differencies : embeddings {lr_embeddings:g} ({n_emb:,} params) "
        f"| backbone {lr:g} ({n_bb:,} params)"
    )

    groupes = [{"params": emb, "lr": float(lr_embeddings), "weight_decay": 0.0}]
    if backbone:
        groupes.append(
            {
                "params": backbone,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            }
        )
    return groupes


def geler_backbone(model):
    """Ne laisse entrainables que les embeddings (phase d'echauffement)."""
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
