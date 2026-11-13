"""
Script d'entraînement CLM (next token prediction) sur le corpus Wolof.
Toutes les valeurs viennent du fichier de config JSON — ne rien modifier ici pour changer une expérimentation.

Usage :
    python train_clm.py --config experiments/configs/clm/v1.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_token(env_var: str) -> str:
    token = os.environ.get(env_var)
    if not token:
        raise ValueError(f"Token HuggingFace introuvable dans la variable d'environnement '{env_var}'.")
    return token


def load_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_wolof_dataset(cfg: dict, hf_token: str):
    data_cfg = cfg["data"]
    print(f"Chargement de {data_cfg['hub_repo_id']} depuis HuggingFace Hub...")
    dataset = load_dataset(data_cfg["hub_repo_id"], split="train", token=hf_token)

    train_dataset = dataset.select(range(data_cfg["train_size"]))
    test_dataset = dataset.select(
        range(data_cfg["train_size"], data_cfg["train_size"] + data_cfg["test_size"])
    )
    return train_dataset, test_dataset


def build_lora_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )


def build_training_args(cfg: dict) -> SFTConfig:
    train_cfg = cfg["training"]
    return SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        dataset_text_field=cfg["data"]["text_column"],
    )


def save_metrics(trainer, cfg: dict) -> None:
    """Sauvegarde l'historique d'entraînement en CSV et JSON."""
    out_cfg = cfg["output"]
    log_history = trainer.state.log_history

    Path(out_cfg["metrics_csv"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out_cfg["stats_json"]).parent.mkdir(parents=True, exist_ok=True)

    with open(out_cfg["stats_json"], "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    all_keys = sorted({k for entry in log_history for k in entry.keys()})
    with open(out_cfg["metrics_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(log_history)

    print(f"Métriques sauvegardées : {out_cfg['metrics_csv']} et {out_cfg['stats_json']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier de config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hf_token = get_token(cfg["output"]["hf_token_env_var"])

    # 1. Modèle + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 2. Données
    train_dataset, test_dataset = load_wolof_dataset(cfg, hf_token)

    # 3. LoRA + config d'entraînement
    lora_config = build_lora_config(cfg)
    training_args = build_training_args(cfg)

    # 4. Entraînement
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()

    # 5. Sauvegarde locale de l'adapter
    out_cfg = cfg["output"]
    Path(out_cfg["local_adapter_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_cfg["local_adapter_dir"])
    tokenizer.save_pretrained(out_cfg["local_adapter_dir"])
    print(f"Adapter sauvegardé localement : {out_cfg['local_adapter_dir']}")

    # 6. Sauvegarde des métriques (CSV + JSON)
    save_metrics(trainer, cfg)

    # 7. Publication sur HuggingFace Hub (modèle de base pour le SFT)
    if out_cfg["push_to_hub"]:
        login(token=hf_token)
        print(f"Publication sur {out_cfg['hub_repo_id']}...")
        trainer.model.push_to_hub(out_cfg["hub_repo_id"], token=hf_token)
        tokenizer.push_to_hub(out_cfg["hub_repo_id"], token=hf_token)
        print("Publication terminée. Ce modèle peut maintenant être utilisé comme base pour le SFT.")


if __name__ == "__main__":
    main()
