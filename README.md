# Wolof-LLM (Galsen LLM) 🇸🇳

Projet open source visant à développer un LLM performant pour la langue Wolof.

## 🏗️ Architecture du dépôt

```text
Wolof-LLM/
├── README.md
├── .gitignore
├── data preprocessing  cleaning/
│   ├── config.json
│   ├── Preprocessing_Cleaning.ipynb
│   ├── preprocessing.py
│   ├── README.md
│   └── outputs/
│       ├── processed/
│       │   └── v1/
│       └── stats/
│           └── stats_v1.json
├── SmolLM3-3B-Base/
│   ├── README.md
│   ├── configs/
│   ├── data/
│   │   ├── train/
│   │   └── val/
│   ├── notebooks/
│   │   ├── clm/
│   │   └── sft/
│   ├── runs/
│   ├── scripts/
│   │   ├── clm/
│   │   └── sft/
│   └── tokenizers/
├── Qwen3.5-Base-2B/
│   ├── README.md
│   ├── configs/
│   ├── data/
│   │   ├── train/
│   │   └── val/
│   ├── notebooks/
│   │   ├── clm/
│   │   └── sft/
│   ├── runs/
│   ├── scripts/
│   │   ├── clm/
│   │   └── sft/
│   └── tokenizers/
├── TinyAya-Base-3B/
│   ├── README.md
│   ├── configs/
│   ├── data/
│   │   ├── train/
│   │   └── val/
│   ├── notebooks/
│   │   ├── clm/
│   │   └── sft/
│   ├── runs/
│   ├── scripts/
│   │   ├── clm/
│   │   └── sft/
│   └── tokenizers/
└── LFM2.5-Base-1B/
    ├── README.md
    ├── configs/
    ├── data/
    │   ├── train/
    │   └── val/
    ├── notebooks/
    │   ├── clm/
    │   └── sft/
    ├── runs/
    ├── scripts/
    │   ├── clm/
    │   └── sft/
    └── tokenizers/
```

## 📦 Description par dossier

- `SmolLM3-3B-Base/` : Modèle 1
- `Qwen3.5-Base-2B/` : Modèle 2
- `TinyAya-Base-3B/` : Modèle 3
- `LFM2.5-Base-1B/` : Modèle 4
- `data preprocessing  cleaning/` : dossier de préparation et nettoyage des données du corpus
