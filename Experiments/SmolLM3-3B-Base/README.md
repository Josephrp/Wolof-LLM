# SmolLM3-3B-Base pour le wolof

Adaptation de SmolLM3 au wolof : nettoyage du corpus, splits reproductibles,
extension du tokenizer, continual pretraining et évaluation BPB.

## Pipeline

```bash
# 0. Nettoyage reproductible du corpus brut source.
python3 scripts/00_clean_corpus.py \
  --input data/data_v1.parquet \
  --output "$DATA/wolof_corpus_clean.parquet"

# 1. Splits figés + contrôle de fuite train/eval.
python3 scripts/01_prepare_data.py \
  --corpus "$DATA/wolof_corpus_clean.parquet" \
  --data-root "$DATA"

# 2. Tokenizer étendu 4 000 + modèle initialisé.
python3 scripts/02_build_tokenizer.py \
  --config configs/tokenizer-ext4000.yaml

# 3. Validation rapide de la chaîne complète.
python3 scripts/03_train.py \
  --config configs/wolof-ext4000-lora.yaml \
  --smoke

# 4. Les quatre runs.

python3 scripts/03_train.py --config configs/wolof-ext4000-full.yaml
python3 scripts/03_train.py --config configs/wolof-ext4000-lora.yaml

python3 scripts/03_train.py --config configs/wolof-default-lora.yaml
python3 scripts/03_train.py --config configs/wolof-default-full.yaml


# 5. Comparaison finale, BPB recalculé pour tous sur le même split.
python3 scripts/04_evaluate.py \
  --data-root "$DATA" \
  --output-root "$OUT"
```

# Résultats v1 cadrage données égales

Continual pretraining monolingue wolof sur SmolLM3-3B-Base.
Chaque configuration fait exactement **une epoch** : les deux tokenizers voient
le même texte, seule la représentation change.

## Protocole

| | |
| --- | --- |
| Corpus | 963 646 textes wolof, ~72 M tokens, 47 sources |
| Évaluation | 5 000 textes held-out, quasi-doublons retirés |
| Batch effectif | 32 séquences × 2 048 tokens = 65 536 tokens/pas |
| Matériel | H200, attention sdpa, bf16 |
| Graine | 42 (une seule) |

Pas calculés pour obtenir 1 epoch : `(72 M × fertilité) ÷ 65 536`
→ 2 234 pas (tokenizer défaut), 1 423 pas (tokenizer étendu).

## Extension du tokenizer

| | Défaut | Étendu (+4 000) |
| --- | --- | --- |
| Vocabulaire | 128 256 | 132 256 |
| Fertilité (tokens/mot) | 2,033 | **1,295** |

−36 % de tokens par mot. Aucune régression sur 20 000 mots testés,
français / anglais / code strictement inchangés.

## Résultats

| Run | Tokenizer | Méthode | Pas | LR | BPB ↓ | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| **wolof-ext4000-lora** | étendu | LoRA | 1 423 | 2e-4 | **0,7413** | 15,83 |
| wolof-default-full | défaut | full FT | 2 234 | 2e-5 | 0,7521 | 5,85 |
| wolof-default-lora | défaut | LoRA | 2 234 | 2e-4 | 0,7892 | 6,38 |
| wolof-ext4000-full | étendu | full FT | 1 423 | 2e-5 | 0,8926 | 27,82 |
| base non entraînée | défaut | — | — | — | 2,1624 | 160,61 |

**Le BPB est la seule métrique comparable entre tokenizers différents.**
La perplexité se calcule par token : elle classerait le tokenizer par défaut
devant l'étendu (6,38 contre 15,83) alors que le BPB donne l'inverse.
Ne comparer les PPL qu'à tokenizer identique.

## Lecture

**En LoRA, l'extension améliore la modélisation causale de 6,1 %**
(0,7413 contre 0,7892). Les deux modèles ont vu le même texte : c'est un gain
de représentation, pas seulement de compression.

**En full fine-tuning, l'extension dégrade** (0,8926 contre 0,7521).
Hypothèse : un learning rate unique à 2e-5 ne peut pas servir à la fois des
poids pré-entraînés, qui exigent de petits pas, et 4 000 embeddings neufs qui
en exigent de grands. C'est ce que la v2 teste, via des learning rates
différenciés et une phase d'échauffement des embeddings.

**Conclusion : l'extension du tokenizer améliore la modélisation causale,
à condition d'employer une méthode qui protège les poids pré-entraînés.**

## Choix de la taille du vocabulaire

| Tokens ajoutés | Fertilité | BPB avant entraînement | BPB après |
| --- | --- | --- | --- |
| 4 000 | 1,295 | 2,1797 | **0,8767** |
| 8 000 | 1,265 | 2,2478 | 0,8815 |

Avant entraînement les métriques divergent : le 8 000 compresse mieux mais
modélise moins bien. Après entraînement l'écart tombe de 0,068 à 0,005 — le
retard du 8 000 venait d'une initialisation imparfaite de ses tokens rares.
Les deux étant équivalents, le **4 000** est retenu : même performance, deux
fois moins d'embeddings à entraîner.

*(Ablation menée en LoRA, à 250 pas, avant la série principale.)*

## Limites

- **Une seule epoch**  conclusions valables à ce budget, pas à convergence.
  Le full fine-tuning, qui déplace 3 milliards de paramètres à un LR dix fois
  plus faible, est structurellement désavantagé sur un budget aussi court.

-

## Modèles

Publiés sur `galsenai`.

<https://huggingface.co/galsenai/SmolLM3-3B-Base_wolof/tree/main/v1>

## V2


**`v2/ext4000-lora`** meilleur BPB (0,5899). C'est le point de départ du SFT.

**v2: best of each.** Chaque configuration reçoit ses meilleurs réglages
(learning rate distinct pour les embeddings, phase d'échauffement en full
fine-tuning) et tourne jusqu'à ce que sa loss de validation cesse de progresser.
Mesure ce que chaque recette peut atteindre.

## Résultats

| Modèle | Tokenizer | Méthode | BPB ↓ | Step | Durée |
|---|---|---|---|---|---|
| **`v2/ext4000-lora`** | étendu | LoRA | **0,5899** | 3 800 | 7,2 h |
| `v2/default-lora` | défaut | LoRA | 0,6005 | 5 700 | 10,7 h |
| `v2/ext4000-full` | étendu | full FT | 0,6955 | 3 300 | 3,8 h |
| `v2/default-full` | défaut | full FT | 0,6993 | 4 600 | 5,2 h |
| *(base non entraînée)* | défaut | — | 2,1624 | — | — |

## Extension du tokenizer

4 000 tokens de mots wolof entiers ajoutés (128 256 → 132 256).
Fertilité : **2,033 → 1,295** tokens/mot, soit −36 %. 

## Ce que montrent les résultats pour le moment

**Le tokenizer étendu gagne dans les deux séries**, en LoRA comme en full
fine-tuning. Mais l'écart se resserre à convergence : 6,1 % en v1 (budget
court), 1,8 % en v2 (jusqu'au plancher). L'extension aide donc surtout quand le
budget de calcul est limité.

**Le gain d'efficacité, lui, reste entier** : les runs à tokenizer étendu
convergent en moins de pas *et* moins d'heures, pour un meilleur BPB.

**Un learning rate unique ne convient pas au full fine-tuning avec vocabulaire
étendu.** En v1, cette configuration était la pire (0,8926). En v2, avec un
learning rate dix fois plus élevé sur les embeddings et une phase
d'échauffement embeddings-only, elle passe à 0,6955. La phase d'échauffement à
elle seule fait tomber le BPB de 1,9914 à 1,5976, backbone gelé.
