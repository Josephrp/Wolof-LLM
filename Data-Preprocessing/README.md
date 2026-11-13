Documentation - Prétraitement et Nettoyage du Dataset Wolof

## 1. Présentation du Dataset

Pour le fichier nous avons le dataset de base depuis hugging face : galsenai/wolof_centalized_corpus 

Le dataset initial contient 303 317 lignes et deux colonnes 
text : les phrases en Wolof

sources : l'origine des données 


## 2. Visualisation initiale

Première inspection des données pour comprendre leur structure et leur contenu avant tout traitement.

## 3. Vérification de la qualité des données

 3.1 Valeurs manquantes et chaînes vides

1 valeur manquante (NaN) détectée
1 chaîne vide (ou whitespace uniquement) détectée

Ces deux lignes ont été supprimées

 3.2 Doublons

Aucun doublon détecté dans le dataset


## 4. Analyse de la distribution des longueurs

Calcul de la longueur de chaque texte (en caractères) pour comprendre la variabilité du dataset :
| Statistique | Valeur | Description |
|---|---|---|
| count | 303 316 | Nombre total de lignes analysées |
| mean | 70.56 | Longueur moyenne d'un texte |
| std | 111.99 | Écart-type élevé → forte variabilité des longueurs |
| min | 0.00 | Texte le plus court |
| 25% | 23.00 | 25% des textes font moins de 23 caractères |
| 50% | 44.00 | Médiane |
| 75% | 81.00 | 75% des textes font moins de 81 caractères |
| max | 33 689.00 | Texte le plus long |



## 5. Suppression de la colonne sources

La colonne "sources" n'apporte pas d'information utile pour le fine-tuning. Elle a été supprimée. Seule la colonne "text" est conservée.

## 6. Nettoyage des textes courts

6.1 Inspection des textes de moins de 23 caractères

Le premier quartile étant à 23 caractères, ces lignes ont été lues manuellement.

Résultat : la majorité de ces phrases courtes ont du sens en Wolof et sont pertinentes pour l'entraînement - elles ont donc été conservées.

6.2 Suppression des mots uniques 

Nous avons visualisé les textes faisant moins de 5 caractères (~1 200 lignes, soit 0,4% du dataset) et il s'agissait en général de mots uniques (~4 990 occurrences uniques, soit 1,6% du dataset). Suite à cette observation, nous avons décidé de supprimer toutes les lignes contenant des mots uniques.


## 7. Suppression des lignes numériques

Les lignes contenant uniquement des chiffres (sans texte Wolof) ont été supprimées, car elles ne contribuent pas à l'apprentissage du langage.

## 8. Nettoyage typographique et structurel

8.1 Suppression des guillemets

1 796 phrases contenaient des guillemets
Les guillemets ont été retirés pour homogénéiser le format du texte

8.2 Suppression des balises et caractères spéciaux

Les éléments suivants ont été nettoyés :

Balises HTML (`<p>`, `<br>`, ...)

Sauts de ligne (`\n`)

Tabulations (`\t`)


## 9. Normalisation

Tous les textes ont été convertis en minuscules pour uniformiser le corpus.


## 10. Sortie 

La sortie est enregistrée sur le dossier : outputs/processed et les statistiques sur :  outputs/stats
