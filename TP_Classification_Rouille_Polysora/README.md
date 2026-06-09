# TP : Diagnostic de la Rouille Polysora sur les Feuilles de Maïs à Madagascar

Projet de classification d'images de feuilles de maïs (saines vs malades) pour détecter la Rouille Polysora (*Puccinia polysora*).

## Structure du projet

```
├── dataset/
│   ├── saines/          # Photos de feuilles saines
│   └── malades/         # Photos de feuilles avec rouille
├── feature_engineering.py   # Partie 1 : extraction de features (HSV, Sobel, ratio vert)
├── max_minority.py          # Partie 2 : métrique Max-Minority + trouver_meilleur_split
├── models.py                # Partie 3a : Arbre + Random Forest from scratch
├── train.py                 # Partie 3b : entraînement, comparaison, sauvegarde modèle
├── app.py                   # Partie 4 : application Streamlit
├── features.csv             # DataFrame des features (généré)
├── model.pkl                # Modèle entraîné (généré)
├── model_meta.pkl           # Métadonnées du modèle (généré)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Extraction des features et entraînement

```bash
python train.py
```

Cela :
- Extrait les features (pct_rouille, rugosite, ratio_vert) de toutes les images → `features.csv`
- Entraîne 4 modèles (Arbre/RF Max-Minority from scratch + Arbre/RF Gini sklearn)
- Affiche le tableau comparatif (Accuracy, Précision, Rappel) et les matrices de confusion
- Sauvegarde le meilleur modèle → `model.pkl`

### 2. Lancement de l'application web

```bash
streamlit run app.py
```

L'application permet de :
- Téléverser une image de feuille de maïs
- Obtenir un diagnostic en temps réel (Saine / Malade)
- Consulter l'historique des analyses dans une galerie visuelle

## Features extraites

| Feature | Description | Justification |
|---------|-------------|---------------|
| `pct_rouille` | % de pixels de teinte rouille (masque HSV) | Les pustules de rouille sont orangées/brunâtres |
| `rugosite` | Variance des gradients de Sobel | Les pustules créent des irrégularités de texture |
| `ratio_vert` | % de pixels verts (chlorophylle) | Une feuille malade perd sa chlorophylle → moins de vert |

## Dataset

200 images par classe (saines/malades) tirées du dataset PlantVillage (Corn Common Rust + Corn Healthy), utilisé comme approximation de la Rouille Polysora.
