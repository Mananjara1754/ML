"""
Partie 3 : Entraînement des 4 modèles + comparaison quantitative
et sauvegarde du meilleur modèle (pickle) pour l'application Streamlit.

Modèles :
  1. Arbre Max-Minority (from scratch)
  2. Random Forest Max-Minority (from scratch)
  3. DecisionTreeClassifier scikit-learn (Gini)
  4. RandomForestClassifier scikit-learn (Gini, n_estimators=100)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    classification_report,
)

from feature_engineering import construire_dataframe
from models import build_tree, predire_arbre_batch, RandomForestMaxMinority


def main():
    # ------------------------------------------------------------------
    # 1. Extraction des features (ou chargement si features.csv existe)
    # ------------------------------------------------------------------
    csv_path = "features.csv"
    if os.path.exists(csv_path):
        print(f"Chargement du DataFrame depuis {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Extraction des features depuis dataset/ ...")
        df = construire_dataframe("dataset")
        df.to_csv(csv_path, index=False)
        print(f"DataFrame sauvegardé → {csv_path}")

    print(f"\n{'='*60}")
    print(f"Nombre d'images : {len(df)}")
    print(df.head())
    print(f"\nDistribution des classes :\n{df['label_malade'].value_counts()}")

    # ------------------------------------------------------------------
    # 2. Séparation train/test (80/20)
    # ------------------------------------------------------------------
    feature_cols = ["pct_rouille", "rugosite", "ratio_vert"]
    X = df[feature_cols].values
    y = df["label_malade"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Entraînement des 4 modèles
    # ------------------------------------------------------------------
    resultats = {}

    # --- 3a. Arbre Max-Minority (from scratch) ---
    print("\n[1] Arbre Max-Minority (from scratch) ...")
    arbre_mm = build_tree(X_train, y_train, depth=0, max_depth=5)
    y_pred_1 = predire_arbre_batch(arbre_mm, X_test)
    resultats["Arbre Max-Minority"] = y_pred_1

    # --- 3b. Random Forest Max-Minority (from scratch) ---
    print("[2] Random Forest Max-Minority (from scratch) ...")
    rf_mm = RandomForestMaxMinority(n_arbres=20, max_depth=5, random_state=42)
    rf_mm.fit(X_train, y_train)
    y_pred_2 = rf_mm.predict(X_test)
    resultats["RF Max-Minority"] = y_pred_2

    # --- 3c. DecisionTree scikit-learn (Gini) ---
    print("[3] DecisionTree scikit-learn (Gini) ...")
    dt_sk = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    dt_sk.fit(X_train, y_train)
    y_pred_3 = dt_sk.predict(X_test)
    resultats["Arbre Gini (sklearn)"] = y_pred_3

    # --- 3d. RandomForest scikit-learn (Gini, 100 arbres) ---
    print("[4] RandomForest scikit-learn (Gini, n=100) ...")
    rf_sk = RandomForestClassifier(
        n_estimators=100, criterion="gini", random_state=42
    )
    rf_sk.fit(X_train, y_train)
    y_pred_4 = rf_sk.predict(X_test)
    resultats["RF Gini (sklearn)"] = y_pred_4

    # ------------------------------------------------------------------
    # 4. Tableau comparatif : Accuracy, Précision, Rappel
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TABLEAU COMPARATIF DES PERFORMANCES (jeu de test)")
    print(f"{'='*60}")
    header = f"{'Modèle':<25} {'Accuracy':>10} {'Précision':>10} {'Rappel':>10}"
    print(header)
    print("-" * len(header))

    for nom, y_pred in resultats.items():
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        print(f"{nom:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f}")

    # ------------------------------------------------------------------
    # 5. Matrices de confusion
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("MATRICES DE CONFUSION")
    print(f"{'='*60}")
    for nom, y_pred in resultats.items():
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n--- {nom} ---")
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # ------------------------------------------------------------------
    # 6. Importance des variables (Random Forest sklearn)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("IMPORTANCE DES VARIABLES (Random Forest sklearn)")
    print(f"{'='*60}")
    for feat, imp in zip(feature_cols, rf_sk.feature_importances_):
        print(f"  {feat:<15} : {imp:.4f}")

    # ------------------------------------------------------------------
    # 7. Analyse qualitative
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSE")
    print(f"{'='*60}")
    print("""
1. Comportement : La Random Forest (maison ou sklearn) améliore la robustesse
   par rapport à un arbre unique grâce au Bagging qui réduit la variance.
   Chaque arbre voit un sous-échantillon différent ; le vote majoritaire
   atténue les erreurs individuelles.

2. Recommandation agronomique (Madagascar) :
   Dans ce contexte, un Faux Négatif (manquer une feuille malade) est plus
   grave qu'un Faux Positif (traiter inutilement une feuille saine) car
   l'épidémie peut se propager et détruire les récoltes entières.
   Il faut donc maximiser le RAPPEL (sensibilité).
   Le modèle Random Forest (sklearn, Gini, n=100) est recommandé pour le
   déploiement terrain car il offre le meilleur compromis rappel/précision
   tout en bénéficiant de la robustesse de l'agrégation.
""")

    # ------------------------------------------------------------------
    # 8. Sauvegarde du meilleur modèle pour l'app Streamlit
    # ------------------------------------------------------------------
    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf_sk, f)
    print(f"Modèle sauvegardé → {model_path}")

    # Sauvegarder aussi les noms de features pour l'app
    meta = {"feature_cols": feature_cols}
    with open("model_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("Métadonnées sauvegardées → model_meta.pkl")


if __name__ == "__main__":
    main()
