"""
Partie 1 : Feature Engineering - Du Pixel aux Caractéristiques
Extraction de descripteurs numériques à partir d'images de feuilles de maïs.

Features extraites :
  X1 - pct_rouille  : pourcentage de pixels de teinte rouille (masque HSV)
  X2 - rugosite     : variance des gradients de Sobel (texture / rugosité)
  X3 - ratio_vert   : proportion de pixels verts (indicateur de chlorophylle saine)
                       -> une feuille saine possède un ratio_vert élevé, tandis qu'une
                          feuille malade perd sa chlorophylle au profit des pustules.
"""

import os
import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fonctions d'extraction de caractéristiques
# ---------------------------------------------------------------------------

def extraire_pct_rouille(img_bgr: np.ndarray) -> float:
    """X1 : Pourcentage de pixels de teinte rouille (marrons/jaunes/oranges)
    dans l'espace HSV.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Teintes rouille : H ∈ [8, 30], S ∈ [50, 255], V ∈ [50, 255]
    lower = np.array([8, 50, 50])
    upper = np.array([30, 255, 255])
    masque = cv2.inRange(hsv, lower, upper)
    nb_pixels_rouille = np.count_nonzero(masque)
    nb_pixels_total = masque.shape[0] * masque.shape[1]
    if nb_pixels_total == 0:
        return 0.0
    return nb_pixels_rouille / nb_pixels_total


def extraire_rugosite(img_bgr: np.ndarray) -> float:
    """X2 : Variance de l'intensité des gradients de Sobel (rugosité/texture).
    Les pustules de rouille créent des variations brusques d'intensité.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.var(magnitude))


def extraire_ratio_vert(img_bgr: np.ndarray) -> float:
    """X3 (feature personnelle) : Proportion de pixels verts (chlorophylle).
    Justification agronomique : une feuille saine est majoritairement verte
    grâce à la chlorophylle. La rouille Polysora détruit les cellules foliaires,
    réduisant la surface verte. Un ratio_vert faible signale une feuille malade.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Vert : H ∈ [35, 85], S ∈ [40, 255], V ∈ [40, 255]
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    masque = cv2.inRange(hsv, lower, upper)
    nb_pixels_verts = np.count_nonzero(masque)
    nb_pixels_total = masque.shape[0] * masque.shape[1]
    if nb_pixels_total == 0:
        return 0.0
    return nb_pixels_verts / nb_pixels_total


def extraire_features_image(chemin_image: str) -> dict | None:
    """Extrait les 3 features d'une image et retourne un dict, ou None si erreur."""
    img = cv2.imread(chemin_image)
    if img is None:
        return None
    return {
        "pct_rouille": extraire_pct_rouille(img),
        "rugosite": extraire_rugosite(img),
        "ratio_vert": extraire_ratio_vert(img),
    }


# ---------------------------------------------------------------------------
# Construction du DataFrame à partir des dossiers dataset/
# ---------------------------------------------------------------------------

def construire_dataframe(dossier_dataset: str = "dataset") -> pd.DataFrame:
    """Parcourt les dossiers saines/ et malades/ et construit le DataFrame."""
    records = []
    for label, sous_dossier in [(0, "saines"), (1, "malades")]:
        chemin = os.path.join(dossier_dataset, sous_dossier)
        if not os.path.isdir(chemin):
            print(f"[WARN] Dossier introuvable : {chemin}")
            continue
        fichiers = sorted(
            f for f in os.listdir(chemin)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for nom_fichier in fichiers:
            chemin_img = os.path.join(chemin, nom_fichier)
            feats = extraire_features_image(chemin_img)
            if feats is None:
                print(f"[WARN] Image illisible : {chemin_img}")
                continue
            feats["ID_Image"] = nom_fichier
            feats["label_malade"] = label
            records.append(feats)

    df = pd.DataFrame(records)
    df = df[["ID_Image", "pct_rouille", "rugosite", "ratio_vert", "label_malade"]]
    return df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    dossier = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    print(f"Extraction des features depuis '{dossier}' ...")
    df = construire_dataframe(dossier)
    csv_path = "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"DataFrame sauvegardé dans {csv_path}  ({len(df)} images)")
    print(df.head(10))
    print("\nStatistiques par classe :")
    print(df.groupby("label_malade")[["pct_rouille", "rugosite", "ratio_vert"]].mean())
