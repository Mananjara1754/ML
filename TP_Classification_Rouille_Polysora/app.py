"""
Partie 4 : Application Web Streamlit
Diagnostic de la Rouille Polysora sur les feuilles de maïs.

Fonctionnalités :
  1. Upload et prédiction en temps réel
  2. Galerie d'historique des détections
"""

import os
import pickle
import shutil
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from feature_engineering import (
    extraire_pct_rouille,
    extraire_rugosite,
    extraire_ratio_vert,
)

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diagnostic Rouille Polysora - Maïs Madagascar",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

@st.cache_resource
def charger_modele():
    with open("model.pkl", "rb") as f:
        modele = pickle.load(f)
    with open("model_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return modele, meta["feature_cols"]


modele, feature_cols = charger_modele()

# ---------------------------------------------------------------------------
# Dossier d'historique
# ---------------------------------------------------------------------------
UPLOAD_DIR = "uploads"
HISTORY_FILE = os.path.join(UPLOAD_DIR, "historique.txt")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def charger_historique() -> list[dict]:
    """Charge l'historique depuis le fichier texte."""
    if not os.path.exists(HISTORY_FILE):
        return []
    entries = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 4:
                entries.append({
                    "filename": parts[0],
                    "diagnostic": parts[1],
                    "pct_rouille": float(parts[2]),
                    "rugosite": float(parts[3]),
                })
    return entries


def sauvegarder_historique(entry: dict):
    """Ajoute une entrée à l'historique."""
    with open(HISTORY_FILE, "a") as f:
        f.write(
            f"{entry['filename']}|{entry['diagnostic']}|"
            f"{entry['pct_rouille']:.6f}|{entry['rugosite']:.4f}\n"
        )


# ---------------------------------------------------------------------------
# Interface principale
# ---------------------------------------------------------------------------

st.title("Diagnostic de la Rouille Polysora")
st.markdown(
    "**Système d'aide au diagnostic pour les techniciens agricoles à Madagascar**  \n"
    "Téléversez une photo de feuille de maïs pour obtenir un diagnostic instantané."
)

st.divider()

# --- Module 1 : Upload et prédiction ---
st.header("Téléversement et Prédiction")

uploaded_file = st.file_uploader(
    "Choisissez une image de feuille de maïs",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    # Lire l'image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Impossible de lire l'image. Veuillez réessayer.")
    else:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.subheader("Image téléversée")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)

        # Extraction des features
        pct_rouille = extraire_pct_rouille(img_bgr)
        rugosite = extraire_rugosite(img_bgr)
        ratio_vert = extraire_ratio_vert(img_bgr)

        features = np.array([[pct_rouille, rugosite, ratio_vert]])
        prediction = modele.predict(features)[0]

        with col_result:
            st.subheader("Résultat du diagnostic")

            st.markdown("**Caractéristiques extraites :**")
            st.write(f"- `pct_rouille` : {pct_rouille:.4f}")
            st.write(f"- `rugosite` : {rugosite:.2f}")
            st.write(f"- `ratio_vert` : {ratio_vert:.4f}")

            st.markdown("---")

            if prediction == 1:
                st.error(
                    "**ATTENTION : Feuille Malade (Rouille Détectée)**\n\n"
                    "Des pustules orangées/brunâtres caractéristiques de la "
                    "Rouille Polysora ont été détectées. "
                    "Un traitement fongicide est recommandé."
                )
                diagnostic = "MALADE"
            else:
                st.success(
                    "**Feuille Saine**\n\n"
                    "Aucun signe de Rouille Polysora détecté. "
                    "La feuille présente un aspect normal."
                )
                diagnostic = "SAINE"

        # Sauvegarder dans l'historique
        save_name = f"{len(charger_historique()):04d}_{uploaded_file.name}"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        cv2.imwrite(save_path, img_bgr)
        sauvegarder_historique({
            "filename": save_name,
            "diagnostic": diagnostic,
            "pct_rouille": pct_rouille,
            "rugosite": rugosite,
        })

st.divider()

# --- Module 2 : Galerie d'historique ---
st.header("Galerie d'Historique des Détections")

historique = charger_historique()

if not historique:
    st.info("Aucune analyse enregistrée. Téléversez une image ci-dessus.")
else:
    n_cols = 4
    cols = st.columns(n_cols)
    for idx, entry in enumerate(reversed(historique)):
        col = cols[idx % n_cols]
        img_path = os.path.join(UPLOAD_DIR, entry["filename"])
        if os.path.exists(img_path):
            with col:
                st.image(img_path, use_container_width=True)
                if entry["diagnostic"] == "MALADE":
                    st.markdown(
                        f"**{entry['diagnostic']}**  \n"
                        f"Rouille : {entry['pct_rouille']:.2%}"
                    )
                else:
                    st.markdown(
                        f"**{entry['diagnostic']}**  \n"
                        f"Rouille : {entry['pct_rouille']:.2%}"
                    )
