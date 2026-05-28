import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title="Prédiction de Loyer - Antananarivo", layout="wide")

# --- 1. CHARGEMENT DU MODELE SAUVEGARDE ---
@st.cache_resource
def load_model():
    # Cache invalidation comment
    model = joblib.load('modele_loyer.pkl')
    scaler = joblib.load('scaler_loyer.pkl')
    feature_cols = joblib.load('colonnes_loyer.pkl')

    quartiers_coords = {
        'Alasora': [-18.9619, 47.5586],
        'Ambatoroka': [-18.9246, 47.5432],
        'Analakely': [-18.9055, 47.5256],
        'Andavamamba': [-18.9149, 47.5028],
        'Ankatso': [-18.9141, 47.5521],
        'Ankorondrano': [-18.8856, 47.5218],
        'Isoraka': [-18.9092, 47.5216],
        'Itaosy': [-18.9248, 47.4578],
        'Ivandry': [-18.8744, 47.5256],
        'Mahamasina': [-18.9157, 47.5233]
    }
    return model, scaler, feature_cols, quartiers_coords

model, scaler, feature_cols, quartiers_coords = load_model()

# --- 2. FONCTION POUR TROUVER LE QUARTIER PAR GPS ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

def trouver_quartier_le_plus_proche(lat, lon):
    min_dist = float('inf')
    quartier_proche = None
    for q, coords in quartiers_coords.items():
        dist = haversine(lat, lon, coords[0], coords[1])
        if dist < min_dist:
            min_dist = dist
            quartier_proche = q
    return quartier_proche

# --- 3. INTERFACE UTILISATEUR ---
st.title("🏡 Prédiction de Loyer à Antananarivo")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Choisissez l'emplacement")
    m = folium.Map(location=[-18.91368, 47.53613], zoom_start=12)
    for q, coords in quartiers_coords.items():
        folium.CircleMarker(location=coords, radius=5, popup=q, color="blue", fill=True).add_to(m)

    map_data = st_folium(m, height=400, width=500)
    quartier_selectionne = "Analakely"
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        quartier_selectionne = trouver_quartier_le_plus_proche(lat, lon)
        st.success(f"📍 Quartier le plus proche : **{quartier_selectionne}**")

with col2:
    st.subheader("📋 Caractéristiques")
    quartier_input = st.selectbox("Quartier", list(quartiers_coords.keys()), index=list(quartiers_coords.keys()).index(quartier_selectionne))
    superficie = st.number_input("Superficie (m²)", min_value=10, max_value=500, value=50)
    chambres = st.number_input("Nombre de chambres", min_value=1, max_value=10, value=2)
    douche_wc = st.selectbox("Douche/WC", ["interieur", "exterieur"])
    acces = st.selectbox("Type d'accès", ["sans", "moto", "voiture", "voiture_avec_parking"])
    meuble = st.selectbox("Meublé", ["non", "oui"])
    etat = st.selectbox("État général", ["bon", "moyen", "mauvais"])

    if st.button("🚀 Prédire le Loyer", use_container_width=True):
        input_data = {
            'superficie': superficie, 'nombre_chambres': chambres,
            'douche_wc': 1 if douche_wc == "interieur" else 0,
            'type_d_acces': {'sans': 0, 'moto': 5, 'voiture': 10, 'voiture_avec_parking': 15}[acces],
            'meuble': 1 if meuble == "oui" else 0,
            'etat_general': {'bon': 10, 'moyen': 5, 'mauvais': 0}[etat]
        }
        input_data['score_confort'] = ((input_data['douche_wc'] == 1) + (input_data['meuble'] == 1) + (input_data['type_d_acces'] >= 10) + (input_data['etat_general'] >= 10))

        for q in quartiers_coords.keys():
            col_name = f"quartier_{q}"
            if col_name in feature_cols:
                input_data[col_name] = 1 if quartier_input == q else 0

        input_df = pd.DataFrame([input_data])
        for col in feature_cols:
            if col not in input_df.columns: input_df[col] = 0
        input_df = input_df[feature_cols]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Le loyer mensuel estimé est de : **{prediction:,.0f} Ar**")

# --- 4. VISUALISATION DES POIDS ---
st.markdown("---")
st.subheader("📊 Poids des variables du modèle")
coefs = pd.DataFrame({'Variable': feature_cols, 'Poids (Coefficient)': model.coef_})
coefs['Poids Absolu'] = coefs['Poids (Coefficient)'].abs()
coefs = coefs.sort_values(by='Poids Absolu', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Poids (Coefficient)', y='Variable', data=coefs, ax=ax, hue='Variable', legend=False, palette="coolwarm")
ax.set_title("Importance de chaque caractéristique")
st.pyplot(fig)
