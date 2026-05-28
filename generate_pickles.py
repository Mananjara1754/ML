import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('mock_logements_antananarivo_1000.csv')

df_preprocessed = df.copy()
df_preprocessed['douche_wc'] = df_preprocessed['douche_wc'].map({'exterieur': 0, 'interieur': 1})
df_preprocessed['meuble'] = df_preprocessed['meuble'].map({'non': 0, 'oui': 1})
df_preprocessed['etat_general'] = df_preprocessed['etat_general'].map({'bon': 10, 'moyen': 5, 'mauvais': 0})
df_preprocessed['type_d_acces'] = df_preprocessed['type_d_acces'].map({'sans': 0, 'moto': 5, 'voiture': 10, 'voiture_avec_parking': 15})

df_preprocessed["score_confort"] = (
    (df_preprocessed["douche_wc"] == 1).astype(int)
    + (df_preprocessed["meuble"] == 1).astype(int)
    + (df_preprocessed["type_d_acces"] >= 10).astype(int)
    + (df_preprocessed["etat_general"] >= 10).astype(int)
)

df_model = pd.get_dummies(df_preprocessed, columns=['quartier'], dtype=int)

corr = df_model.corr(numeric_only=True)
threshold = 0.9
to_drop = set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            to_drop.add(corr.columns[i])
df_reduced = df_model.drop(columns=to_drop)

X = df_reduced.drop("loyer_mensuel", axis=1)
y = df_reduced["loyer_mensuel"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

model = LinearRegression()
model.fit(X_scaled, y) # Entraînement du modèle

# Sauvegarde des fichiers
joblib.dump(model, 'modele_loyer.pkl')
joblib.dump(scaler, 'scaler_loyer.pkl')
joblib.dump(X.columns.tolist(), 'colonnes_loyer.pkl')

print("Modèle entraîné et sauvegardé avec succès !")
