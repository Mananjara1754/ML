import numpy as np

def descente_gradient(X, y, theta, alpha, iterations):
    """
    X : Matrice des données (m lignes, n colonnes)
    y : Vecteur des valeurs réelles
    theta : Vecteur des paramètres (poids)
    alpha : Taux d'apprentissage
    iterations : Nombre de répétitions
    """
    m = len(y)  # Nombre d'exemples
    historique_cout = []

    for i in range(iterations):
        # 1. Calcul des prédictions (Modèle : y_hat = X * theta)
        y_pred = np.dot(X, theta)

        # 2. Calcul de l'erreur
        erreurs = y_pred - y

        # 3. Calcul du gradient (avec 1/2m du cours → le 2 s'annule → 1/m)
        gradient = (1 / m) * np.dot(X.T, erreurs)

        # 4. Mise à jour des paramètres
        theta = theta - alpha * gradient

        # 5. Calcul du coût APRÈS mise à jour (pour un suivi correct)
        y_pred_new = np.dot(X, theta)
        cout = (1 / (2 * m)) * np.sum((y_pred_new - y) ** 2)
        historique_cout.append(cout)

    return theta, historique_cout


# --- Données ---
X = np.array([
    [8, 9],
    [3, 9],
    [4, 10]
])
y = np.array([1, 2, 3])  # ✅ Bug 1 corrigé : valeurs cibles ajoutées

# Ajout de la colonne pour le biais (b)
X_biais = np.column_stack((np.ones(len(X)), X))

# Initialisation des paramètres à 0
theta_initial = np.zeros(X_biais.shape[1])  # ✅ Bug 2 corrigé : [1] au lieu de [11]

# Paramètres d'apprentissage
alpha = 0.0001
iterations = 1000

# Lancement
theta_final, couts = descente_gradient(X_biais, y, theta_initial, alpha, iterations)
print(f"Paramètres optimisés : {theta_final}")
print(f"Coût final           : {couts[-1]:.6f}")