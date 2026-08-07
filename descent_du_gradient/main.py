import numpy as np

def descente_gradient(X, y, theta, alpha, iterations):
    """
    Résout par descente de gradient ce que l'équation normale (X^T X)^-1 X^T y
    résout directement : minimiser le coût J(theta) = (1/2m) * sum((X.theta - y)^2).

    X : Matrice des données, m lignes (exemples) x n colonnes (variables + biais)
    y : Vecteur des valeurs réelles, taille m
    theta : Vecteur des paramètres (poids), taille n — point de départ
    alpha : Taux d'apprentissage (pas de mise à jour à chaque itération)
    iterations : Nombre de pas de descente à effectuer
    """
    m = len(y)  # nombre d'exemples, utilisé pour moyenner l'erreur sur tout le jeu de données
    historique_cout = []

    for i in range(iterations):
        # Prédiction du modèle linéaire : y_hat = X . theta
        y_pred = np.dot(X, theta)

        # Erreur signée par exemple (positive si on surestime, négative si on sous-estime)
        erreurs = y_pred - y

        # Gradient de J par rapport à theta : X^T . erreurs / m
        # (dérivée de (1/2m)*sum(erreurs^2) → le facteur 2 s'annule avec le carré, il reste 1/m)
        gradient = (1 / m) * np.dot(X.T, erreurs)

        # Pas de descente : on avance dans le sens opposé au gradient, proportionnellement à alpha
        theta = theta - alpha * gradient

        # Coût recalculé avec le theta mis à jour, pour que la courbe de coût
        # reflète bien l'état du modèle à la fin de l'itération i
        y_pred_new = np.dot(X, theta)
        cout = (1 / (2 * m)) * np.sum((y_pred_new - y) ** 2)
        historique_cout.append(cout)

    return theta, historique_cout


# --- Données d'exemple (3 individus, 2 variables) ---
X = np.array([
    [8, 9],
    [3, 9],
    [4, 10]
])
y = np.array([1, 2, 3])  # valeurs cibles associées à chaque ligne de X

# Ajout d'une colonne de 1 devant X : theta[0] devient alors le terme de biais (intercept)
X_biais = np.column_stack((np.ones(len(X)), X))

# theta doit avoir une composante par colonne de X_biais (biais + variables), d'où X_biais.shape[1]
theta_initial = np.zeros(X_biais.shape[1])

# Paramètres d'apprentissage
alpha = 0.0001
iterations = 1000

# Lancement
theta_final, couts = descente_gradient(X_biais, y, theta_initial, alpha, iterations)
print(f"Paramètres optimisés : {theta_final}")
print(f"Coût final           : {couts[-1]:.6f}")