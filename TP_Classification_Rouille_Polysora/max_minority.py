"""
Partie 2 : L'Indice "Max-Minority"
Métrique de pureté personnalisée et recherche du meilleur split par balayage.

Pureté P(t) = max(n_0/N, n_1/N)  (proportion de la classe majoritaire)
Psplit = (|G|/N)*P(G) + (|D|/N)*P(D)   — on maximise Psplit.
"""

import numpy as np


def purete(y: np.ndarray) -> float:
    """Calcule la pureté Max-Minority d'un noeud.
    P(t) = max(n_c / N) pour c dans {0, 1}.
    """
    n = len(y)
    if n == 0:
        return 1.0
    compte = np.bincount(y.astype(int), minlength=2)
    return float(compte.max()) / n


def trouver_meilleur_split(X_column: np.ndarray, y: np.ndarray):
    """Teste tous les seuils possibles pour une variable continue et retourne
    le seuil s qui maximise Psplit, ainsi que la valeur de cette pureté.

    Paramètres
    ----------
    X_column : array-like des valeurs de la variable à tester
    y        : array-like des labels correspondants (0 ou 1)

    Retourne
    --------
    (meilleur_seuil, meilleure_purete)
    """
    X_column = np.asarray(X_column, dtype=float)
    y = np.asarray(y, dtype=int)

    # Étape 1 : Trier les données par X_column
    indices_tri = np.argsort(X_column)
    X_tri = X_column[indices_tri]
    y_tri = y[indices_tri]
    N = len(y_tri)

    # Étape 2 : Initialiser les variables pour le meilleur seuil
    meilleur_seuil = None
    meilleure_purete = -1.0

    # Étape 3 : Parcourir les seuils candidats (milieux entre valeurs
    # consécutives uniques)
    valeurs_uniques = np.unique(X_tri)
    if len(valeurs_uniques) < 2:
        return None, purete(y)

    for i in range(len(valeurs_uniques) - 1):
        seuil = (valeurs_uniques[i] + valeurs_uniques[i + 1]) / 2.0

        masque_gauche = X_tri <= seuil
        y_gauche = y_tri[masque_gauche]
        y_droite = y_tri[~masque_gauche]
        n_g = len(y_gauche)
        n_d = len(y_droite)

        if n_g == 0 or n_d == 0:
            continue

        p_split = (n_g / N) * purete(y_gauche) + (n_d / N) * purete(y_droite)

        if p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil = seuil

    # Étape 4 : Retourner le seuil optimal et sa pureté associée
    return meilleur_seuil, meilleure_purete
