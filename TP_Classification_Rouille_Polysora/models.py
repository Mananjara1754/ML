"""
Partie 3 : Arbres et Forêts "From Scratch" avec la métrique Max-Minority.
"""

import numpy as np
from max_minority import purete, trouver_meilleur_split


# ---------------------------------------------------------------------------
# Arbre de Décision Max-Minority (from scratch)
# ---------------------------------------------------------------------------

class NoeudDecision:
    """Noeud interne ou feuille d'un arbre de décision."""

    def __init__(self, prediction: int | None = None,
                 feature_idx: int | None = None,
                 seuil: float | None = None,
                 gauche: "NoeudDecision | None" = None,
                 droite: "NoeudDecision | None" = None):
        self.prediction = prediction
        self.feature_idx = feature_idx
        self.seuil = seuil
        self.gauche = gauche
        self.droite = droite

    def est_feuille(self) -> bool:
        return self.prediction is not None


def _vote_majoritaire(y: np.ndarray) -> int:
    """Retourne la classe majoritaire."""
    compte = np.bincount(y.astype(int), minlength=2)
    return int(np.argmax(compte))


def build_tree(X: np.ndarray, y: np.ndarray,
               depth: int = 0, max_depth: int = 5) -> NoeudDecision:
    """Construit récursivement un arbre de décision avec la métrique Max-Minority.

    Conditions d'arrêt :
      - Noeud 100% pur (P(t) = 1.0)
      - Profondeur maximale atteinte
      - Moins de 2 exemples
    """
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)

    # Conditions d'arrêt
    if purete(y) == 1.0 or depth >= max_depth or len(y) < 2:
        return NoeudDecision(prediction=_vote_majoritaire(y))

    meilleur_feat = None
    meilleur_seuil = None
    meilleure_purete = -1.0
    n_features = X.shape[1]

    for j in range(n_features):
        seuil, p = trouver_meilleur_split(X[:, j], y)
        if seuil is not None and p > meilleure_purete:
            meilleure_purete = p
            meilleur_seuil = seuil
            meilleur_feat = j

    # Si aucun split améliorant n'a été trouvé
    if meilleur_feat is None:
        return NoeudDecision(prediction=_vote_majoritaire(y))

    masque_g = X[:, meilleur_feat] <= meilleur_seuil
    X_g, y_g = X[masque_g], y[masque_g]
    X_d, y_d = X[~masque_g], y[~masque_g]

    if len(y_g) == 0 or len(y_d) == 0:
        return NoeudDecision(prediction=_vote_majoritaire(y))

    gauche = build_tree(X_g, y_g, depth + 1, max_depth)
    droite = build_tree(X_d, y_d, depth + 1, max_depth)

    return NoeudDecision(
        feature_idx=meilleur_feat,
        seuil=meilleur_seuil,
        gauche=gauche,
        droite=droite,
    )


def predire_arbre(noeud: NoeudDecision, x: np.ndarray) -> int:
    """Prédit la classe d'un seul échantillon x."""
    if noeud.est_feuille():
        return noeud.prediction
    if x[noeud.feature_idx] <= noeud.seuil:
        return predire_arbre(noeud.gauche, x)
    return predire_arbre(noeud.droite, x)


def predire_arbre_batch(noeud: NoeudDecision, X: np.ndarray) -> np.ndarray:
    """Prédit les classes pour un batch d'échantillons."""
    return np.array([predire_arbre(noeud, x) for x in X])


# ---------------------------------------------------------------------------
# Random Forest Max-Minority (from scratch)
# ---------------------------------------------------------------------------

class RandomForestMaxMinority:
    """Forêt aléatoire utilisant la métrique Max-Minority.
    Implémente le Bagging + vote majoritaire.
    """

    def __init__(self, n_arbres: int = 20, max_depth: int = 5,
                 random_state: int = 42):
        self.n_arbres = n_arbres
        self.max_depth = max_depth
        self.random_state = random_state
        self.arbres: list[NoeudDecision] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entraîne la forêt par Bagging."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        N = len(y)
        self.arbres = []
        for _ in range(self.n_arbres):
            # Sous-échantillonnage avec remplacement (bagging)
            indices = rng.choice(N, size=N, replace=True)
            X_bag, y_bag = X[indices], y[indices]
            arbre = build_tree(X_bag, y_bag, depth=0, max_depth=self.max_depth)
            self.arbres.append(arbre)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction par vote majoritaire."""
        X = np.asarray(X, dtype=float)
        all_preds = np.array([predire_arbre_batch(arbre, X) for arbre in self.arbres])
        # all_preds : shape (n_arbres, n_samples)
        votes = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=2).argmax(), 0, all_preds
        )
        return votes
