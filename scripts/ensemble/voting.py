"""Soft Voting pour Ensemble - ISO 5055.

Ce module contient les fonctions de soft voting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MODEL_NAMES: tuple[str, ...] = ("CatBoost", "XGBoost", "LightGBM")


def compute_soft_voting(
    test_matrix: NDArray[np.float64],
    weights: dict[str, float] | None = None,
) -> NDArray[np.float64]:
    """Calcule le soft voting (moyenne ponderee des probabilites).

    Args:
    ----
        test_matrix: Matrice (n_samples, n_models) des probabilites
        weights: Poids par modele (None = moyenne simple)

    Returns:
    -------
        Probabilites moyennees
    """
    if weights is None:
        return test_matrix.mean(axis=1)

    weight_array = np.array([weights.get(name, 1.0) for name in MODEL_NAMES])
    weight_array = weight_array / weight_array.sum()
    return np.average(test_matrix, weights=weight_array, axis=1)
