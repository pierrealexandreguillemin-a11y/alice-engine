"""Module: scripts/comparison/mcnemar_test.py - McNemar Statistical Tests.

Document ID: ALICE-MOD-MCNEMAR-001
Version: 1.0.0

Implementation du test McNemar 5x2cv (Dietterich 1998).
Recommande pour comparer deux classifieurs.

References
----------
- Dietterich, T.G. (1998). Approximate Statistical Tests for
  Comparing Supervised Classification Learning Algorithms

ISO Compliance:
- ISO/IEC 24029:2021 - Statistical validation robuste
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<250 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class McNemarResult:
    """Resultat du test McNemar.

    Attributes
    ----------
        statistic: Valeur de la statistique de test
        p_value: P-value du test
        significant: True si difference significative (p < alpha)
        effect_size: Taille d'effet (Cohen's h)
        confidence_interval: Intervalle de confiance 95%
        model_a_mean_accuracy: Accuracy moyenne modele A
        model_b_mean_accuracy: Accuracy moyenne modele B
        winner: Nom du gagnant ('model_a', 'model_b', ou None si tie)
    """

    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: tuple[float, float]
    model_a_mean_accuracy: float
    model_b_mean_accuracy: float
    winner: str | None


def mcnemar_simple_test(
    y_true: NDArray[np.int64],
    y_pred_a: NDArray[np.int64],
    y_pred_b: NDArray[np.int64],
    alpha: float = 0.05,
) -> McNemarResult:
    """Test McNemar simple pour deux modeles.

    Args:
    ----
        y_true: Labels vrais
        y_pred_a: Predictions modele A
        y_pred_b: Predictions modele B
        alpha: Seuil de significativite

    Returns:
    -------
        McNemarResult avec statistique et p-value

    ISO 24029: Validation statistique de base.
    """
    # Table de contingence
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # n01: A wrong, B correct
    # n10: A correct, B wrong
    n01 = np.sum(~correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)

    # McNemar avec correction de continuite
    if n01 + n10 == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    # Accuracies
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    # Effect size (Cohen's h)
    effect_size = 2 * (np.arcsin(np.sqrt(acc_a)) - np.arcsin(np.sqrt(acc_b)))

    # Confidence interval pour la difference
    diff = acc_a - acc_b
    se = np.sqrt((acc_a * (1 - acc_a) + acc_b * (1 - acc_b)) / len(y_true))
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    # Determiner le gagnant
    winner = None
    if p_value < alpha:
        winner = "model_a" if acc_a > acc_b else "model_b"

    return McNemarResult(
        statistic=float(chi2_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(effect_size),
        confidence_interval=ci,
        model_a_mean_accuracy=float(acc_a),
        model_b_mean_accuracy=float(acc_b),
        winner=winner,
    )


def mcnemar_5x2cv_test(
    model_a_fit: Callable,
    model_b_fit: Callable,
    model_a_predict: Callable,
    model_b_predict: Callable,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_iterations: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> McNemarResult:
    """Test McNemar 5x2cv (Dietterich 1998).

    Methode recommandee pour comparer deux classifieurs.
    Evite les biais du t-test sur donnees non-independantes.
    Type I error acceptable (< 0.05), puissant pour modeles couteux.

    Args:
    ----
        model_a_fit: Fonction fit(X_train, y_train) pour modele A
        model_b_fit: Fonction fit(X_train, y_train) pour modele B
        model_a_predict: Fonction predict(X_test) pour modele A
        model_b_predict: Fonction predict(X_test) pour modele B
        X: Features
        y: Labels
        n_iterations: Nombre d'iterations (defaut 5)
        alpha: Seuil de significativite
        random_state: Seed pour reproductibilite

    Returns:
    -------
        McNemarResult avec statistique et p-value

    ISO 24029: Validation statistique robuste.
    """
    logger.info(f"Starting McNemar 5x2cv test with {n_iterations} iterations")

    all_n01 = []
    all_n10 = []
    all_acc_a = []
    all_acc_b = []

    rng = np.random.RandomState(random_state)

    for i in range(n_iterations):
        # 2-fold stratified CV
        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rng.randint(0, 10000))

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Entrainer les modeles
            model_a_fit(X_train, y_train)
            model_b_fit(X_train, y_train)

            # Predictions
            pred_a = model_a_predict(X_test)
            pred_b = model_b_predict(X_test)

            # Contingency table
            correct_a = pred_a == y_test
            correct_b = pred_b == y_test

            n01 = np.sum(~correct_a & correct_b)  # A wrong, B correct
            n10 = np.sum(correct_a & ~correct_b)  # A correct, B wrong

            all_n01.append(n01)
            all_n10.append(n10)
            all_acc_a.append(correct_a.mean())
            all_acc_b.append(correct_b.mean())

            logger.debug(f"Iter {i + 1}, Fold {fold_idx + 1}: n01={n01}, n10={n10}")

    # Calculer la statistique 5x2cv
    # Formule de Dietterich (1998)
    all_n01 = np.array(all_n01)
    all_n10 = np.array(all_n10)

    # Difference par fold
    diffs = all_n01 - all_n10

    # Moyenne et variance
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)

    # Statistique t
    if var_diff > 0:
        t_stat = mean_diff / np.sqrt(var_diff / len(diffs))
        # Degres de liberte = n_iterations (5 par defaut)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_iterations))
    else:
        t_stat = 0.0
        p_value = 1.0

    # Accuracies moyennes
    mean_acc_a = np.mean(all_acc_a)
    mean_acc_b = np.mean(all_acc_b)

    # Effect size
    effect_size = 2 * (np.arcsin(np.sqrt(mean_acc_a)) - np.arcsin(np.sqrt(mean_acc_b)))

    # Confidence interval
    diff = mean_acc_a - mean_acc_b
    se = np.sqrt(np.var(np.array(all_acc_a) - np.array(all_acc_b), ddof=1) / len(all_acc_a))
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    # Determiner le gagnant
    winner = None
    if p_value < alpha:
        winner = "model_a" if mean_acc_a > mean_acc_b else "model_b"

    logger.info(f"McNemar 5x2cv: t={t_stat:.3f}, p={p_value:.4f}, significant={p_value < alpha}")

    return McNemarResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(effect_size),
        confidence_interval=ci,
        model_a_mean_accuracy=float(mean_acc_a),
        model_b_mean_accuracy=float(mean_acc_b),
        winner=winner,
    )


def bootstrap_confidence_interval(
    accuracies_a: list[float],
    accuracies_b: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Calcule l'intervalle de confiance bootstrap pour la difference.

    Args:
    ----
        accuracies_a: Accuracies du modele A
        accuracies_b: Accuracies du modele B
        n_bootstrap: Nombre d'echantillons bootstrap
        confidence: Niveau de confiance
        random_state: Seed pour reproductibilite

    Returns:
    -------
        Tuple (lower, upper) de l'intervalle de confiance
    """
    rng = np.random.RandomState(random_state)
    diffs = np.array(accuracies_a) - np.array(accuracies_b)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        bootstrap_diffs.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_diffs, alpha * 100)
    upper = np.percentile(bootstrap_diffs, (1 - alpha) * 100)

    return (float(lower), float(upper))
