"""Calcul des metriques de biais - ISO 24027.

Ce module contient les fonctions de calcul de metriques:
- compute_bias_metrics_by_group: Metriques par groupe
- compute_bias_by_elo_range: Metriques par tranche Elo

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<200 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scripts.fairness.thresholds import DEFAULT_THRESHOLDS, BiasThresholds
from scripts.fairness.types import BiasLevel, BiasMetrics

logger = logging.getLogger(__name__)


def _compute_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le True Positive Rate (Recall)."""
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    return float(true_positives / positives)


def _determine_bias_level(
    spd: float,
    eod: float,
    dir_value: float,
    thresholds: BiasThresholds,
) -> BiasLevel:
    """Determine le niveau de biais global."""
    # Critique si un des seuils critiques est depasse
    if (
        abs(spd) >= thresholds.spd_critical
        or abs(eod) >= thresholds.eod_critical
        or dir_value < 0.6
        or dir_value > 1.5
    ):
        return BiasLevel.CRITICAL

    # Warning si un des seuils warning est depasse
    if (
        abs(spd) >= thresholds.spd_warning
        or abs(eod) >= thresholds.eod_warning
        or dir_value < thresholds.dir_min
        or dir_value > thresholds.dir_max
    ):
        return BiasLevel.WARNING

    return BiasLevel.ACCEPTABLE


def compute_bias_metrics_by_group(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    reference_group: str | None = None,
    thresholds: BiasThresholds | None = None,
) -> list[BiasMetrics]:
    """Calcule les metriques de biais par groupe.

    Args:
    ----
        y_true: Labels reels (0/1)
        y_pred: Predictions (0/1)
        groups: Groupe d'appartenance pour chaque echantillon
        reference_group: Groupe de reference (defaut: groupe majoritaire)
        thresholds: Seuils de biais (defaut: ISO 24027)

    Returns:
    -------
        Liste de BiasMetrics pour chaque groupe

    ISO 24027: Calcul des metriques de fairness standard.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Convertir en arrays numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    # Identifier les groupes uniques
    unique_groups = np.unique(groups)

    # Determiner le groupe de reference (majoritaire si non specifie)
    if reference_group is None:
        group_sizes = {g: np.sum(groups == g) for g in unique_groups}
        reference_group = max(group_sizes, key=group_sizes.get)

    logger.info(f"Analyse biais: {len(unique_groups)} groupes, reference={reference_group}")

    # Calculer metriques du groupe de reference
    ref_mask = groups == reference_group
    ref_positive_rate = np.mean(y_pred[ref_mask]) if np.sum(ref_mask) > 0 else 0
    ref_tpr = _compute_tpr(y_true[ref_mask], y_pred[ref_mask])

    # Calculer metriques pour chaque groupe
    metrics_list = []

    for group in unique_groups:
        metrics = _compute_group_metrics(
            y_true, y_pred, groups, group, ref_positive_rate, ref_tpr, thresholds
        )
        if metrics is not None:
            metrics_list.append(metrics)

    return metrics_list


def _compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group: str,
    ref_positive_rate: float,
    ref_tpr: float,
    thresholds: BiasThresholds,
) -> BiasMetrics | None:
    """Calcule les metriques pour un groupe specifique."""
    mask = groups == group
    group_size = int(np.sum(mask))

    if group_size == 0:
        return None

    # Taux de prediction positive
    positive_rate = float(np.mean(y_pred[mask]))

    # True Positive Rate
    tpr = _compute_tpr(y_true[mask], y_pred[mask])

    # SPD: Statistical Parity Difference
    spd = positive_rate - ref_positive_rate

    # EOD: Equal Opportunity Difference
    eod = tpr - ref_tpr

    # DIR: Disparate Impact Ratio
    if ref_positive_rate > 0:
        dir_value = positive_rate / ref_positive_rate
    else:
        dir_value = 1.0 if positive_rate == 0 else float("inf")

    # Determiner le niveau de biais
    level = _determine_bias_level(spd, eod, dir_value, thresholds)

    return BiasMetrics(
        group_name=str(group),
        group_size=group_size,
        positive_rate=round(positive_rate, 4),
        true_positive_rate=round(tpr, 4),
        spd=round(spd, 4),
        eod=round(eod, 4),
        dir=round(dir_value, 4),
        level=level,
    )


def compute_bias_by_elo_range(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    elo_ratings: np.ndarray | pd.Series,
    bins: list[int] | None = None,
    thresholds: BiasThresholds | None = None,
) -> list[BiasMetrics]:
    """Calcule les metriques de biais par tranche Elo.

    Args:
    ----
        y_true: Labels reels (0/1)
        y_pred: Predictions (0/1)
        elo_ratings: Classement Elo de chaque joueur
        bins: Tranches Elo (defaut: [0, 1400, 1600, 1800, 2000, 2200, 2400, 3000])
        thresholds: Seuils de biais

    Returns:
    -------
        Liste de BiasMetrics par tranche Elo

    ISO 24027: Analyse par attribut sensible (niveau de jeu).
    """
    if bins is None:
        # Tranches standard FFE
        bins = [0, 1400, 1600, 1800, 2000, 2200, 2400, 3000]

    elo_ratings = np.asarray(elo_ratings)

    # Creer les labels de groupes
    groups = pd.cut(
        elo_ratings,
        bins=bins,
        labels=[f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)],
        include_lowest=True,
    )

    # Groupe de reference: 1800-2000 (niveau intermediaire standard)
    reference_group = "1800-2000"

    return compute_bias_metrics_by_group(
        y_true=y_true,
        y_pred=y_pred,
        groups=groups,
        reference_group=reference_group,
        thresholds=thresholds,
    )
