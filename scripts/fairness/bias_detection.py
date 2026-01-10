"""Module: scripts/fairness/bias_detection.py - Détection des biais ML.

Document ID: ALICE-MOD-BIAS-001
Version: 1.0.0
Lines: ~480 (ISO 5055 compliant)

Ce module implémente les métriques de fairness pour détecter
les biais dans les prédictions des modèles ALICE.

Métriques implémentées (ISO 24027 / Fairlearn):
- SPD (Statistical Parity Difference): Différence de taux de prédiction positive
  P(Y_pred=1|G=g) - P(Y_pred=1|G=ref)
- EOD (Equal Opportunity Difference): Différence de TPR entre groupes
  TPR(G=g) - TPR(G=ref)
- DIR (Disparate Impact Ratio): Ratio de taux de prédiction positive
  P(Y_pred=1|G=g) / P(Y_pred=1|G=ref)

Seuils (basés sur recherche académique et réglementations):
- SPD: |SPD| < 0.1 (acceptable), >= 0.2 (critique)
- DIR: 0.8 <= DIR <= 1.25 (règle des 4/5, EEOC guidelines)
- EOD: |EOD| < 0.1 (acceptable), >= 0.2 (critique)

Classes:
- BiasLevel: Enum (ACCEPTABLE, WARNING, CRITICAL)
- BiasThresholds: Dataclass seuils configurables
- BiasMetrics: Dataclass métriques par groupe
- BiasReport: Dataclass rapport complet

Functions:
- compute_bias_metrics_by_group(): Calcul métriques par groupe
- compute_bias_by_elo_range(): Analyse par tranche Elo FFE
- check_bias_thresholds(): Vérification des seuils
- generate_fairness_report(): Génération rapport ISO

Sources académiques:
- Fairlearn: https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html
- EEOC: https://www.eeoc.gov/laws/guidance/disparate-impact-theory
- Feldman et al. "Certifying and Removing Disparate Impact" (KDD 2015)
- Hardt et al. "Equality of Opportunity in Supervised Learning" (NeurIPS 2016)

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems (primary)
- ISO/IEC 42001:2023 - AI Management System (traçabilité)
- ISO/IEC 25059:2023 - AI Quality Model (métriques)
- ISO/IEC 5055:2021 - Code Quality (module <500 lignes)

See Also
--------
- scripts/fairness/__init__.py - Package exports
- tests/test_fairness_bias_detection.py - 29 tests unitaires
- docs/iso/AI_RISK_ASSESSMENT.md - Section R3: Fairness Risks
- docs/iso/STATEMENT_OF_APPLICABILITY.md - Control B.4.4

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BiasLevel(Enum):
    """Niveau de biais selon ISO 24027."""

    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BiasThresholds:
    """Seuils de biais ISO 24027.

    Sources:
    - EEOC 4/5 rule: DIR entre 0.8 et 1.25
    - Fairlearn recommendations: SPD < 0.1
    - Academic consensus: EOD < 0.1
    """

    spd_warning: float = 0.1
    spd_critical: float = 0.2
    dir_min: float = 0.8  # EEOC 4/5 rule
    dir_max: float = 1.25
    eod_warning: float = 0.1
    eod_critical: float = 0.2


# Seuils par défaut (ISO 24027 + EEOC + Fairlearn)
DEFAULT_THRESHOLDS = BiasThresholds()


@dataclass
class BiasMetrics:
    """Métriques de biais pour un groupe.

    Attributes
    ----------
        group_name: Nom du groupe (ex: "N1", "GM", "1800-2000")
        group_size: Nombre d'échantillons dans le groupe
        positive_rate: Taux de prédiction positive
        true_positive_rate: TPR (recall) du groupe
        spd: Statistical Parity Difference vs référence
        eod: Equal Opportunity Difference vs référence
        dir: Disparate Impact Ratio vs référence
        level: Niveau de biais (acceptable/warning/critical)
    """

    group_name: str
    group_size: int
    positive_rate: float
    true_positive_rate: float
    spd: float = 0.0
    eod: float = 0.0
    dir: float = 1.0
    level: BiasLevel = BiasLevel.ACCEPTABLE


@dataclass
class BiasReport:
    """Rapport complet de biais ISO 24027.

    Attributes
    ----------
        timestamp: Date/heure du rapport
        model_name: Nom du modèle analysé
        total_samples: Nombre total d'échantillons
        metrics_by_group: Métriques par groupe
        overall_level: Niveau de biais global
        recommendations: Recommandations d'action
    """

    timestamp: str
    model_name: str
    total_samples: int
    feature_analyzed: str
    reference_group: str
    metrics_by_group: list[BiasMetrics] = field(default_factory=list)
    overall_level: BiasLevel = BiasLevel.ACCEPTABLE
    recommendations: list[str] = field(default_factory=list)
    iso_compliance: dict[str, Any] = field(default_factory=dict)


def compute_bias_metrics_by_group(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    reference_group: str | None = None,
    thresholds: BiasThresholds | None = None,
) -> list[BiasMetrics]:
    """Calcule les métriques de biais par groupe.

    Args:
    ----
        y_true: Labels réels (0/1)
        y_pred: Prédictions (0/1)
        groups: Groupe d'appartenance pour chaque échantillon
        reference_group: Groupe de référence (défaut: groupe majoritaire)
        thresholds: Seuils de biais (défaut: ISO 24027)

    Returns:
    -------
        Liste de BiasMetrics pour chaque groupe

    ISO 24027: Calcul des métriques de fairness standard.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Convertir en arrays numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    # Identifier les groupes uniques
    unique_groups = np.unique(groups)

    # Déterminer le groupe de référence (majoritaire si non spécifié)
    if reference_group is None:
        group_sizes = {g: np.sum(groups == g) for g in unique_groups}
        reference_group = max(group_sizes, key=group_sizes.get)

    logger.info(f"Analyse biais: {len(unique_groups)} groupes, référence={reference_group}")

    # Calculer métriques du groupe de référence
    ref_mask = groups == reference_group
    ref_positive_rate = np.mean(y_pred[ref_mask]) if np.sum(ref_mask) > 0 else 0
    ref_tpr = _compute_tpr(y_true[ref_mask], y_pred[ref_mask])

    # Calculer métriques pour chaque groupe
    metrics_list = []

    for group in unique_groups:
        mask = groups == group
        group_size = int(np.sum(mask))

        if group_size == 0:
            continue

        # Taux de prédiction positive
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

        # Déterminer le niveau de biais
        level = _determine_bias_level(spd, eod, dir_value, thresholds)

        metrics_list.append(
            BiasMetrics(
                group_name=str(group),
                group_size=group_size,
                positive_rate=round(positive_rate, 4),
                true_positive_rate=round(tpr, 4),
                spd=round(spd, 4),
                eod=round(eod, 4),
                dir=round(dir_value, 4),
                level=level,
            )
        )

    return metrics_list


def compute_bias_by_elo_range(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    elo_ratings: np.ndarray | pd.Series,
    bins: list[int] | None = None,
    thresholds: BiasThresholds | None = None,
) -> list[BiasMetrics]:
    """Calcule les métriques de biais par tranche Elo.

    Args:
    ----
        y_true: Labels réels (0/1)
        y_pred: Prédictions (0/1)
        elo_ratings: Classement Elo de chaque joueur
        bins: Tranches Elo (défaut: [0, 1400, 1600, 1800, 2000, 2200, 2400, 3000])
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

    # Créer les labels de groupes
    groups = pd.cut(
        elo_ratings,
        bins=bins,
        labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)],
        include_lowest=True,
    )

    # Groupe de référence: 1800-2000 (niveau intermédiaire standard)
    reference_group = "1800-2000"

    return compute_bias_metrics_by_group(
        y_true=y_true,
        y_pred=y_pred,
        groups=groups,
        reference_group=reference_group,
        thresholds=thresholds,
    )


def check_bias_thresholds(
    metrics: list[BiasMetrics],
    thresholds: BiasThresholds | None = None,
) -> tuple[BiasLevel, list[str]]:
    """Vérifie les seuils de biais et génère des alertes.

    Args:
    ----
        metrics: Liste des métriques par groupe
        thresholds: Seuils de biais

    Returns:
    -------
        Tuple (niveau_global, liste_alertes)

    ISO 24027: Détection des dépassements de seuils.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    alerts = []
    worst_level = BiasLevel.ACCEPTABLE

    for m in metrics:
        group_alerts = []

        # Vérifier SPD
        if abs(m.spd) >= thresholds.spd_critical:
            group_alerts.append(f"SPD critique ({m.spd:.3f})")
            worst_level = BiasLevel.CRITICAL
        elif abs(m.spd) >= thresholds.spd_warning:
            group_alerts.append(f"SPD warning ({m.spd:.3f})")
            if worst_level != BiasLevel.CRITICAL:
                worst_level = BiasLevel.WARNING

        # Vérifier DIR (règle EEOC 4/5)
        if m.dir < thresholds.dir_min or m.dir > thresholds.dir_max:
            if m.dir < 0.6 or m.dir > 1.5:
                group_alerts.append(f"DIR critique ({m.dir:.3f})")
                worst_level = BiasLevel.CRITICAL
            else:
                group_alerts.append(f"DIR warning ({m.dir:.3f})")
                if worst_level != BiasLevel.CRITICAL:
                    worst_level = BiasLevel.WARNING

        # Vérifier EOD
        if abs(m.eod) >= thresholds.eod_critical:
            group_alerts.append(f"EOD critique ({m.eod:.3f})")
            worst_level = BiasLevel.CRITICAL
        elif abs(m.eod) >= thresholds.eod_warning:
            group_alerts.append(f"EOD warning ({m.eod:.3f})")
            if worst_level != BiasLevel.CRITICAL:
                worst_level = BiasLevel.WARNING

        if group_alerts:
            alerts.append(f"Groupe '{m.group_name}': {', '.join(group_alerts)}")

    return worst_level, alerts


def generate_fairness_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    model_name: str = "Unknown",
    feature_name: str = "Unknown",
    reference_group: str | None = None,
    thresholds: BiasThresholds | None = None,
) -> BiasReport:
    """Génère un rapport de fairness complet ISO 24027.

    Args:
    ----
        y_true: Labels réels
        y_pred: Prédictions
        groups: Groupes d'appartenance
        model_name: Nom du modèle
        feature_name: Nom de la feature analysée
        reference_group: Groupe de référence
        thresholds: Seuils de biais

    Returns:
    -------
        BiasReport complet avec métriques et recommandations

    ISO 24027: Rapport de biais traçable.
    ISO 42001: Documentation des décisions AI.
    """
    logger.info(f"Génération rapport fairness: {model_name}, feature={feature_name}")

    # Calculer métriques
    metrics = compute_bias_metrics_by_group(
        y_true=y_true,
        y_pred=y_pred,
        groups=groups,
        reference_group=reference_group,
        thresholds=thresholds,
    )

    # Vérifier seuils
    overall_level, alerts = check_bias_thresholds(metrics, thresholds)

    # Générer recommandations
    recommendations = _generate_recommendations(metrics, overall_level)

    # Déterminer le groupe de référence utilisé
    if reference_group is None:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        group_sizes = {g: np.sum(groups_arr == g) for g in unique_groups}
        reference_group = str(max(group_sizes, key=group_sizes.get))

    # Créer le rapport
    report = BiasReport(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        total_samples=len(y_true),
        feature_analyzed=feature_name,
        reference_group=reference_group,
        metrics_by_group=metrics,
        overall_level=overall_level,
        recommendations=recommendations,
        iso_compliance={
            "iso_24027": True,
            "iso_42001": True,
            "eeoc_4_5_rule": all(0.8 <= m.dir <= 1.25 for m in metrics if m.dir != float("inf")),
            "alerts": alerts,
        },
    )

    logger.info(f"  Niveau global: {overall_level.value}")
    logger.info(f"  {len(alerts)} alertes générées")

    return report


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
    """Détermine le niveau de biais global."""
    # Critique si un des seuils critiques est dépassé
    if (
        abs(spd) >= thresholds.spd_critical
        or abs(eod) >= thresholds.eod_critical
        or dir_value < 0.6
        or dir_value > 1.5
    ):
        return BiasLevel.CRITICAL

    # Warning si un des seuils warning est dépassé
    if (
        abs(spd) >= thresholds.spd_warning
        or abs(eod) >= thresholds.eod_warning
        or dir_value < thresholds.dir_min
        or dir_value > thresholds.dir_max
    ):
        return BiasLevel.WARNING

    return BiasLevel.ACCEPTABLE


def _generate_recommendations(
    metrics: list[BiasMetrics],
    overall_level: BiasLevel,
) -> list[str]:
    """Génère des recommandations basées sur les métriques."""
    recommendations = []

    if overall_level == BiasLevel.CRITICAL:
        recommendations.append(
            "URGENT: Biais critique détecté. Suspendre déploiement et investiguer."
        )

    # Analyser les groupes problématiques
    critical_groups = [m for m in metrics if m.level == BiasLevel.CRITICAL]
    warning_groups = [m for m in metrics if m.level == BiasLevel.WARNING]

    if critical_groups:
        group_names = ", ".join(m.group_name for m in critical_groups)
        recommendations.append(
            f"Groupes critiques: {group_names}. Rééquilibrer données d'entraînement."
        )

    if warning_groups:
        group_names = ", ".join(m.group_name for m in warning_groups)
        recommendations.append(
            f"Groupes à surveiller: {group_names}. Monitoring renforcé recommandé."
        )

    # Recommandations spécifiques
    for m in metrics:
        if m.dir < 0.8:
            recommendations.append(
                f"Groupe '{m.group_name}': sous-représenté (DIR={m.dir:.2f}). "
                "Considérer oversampling ou ajustement des seuils."
            )
        elif m.dir > 1.25:
            recommendations.append(
                f"Groupe '{m.group_name}': sur-représenté (DIR={m.dir:.2f}). "
                "Considérer undersampling ou post-processing."
            )

    if overall_level == BiasLevel.ACCEPTABLE and not recommendations:
        recommendations.append("Aucun biais significatif détecté. Maintenir monitoring régulier.")

    return recommendations
