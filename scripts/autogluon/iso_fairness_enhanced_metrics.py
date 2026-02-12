"""Calcul des métriques de fairness - ISO 24027.

Ce module contient les fonctions de calcul par groupe et métriques agrégées.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI (Clause 7: Assessment)
- ISO/IEC 5055:2021 - Code Quality (SRP)

Document ID: ALICE-SCRIPT-ISO24027-METRICS-001
Version: 1.0.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

import numpy as np

from scripts.autogluon.iso_types import FairnessMetrics, GroupAnalysis


def analyze_groups(
    y_true: np.ndarray, y_pred: np.ndarray, protected: np.ndarray
) -> list[GroupAnalysis]:
    """Analyse détaillée par groupe."""
    groups = np.unique(protected)
    analyses = []
    all_positive_rates: list[float] = []

    for group in groups:
        mask = protected == group
        n = mask.sum()
        if n < 30:  # Minimum statistique
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        # Métriques de base
        positive_rate = float((y_p == 1).mean())
        all_positive_rates.append(positive_rate)

        # TPR (recall): TP / (TP + FN)
        true_positives = ((y_p == 1) & (y_t == 1)).sum()
        actual_positives = (y_t == 1).sum()
        tpr = true_positives / actual_positives if actual_positives > 0 else 0

        # FPR: FP / (FP + TN)
        false_positives = ((y_p == 1) & (y_t == 0)).sum()
        actual_negatives = (y_t == 0).sum()
        fpr = false_positives / actual_negatives if actual_negatives > 0 else 0

        # Precision: TP / (TP + FP)
        predicted_positives = (y_p == 1).sum()
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0

        analyses.append(
            GroupAnalysis(
                group_name=str(group) if group else "(vide)",
                sample_count=int(n),
                positive_rate=float(positive_rate),
                true_positive_rate=float(tpr),
                false_positive_rate=float(fpr),
                precision=float(precision),
                is_disadvantaged=False,  # Set after mean calculation
                deviation_from_mean=0.0,
            )
        )

    # Calculer la déviation par rapport à la moyenne
    if all_positive_rates:
        mean_rate = np.mean(all_positive_rates)
        for analysis in analyses:
            analysis.deviation_from_mean = analysis.positive_rate - mean_rate
            analysis.is_disadvantaged = analysis.deviation_from_mean < -0.05

    return analyses


def compute_fairness_metrics(group_analyses: list[GroupAnalysis]) -> FairnessMetrics:
    """Calcule les métriques de fairness."""
    if not group_analyses:
        return FairnessMetrics(1.0, 1.0, 1.0, 1.0)

    positive_rates = [g.positive_rate for g in group_analyses]
    tprs = [g.true_positive_rate for g in group_analyses]
    fprs = [g.false_positive_rate for g in group_analyses]
    precisions = [g.precision for g in group_analyses if g.precision > 0]

    # Demographic Parity Ratio (min/max positive rate)
    dp_ratio = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 1.0

    # Equalized Odds Ratio (average of TPR and FPR ratios)
    tpr_ratio = min(tprs) / max(tprs) if max(tprs) > 0 else 1.0
    fpr_ratio = min(fprs) / max(fprs) if max(fprs) > 0 else 1.0
    eo_ratio = (tpr_ratio + fpr_ratio) / 2

    # Predictive Parity Ratio (min/max precision)
    pp_ratio = min(precisions) / max(precisions) if precisions and max(precisions) > 0 else 1.0

    # Calibration Score (1 - variance of positive rates)
    calibration = 1.0 - np.std(positive_rates) if len(positive_rates) > 1 else 1.0

    return FairnessMetrics(
        demographic_parity_ratio=float(dp_ratio),
        equalized_odds_ratio=float(eo_ratio),
        predictive_parity_ratio=float(pp_ratio),
        calibration_score=float(calibration),
    )
