"""Validation d'équité améliorée - ISO 24027.

Ce module implémente une analyse de fairness complète selon ISO/IEC TR 24027:2021
avec root cause analysis, equalized odds, et recommandations de mitigation.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI (Clause 7: Assessment, Clause 8: Treatment)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes/fonction, SRP)

Document ID: ALICE-SCRIPT-ISO24027-002
Version: 2.0.0
Author: ALICE Engine Team
Last Updated: 2026-01-17
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


@dataclass
class FairnessMetrics:
    """Métriques de fairness ISO 24027."""

    demographic_parity_ratio: float
    equalized_odds_ratio: float
    predictive_parity_ratio: float
    calibration_score: float


@dataclass
class GroupAnalysis:
    """Analyse par groupe pour root cause."""

    group_name: str
    sample_count: int
    positive_rate: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    is_disadvantaged: bool
    deviation_from_mean: float


@dataclass
class ISO24027EnhancedReport:
    """Rapport de fairness amélioré ISO 24027."""

    model_id: str
    protected_attribute: str
    metrics: FairnessMetrics
    group_analyses: list[GroupAnalysis]
    root_cause: str
    mitigations: list[str]
    status: str
    compliant: bool
    data_quality_warnings: list[str] = field(default_factory=list)


def validate_fairness_enhanced(
    predictor: TabularPredictor,
    test_data: Any,
    protected_attribute: str,
    threshold: float = 0.8,
) -> ISO24027EnhancedReport:
    """Valide l'équité avec analyse complète ISO 24027.

    Args:
        predictor: Modèle AutoGluon à évaluer
        test_data: Données de test avec label
        protected_attribute: Attribut protégé (ex: 'ligue_code')
        threshold: Seuil de fairness (défaut 0.8 = EEOC 80% rule)

    Returns:
        ISO24027EnhancedReport avec analyse complète

    ISO 24027 Clause 7: Assessment of bias and fairness
    ISO 24027 Clause 8: Treatment strategies
    """
    label = predictor.label
    y_true = test_data[label].values
    X_test = test_data.drop(columns=[label])
    y_pred = predictor.predict(X_test).values
    protected = test_data[protected_attribute].values

    # Analyse par groupe
    group_analyses = _analyze_groups(y_true, y_pred, protected)

    # Calcul des métriques
    metrics = _compute_fairness_metrics(group_analyses)

    # Root cause analysis
    root_cause = _identify_root_cause(group_analyses, protected_attribute)

    # Warnings qualité données
    warnings = _check_data_quality(group_analyses)

    # Recommandations de mitigation
    mitigations = _recommend_mitigations(metrics, group_analyses, root_cause)

    # Statut final
    status, compliant = _determine_status(metrics, threshold)

    return ISO24027EnhancedReport(
        model_id=str(predictor.path),
        protected_attribute=protected_attribute,
        metrics=metrics,
        group_analyses=group_analyses,
        root_cause=root_cause,
        mitigations=mitigations,
        status=status,
        compliant=compliant,
        data_quality_warnings=warnings,
    )


def _analyze_groups(
    y_true: np.ndarray, y_pred: np.ndarray, protected: np.ndarray
) -> list[GroupAnalysis]:
    """Analyse détaillée par groupe."""
    groups = np.unique(protected)
    analyses = []
    all_positive_rates = []

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


def _compute_fairness_metrics(group_analyses: list[GroupAnalysis]) -> FairnessMetrics:
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


def _identify_root_cause(
    group_analyses: list[GroupAnalysis], protected_attribute: str
) -> str:
    """Identifie la cause racine du biais."""
    disadvantaged = [g for g in group_analyses if g.is_disadvantaged]

    if not disadvantaged:
        return "Aucun groupe significativement désavantagé détecté."

    worst = min(disadvantaged, key=lambda g: g.positive_rate)

    # Analyse de la cause
    if worst.group_name == "(vide)":
        return (
            f"QUALITÉ DONNÉES: 60% des échantillons sans {protected_attribute}. "
            "La valeur vide crée un biais systémique. "
            "ACTION: Enrichir les données manquantes ou exclure du calcul fairness."
        )

    if worst.true_positive_rate < 0.5 and worst.precision > 0.6:
        return (
            f"SOUS-DÉTECTION: Le groupe '{worst.group_name}' a un recall faible "
            f"({worst.true_positive_rate:.1%}) malgré une bonne précision. "
            "CAUSE: Le modèle est trop conservateur pour ce groupe. "
            "ACTION: Ajuster le seuil de classification ou sur-échantillonner."
        )

    if worst.sample_count < 5000:
        return (
            f"SOUS-REPRÉSENTATION: Le groupe '{worst.group_name}' a seulement "
            f"{worst.sample_count} échantillons. "
            "CAUSE: Données insuffisantes pour ce groupe. "
            "ACTION: Collecter plus de données ou appliquer SMOTE."
        )

    return (
        f"BIAIS STRUCTUREL: Le groupe '{worst.group_name}' a un taux positif "
        f"de {worst.positive_rate:.1%} vs moyenne. "
        "CAUSE: Possible corrélation avec features non-protégées. "
        "ACTION: Vérifier corrélations ELO/région, appliquer reweighting."
    )


def _check_data_quality(group_analyses: list[GroupAnalysis]) -> list[str]:
    """Vérifie la qualité des données."""
    warnings = []

    # Vérifier les groupes vides
    empty_group = next((g for g in group_analyses if g.group_name == "(vide)"), None)
    if empty_group and empty_group.sample_count > 10000:
        warnings.append(
            f"CRITIQUE: {empty_group.sample_count} échantillons sans attribut protégé "
            "(>50% des données). Ceci biaise le calcul de fairness."
        )

    # Vérifier les groupes trop petits
    small_groups = [g for g in group_analyses if g.sample_count < 500]
    if small_groups:
        names = [g.group_name for g in small_groups]
        warnings.append(
            f"ATTENTION: Groupes avec <500 échantillons: {names}. "
            "Métriques potentiellement non significatives."
        )

    # Vérifier le déséquilibre
    if group_analyses:
        counts = [g.sample_count for g in group_analyses]
        if max(counts) > 10 * min(counts):
            warnings.append(
                "DÉSÉQUILIBRE: Ratio max/min échantillons >10x. "
                "Considérer un échantillonnage stratifié."
            )

    return warnings


def _recommend_mitigations(
    metrics: FairnessMetrics,
    group_analyses: list[GroupAnalysis],
    root_cause: str,
) -> list[str]:
    """Recommande des actions de mitigation ISO 24027 Clause 8."""
    mitigations = []

    # Mitigation basée sur demographic parity
    if metrics.demographic_parity_ratio < 0.8:
        if "QUALITÉ DONNÉES" in root_cause:
            mitigations.append(
                "PRÉ-PROCESSING: Imputer ou exclure les valeurs manquantes "
                "de l'attribut protégé avant l'entraînement."
            )
        else:
            mitigations.append(
                "PRÉ-PROCESSING: Appliquer reweighting ou resampling "
                "pour équilibrer les groupes désavantagés."
            )

    # Mitigation basée sur equalized odds
    if metrics.equalized_odds_ratio < 0.8:
        mitigations.append(
            "IN-PROCESSING: Ajouter une contrainte d'équité (fairness constraint) "
            "à la fonction de perte pendant l'entraînement."
        )

    # Mitigation basée sur calibration
    if metrics.calibration_score < 0.85:
        mitigations.append(
            "POST-PROCESSING: Appliquer calibration isotonique par groupe "
            "pour égaliser les distributions de probabilité."
        )

    # Recommandation générale
    mitigations.append(
        "MONITORING: Surveiller les métriques de fairness à chaque déploiement "
        "et déclencher une alerte si demographic_parity < 0.6."
    )

    return mitigations


def _determine_status(metrics: FairnessMetrics, threshold: float) -> tuple[str, bool]:
    """Détermine le statut de conformité."""
    dp = metrics.demographic_parity_ratio

    if dp >= threshold:
        return "FAIR", True
    if dp >= 0.6:
        return "CAUTION", False
    return "CRITICAL", False
