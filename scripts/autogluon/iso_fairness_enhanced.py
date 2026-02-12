"""Validation d'équité améliorée - ISO 24027.

Ce module implémente une analyse de fairness complète selon ISO/IEC TR 24027:2021
avec root cause analysis, equalized odds, et recommandations de mitigation.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI (Clause 7: Assessment, Clause 8: Treatment)
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, SRP)

Document ID: ALICE-SCRIPT-ISO24027-002
Version: 2.1.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

from scripts.autogluon.iso_fairness_enhanced_metrics import (
    analyze_groups,
    compute_fairness_metrics,
)
from scripts.autogluon.iso_types import (
    FairnessMetrics,
    GroupAnalysis,
    ISO24027EnhancedReport,
)

# Re-export for backward compatibility
__all__ = [
    "FairnessMetrics",
    "GroupAnalysis",
    "ISO24027EnhancedReport",
    "validate_fairness_enhanced",
]


def validate_fairness_enhanced(
    predictor: TabularPredictor,
    test_data: Any,
    protected_attribute: str,
    threshold: float = 0.8,
    exclude_empty: bool = True,
) -> ISO24027EnhancedReport:
    """Valide l'équité avec analyse complète ISO 24027.

    Args:
    ----
        predictor: Modèle AutoGluon à évaluer
        test_data: Données de test avec label
        protected_attribute: Attribut protégé (ex: 'ligue_code')
        threshold: Seuil de fairness (défaut 0.8 = EEOC 80% rule)
        exclude_empty: Exclure les valeurs vides (ex: compétitions nationales)

    Returns:
    -------
        ISO24027EnhancedReport avec analyse complète

    ISO 24027 Clause 7: Assessment of bias and fairness
    ISO 24027 Clause 8: Treatment strategies

    Note: Pour ligue_code, les valeurs vides représentent les compétitions
    nationales où le code ligue n'est pas applicable. exclude_empty=True
    les exclut du calcul de fairness (comportement correct).
    """
    label = predictor.label

    # Filtrer les valeurs vides si demandé
    if exclude_empty:
        mask = test_data[protected_attribute].astype(str).str.strip() != ""
        test_data_filtered = test_data[mask].copy()
        excluded_count = (~mask).sum()
    else:
        test_data_filtered = test_data
        excluded_count = 0

    y_true = test_data_filtered[label].values
    X_test = test_data_filtered.drop(columns=[label])
    y_pred = predictor.predict(X_test).values
    protected = test_data_filtered[protected_attribute].values

    # Analyse par groupe
    group_analyses = analyze_groups(y_true, y_pred, protected)

    # Calcul des métriques
    metrics = compute_fairness_metrics(group_analyses)

    # Root cause analysis
    root_cause = _identify_root_cause(group_analyses, protected_attribute)

    # Warnings qualité données
    warnings = _check_data_quality(group_analyses)

    # Ajouter info sur les exclusions
    if excluded_count > 0:
        warnings.insert(
            0,
            (
                f"INFO: {excluded_count} échantillons exclus (attribut protégé vide). "
                "Pour ligue_code, ceci représente les compétitions nationales "
                "où le code ligue n'est pas applicable."
            ),
        )

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


def _identify_root_cause(group_analyses: list[GroupAnalysis], protected_attribute: str) -> str:
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
    warnings: list[str] = []

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
    mitigations: list[str] = []

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
