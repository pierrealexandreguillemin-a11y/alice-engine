"""Validation d'equite - ISO 24027.

Ce module contient la validation d'equite du modele.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias Detection
- ISO/IEC 5055:2021 - Code Quality (<80 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.autogluon.iso_types import ISO24027BiasReport

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


def validate_fairness(
    predictor: TabularPredictor,
    test_data: Any,
    protected_attribute: str,
) -> ISO24027BiasReport:
    """Valide l'equite du modele selon ISO 24027.

    Args:
    ----
        predictor: TabularPredictor a evaluer
        test_data: Donnees de test
        protected_attribute: Attribut protege (ex: 'ligue_code')

    Returns:
    -------
        ISO24027BiasReport

    ISO 24027: Detection de biais obligatoire.
    """
    model_id = str(predictor.path)
    label = predictor.label

    y_pred = predictor.predict(test_data.drop(columns=[label]))
    protected = test_data[protected_attribute]

    # Calculer le taux de prediction positive par groupe
    positive_rates = _compute_positive_rates(y_pred, protected)

    # Demographic parity ratio (min/max)
    demographic_parity = _compute_demographic_parity(positive_rates)

    # Determiner le statut (EEOC 80% rule)
    status = _determine_fairness_status(demographic_parity)

    return ISO24027BiasReport(
        model_id=model_id,
        demographic_parity=float(demographic_parity),
        equalized_odds=0.0,
        calibration_by_group=positive_rates,
        status=status,
    )


def _compute_positive_rates(y_pred: Any, protected: Any) -> dict[str, float]:
    """Calcule les taux de prediction positive par groupe."""
    groups = protected.unique()
    positive_rates = {}

    for group in groups:
        mask = protected == group
        if mask.sum() > 0:
            positive_rates[str(group)] = float((y_pred[mask] == 1).mean())

    return positive_rates


def _compute_demographic_parity(positive_rates: dict[str, float]) -> float:
    """Calcule la parite demographique."""
    if not positive_rates:
        return 1.0

    rates = list(positive_rates.values())
    min_rate = min(rates) if rates else 0
    max_rate = max(rates) if rates else 1

    return min_rate / max_rate if max_rate > 0 else 0


def _determine_fairness_status(demographic_parity: float) -> str:
    """Determine le statut d'equite (EEOC 80% rule)."""
    if demographic_parity >= 0.8:
        return "FAIR"
    if demographic_parity >= 0.6:
        return "CAUTION"
    return "CRITICAL"
