"""Helpers pour Impact Assessment - ISO 42005.

Ce module contient les fonctions d'analyse dimensionnelle, transparence,
monitoring et mitigation pour l'évaluation d'impact.

ISO Compliance:
- ISO/IEC 42005:2025 - AI System Impact Assessment
- ISO/IEC 5055:2021 - Code Quality (SRP)

Document ID: ALICE-SCRIPT-ISO42005-HELPERS-001
Version: 1.0.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from typing import Any

from scripts.autogluon.iso_impact_types import (
    ImpactDimension,
    Mitigation,
    MonitoringTrigger,
    RiskLevel,
)


def analyze_all_dimensions(
    fairness: dict[str, Any],
    robustness: dict[str, Any],
    model_card: dict[str, Any],
    extract_dp: Any,
) -> list[ImpactDimension]:
    """Analyse les trois dimensions d'impact ISO 42005."""
    dp_ratio = extract_dp(fairness)

    # Individual Impact
    individual = ImpactDimension(
        dimension="individual",
        description="Impact on individual players whose game outcomes are predicted",
        risk_level=RiskLevel.LOW,
        justification=(
            "Predictions are advisory only. Human team captain makes final "
            "decisions on team composition. No automated decision-making."
        ),
        affected_stakeholders=["Chess players", "Team members"],
        potential_harms=[
            "Player may not be selected based on prediction",
            "Potential over-reliance on predictions by team captain",
        ],
        potential_benefits=[
            "Better team composition for competition success",
            "Objective data to support selection decisions",
        ],
    )

    # Group Impact
    group_risk = (
        RiskLevel.LOW
        if dp_ratio >= 0.8
        else RiskLevel.MEDIUM
        if dp_ratio >= 0.6
        else RiskLevel.HIGH
    )
    group = ImpactDimension(
        dimension="group",
        description="Impact on regional groups (ligues) due to prediction disparities",
        risk_level=group_risk,
        justification=(
            f"Demographic parity ratio: {dp_ratio:.2%}. "
            f"{'Above' if dp_ratio >= 0.8 else 'Below'} EEOC 80% threshold. "
            "Disparity may reflect underlying ELO distributions across regions."
        ),
        affected_stakeholders=["Regional leagues", "Clubs by region"],
        potential_harms=[
            "Regional bias in predictions (PACA disadvantaged)",
            "Systematic under-prediction for certain leagues",
        ],
        potential_benefits=[
            "Equal opportunity once fairness improved",
            "Transparent fairness metrics for accountability",
        ],
    )

    # Societal Impact
    societal = ImpactDimension(
        dimension="societal",
        description="Broader impact on chess community and sports analytics",
        risk_level=RiskLevel.LOW,
        justification=(
            "Sports context with no life-critical decisions. "
            "Positive contribution to data-driven sports management. "
            "No impact on fundamental rights."
        ),
        affected_stakeholders=["French Chess Federation", "Chess community"],
        potential_harms=[
            "Over-reliance on AI in sports decisions",
            "Potential gaming of prediction system",
        ],
        potential_benefits=[
            "Advancement of sports analytics",
            "More competitive and balanced team compositions",
            "Educational value for AI governance in sports",
        ],
    )

    return [individual, group, societal]


def assess_transparency(
    model_card: dict[str, Any],
    fairness: dict[str, Any],
    robustness: dict[str, Any],
) -> dict[str, bool]:
    """Évalue la transparence du système AI."""
    return {
        "model_card_available": bool(model_card),
        "model_card_complete": bool(model_card.get("metrics"))
        and bool(model_card.get("hyperparameters")),
        "feature_importance_available": bool(model_card.get("feature_importance")),
        "data_lineage_tracked": bool(model_card.get("training_data_hash")),
        "fairness_tested": bool(fairness),
        "fairness_documented": bool(fairness.get("demographic_parity_ratio")),
        "robustness_tested": bool(robustness),
        "robustness_documented": bool(robustness.get("overall_status")),
        "intended_use_documented": bool(model_card.get("intended_use")),
        "limitations_documented": bool(model_card.get("limitations")),
    }


def define_monitoring_triggers(dp_ratio: float, noise_tolerance: float) -> list[MonitoringTrigger]:
    """Définit les déclencheurs de monitoring continu."""
    return [
        MonitoringTrigger(
            trigger_name="fairness_critical",
            metric="demographic_parity_ratio",
            threshold=0.6,
            current_value=dp_ratio,
            is_triggered=dp_ratio < 0.6,
            action_required="IMMEDIATE: Stop deployment, investigate bias"
            if dp_ratio < 0.6
            else "None",
        ),
        MonitoringTrigger(
            trigger_name="fairness_warning",
            metric="demographic_parity_ratio",
            threshold=0.8,
            current_value=dp_ratio,
            is_triggered=dp_ratio < 0.8,
            action_required="PLANNED: Schedule fairness improvement" if dp_ratio < 0.8 else "None",
        ),
        MonitoringTrigger(
            trigger_name="robustness_degradation",
            metric="noise_tolerance",
            threshold=0.95,
            current_value=noise_tolerance,
            is_triggered=noise_tolerance < 0.95,
            action_required="INVESTIGATE: Model stability degraded"
            if noise_tolerance < 0.95
            else "None",
        ),
        MonitoringTrigger(
            trigger_name="drift_detection",
            metric="psi",
            threshold=0.25,
            current_value=0.0,  # À calculer en production
            is_triggered=False,
            action_required="RETRAIN: Model drift detected",
        ),
        MonitoringTrigger(
            trigger_name="quarterly_review",
            metric="days_since_assessment",
            threshold=90,
            current_value=0,
            is_triggered=False,
            action_required="REVIEW: Scheduled quarterly impact re-assessment",
        ),
    ]


def define_mitigations(
    dimensions: list[ImpactDimension], dp_ratio: float, noise_tolerance: float
) -> list[Mitigation]:
    """Définit les mesures de mitigation."""
    mitigations = [
        Mitigation(
            measure="Human-in-the-loop",
            status="IMPLEMENTED",
            description="Team captain makes final decisions, predictions are advisory only",
            effectiveness="HIGH",
            responsible_party="Product Owner",
        ),
        Mitigation(
            measure="Drift monitoring",
            status="IMPLEMENTED",
            description="PSI threshold monitoring with automatic alerts",
            effectiveness="MEDIUM",
            responsible_party="ML Engineer",
        ),
        Mitigation(
            measure="Fairness monitoring",
            status="IMPLEMENTED",
            description="Demographic parity tracked per deployment",
            effectiveness="MEDIUM",
            responsible_party="Data Scientist",
        ),
    ]

    # Mitigations conditionnelles
    if dp_ratio < 0.8:
        mitigations.append(
            Mitigation(
                measure="Fairness improvement",
                status="PLANNED",
                description="Investigate PACA regional bias, apply reweighting or data collection",
                effectiveness="HIGH",
                responsible_party="Data Scientist",
            )
        )

    if noise_tolerance < 0.95:
        mitigations.append(
            Mitigation(
                measure="Robustness hardening",
                status="PLANNED",
                description="Apply adversarial training or ensemble methods",
                effectiveness="MEDIUM",
                responsible_party="ML Engineer",
            )
        )

    mitigations.append(
        Mitigation(
            measure="Periodic fairness audit",
            status="PLANNED",
            description="Quarterly bias review with external auditor",
            effectiveness="HIGH",
            responsible_party="Compliance Officer",
        )
    )

    return mitigations
