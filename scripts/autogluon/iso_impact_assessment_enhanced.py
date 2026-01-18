"""Impact Assessment amélioré - ISO 42005:2025.

Ce module implémente une évaluation d'impact AI complète selon ISO/IEC 42005:2025
avec monitoring triggers, lifecycle integration, et processus en 10 étapes.

ISO Compliance:
- ISO/IEC 42005:2025 - AI System Impact Assessment
- ISO/IEC 42001:2023 - AI Management System (Annex A integration)
- ISO/IEC 5055:2021 - Code Quality (<100 lignes/fonction)

Document ID: ALICE-SCRIPT-ISO42005-002
Version: 2.0.0
Author: ALICE Engine Team
Last Updated: 2026-01-17
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class RiskLevel(Enum):
    """Niveaux de risque ISO 42005."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AssessmentPhase(Enum):
    """Phases du lifecycle où l'assessment s'applique."""

    PLANNING = "planning"
    PRE_DEPLOYMENT = "pre_deployment"
    OPERATIONAL = "operational"
    MONITORING = "monitoring"
    DECOMMISSIONING = "decommissioning"


@dataclass
class ImpactDimension:
    """Dimension d'impact (individu, groupe, société)."""

    dimension: str
    description: str
    risk_level: RiskLevel
    justification: str
    affected_stakeholders: list[str]
    potential_harms: list[str]
    potential_benefits: list[str]


@dataclass
class MonitoringTrigger:
    """Déclencheur de ré-évaluation automatique."""

    trigger_name: str
    metric: str
    threshold: float
    current_value: float
    is_triggered: bool
    action_required: str


@dataclass
class Mitigation:
    """Mesure de mitigation ISO 42005."""

    measure: str
    status: str  # IMPLEMENTED, PLANNED, NOT_APPLICABLE
    description: str
    effectiveness: str  # HIGH, MEDIUM, LOW
    responsible_party: str


@dataclass
class ISO42005EnhancedReport:
    """Rapport d'impact amélioré ISO 42005."""

    # Metadata
    iso_standard: str = "ISO/IEC 42005:2025"
    assessment_id: str = ""
    assessment_date: str = ""
    assessment_phase: AssessmentPhase = AssessmentPhase.PRE_DEPLOYMENT
    version: str = "2.0.0"

    # System identification
    model_id: str = ""
    system_name: str = ""
    system_purpose: str = ""
    system_domain: str = ""
    intended_users: list[str] = field(default_factory=list)

    # Impact analysis (10-step process)
    impact_dimensions: list[ImpactDimension] = field(default_factory=list)
    overall_impact_level: RiskLevel = RiskLevel.MEDIUM

    # Transparency & Accountability
    transparency_assessment: dict[str, bool] = field(default_factory=dict)
    accountability_chain: list[str] = field(default_factory=list)

    # Mitigations
    mitigations: list[Mitigation] = field(default_factory=list)

    # Monitoring (continuous assessment)
    monitoring_triggers: list[MonitoringTrigger] = field(default_factory=list)
    next_assessment_date: str = ""

    # Decision
    recommendation: str = ""
    conditions: list[str] = field(default_factory=list)
    approver: str = ""


def assess_impact_enhanced(
    fairness_report_path: Path,
    robustness_report_path: Path,
    model_card_path: Path,
    phase: AssessmentPhase = AssessmentPhase.PRE_DEPLOYMENT,
) -> ISO42005EnhancedReport:
    """Génère un assessment d'impact complet ISO 42005.

    Processus en 10 étapes selon ISO 42005:2025:
    1. Scoping - Définir le périmètre
    2. Responsibility assignment - Identifier les responsables
    3. Threshold definition - Définir les seuils
    4. Execution - Collecter les données
    5. Analysis - Analyser les impacts
    6. Documentation - Documenter les résultats
    7. Oversight - Validation hiérarchique
    8. Monitoring - Définir le suivi
    9. Integration - Intégrer au système de gestion des risques
    10. Review - Planifier les révisions

    Args:
        fairness_report_path: Chemin vers le rapport ISO 24027
        robustness_report_path: Chemin vers le rapport ISO 24029
        model_card_path: Chemin vers la Model Card ISO 42001
        phase: Phase du lifecycle

    Returns:
        ISO42005EnhancedReport complet
    """
    # Charger les rapports existants
    fairness = _load_json(fairness_report_path)
    robustness = _load_json(robustness_report_path)
    model_card = _load_json(model_card_path)

    # Step 1-2: Scoping & Responsibility
    assessment_id = f"ALICE-IA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Step 3: Thresholds (from existing reports)
    dp_ratio = _extract_demographic_parity(fairness)
    noise_tolerance = _extract_noise_tolerance(robustness)

    # Step 4-5: Execution & Analysis
    impact_dimensions = _analyze_all_dimensions(fairness, robustness, model_card)
    overall_level = _determine_overall_impact(impact_dimensions, dp_ratio)

    # Step 6: Documentation (transparency)
    transparency = _assess_transparency(model_card, fairness, robustness)

    # Step 7: Oversight chain
    accountability = [
        "Data Scientist: Model development and validation",
        "ML Engineer: Deployment and monitoring",
        "Product Owner: Business decisions",
        "Compliance Officer: Regulatory oversight",
    ]

    # Step 8: Monitoring triggers
    triggers = _define_monitoring_triggers(dp_ratio, noise_tolerance)

    # Step 9: Mitigations
    mitigations = _define_mitigations(impact_dimensions, dp_ratio, noise_tolerance)

    # Step 10: Review schedule
    next_review = _calculate_next_review(overall_level, phase)

    # Decision
    recommendation, conditions = _make_recommendation(
        overall_level, dp_ratio, noise_tolerance, triggers
    )

    return ISO42005EnhancedReport(
        assessment_id=assessment_id,
        assessment_date=datetime.now().isoformat(),
        assessment_phase=phase,
        model_id=model_card.get("model_id", "unknown"),
        system_name="ALICE Chess Prediction Engine",
        system_purpose="Predict chess game outcomes for team composition optimization",
        system_domain="Sports analytics - Team chess competitions",
        intended_users=["Club managers", "Team captains", "Chess coaches"],
        impact_dimensions=impact_dimensions,
        overall_impact_level=overall_level,
        transparency_assessment=transparency,
        accountability_chain=accountability,
        mitigations=mitigations,
        monitoring_triggers=triggers,
        next_assessment_date=next_review,
        recommendation=recommendation,
        conditions=conditions,
        approver="",
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Charge un fichier JSON."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _extract_demographic_parity(fairness: dict[str, Any]) -> float:
    """Extrait le ratio de parité démographique (supporte les deux formats)."""
    # Format enhanced: metrics.demographic_parity_ratio
    if "metrics" in fairness and isinstance(fairness["metrics"], dict):
        return fairness["metrics"].get("demographic_parity_ratio", 1.0)
    # Format original: demographic_parity_ratio
    return fairness.get("demographic_parity_ratio", 1.0)


def _extract_noise_tolerance(robustness: dict[str, Any]) -> float:
    """Extrait la tolérance au bruit du rapport de robustesse."""
    # Format enhanced: overall_noise_tolerance
    if "overall_noise_tolerance" in robustness:
        return robustness["overall_noise_tolerance"]
    # Format original: tests.noise_robustness.noise_tolerance
    tests = robustness.get("tests", {})
    noise_test = tests.get("noise_robustness", {})
    return noise_test.get("noise_tolerance", 1.0)


def _analyze_all_dimensions(
    fairness: dict, robustness: dict, model_card: dict
) -> list[ImpactDimension]:
    """Analyse les trois dimensions d'impact ISO 42005."""
    dp_ratio = _extract_demographic_parity(fairness)

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


def _determine_overall_impact(dimensions: list[ImpactDimension], dp_ratio: float) -> RiskLevel:
    """Détermine le niveau d'impact global."""
    risk_levels = [d.risk_level for d in dimensions]

    if RiskLevel.CRITICAL in risk_levels:
        return RiskLevel.CRITICAL
    if RiskLevel.HIGH in risk_levels:
        return RiskLevel.HIGH
    if dp_ratio < 0.6:
        return RiskLevel.HIGH
    if RiskLevel.MEDIUM in risk_levels or dp_ratio < 0.8:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _assess_transparency(model_card: dict, fairness: dict, robustness: dict) -> dict[str, bool]:
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


def _define_monitoring_triggers(dp_ratio: float, noise_tolerance: float) -> list[MonitoringTrigger]:
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


def _define_mitigations(
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


def _calculate_next_review(level: RiskLevel, phase: AssessmentPhase) -> str:
    """Calcule la date de prochaine révision selon le risque."""
    from datetime import timedelta

    now = datetime.now()

    if level == RiskLevel.CRITICAL:
        delta = timedelta(days=7)  # Weekly for critical
    elif level == RiskLevel.HIGH:
        delta = timedelta(days=30)  # Monthly for high
    elif level == RiskLevel.MEDIUM:
        delta = timedelta(days=90)  # Quarterly for medium
    else:
        delta = timedelta(days=180)  # Semi-annual for low

    # Adjust for phase
    if phase == AssessmentPhase.OPERATIONAL:
        delta = delta / 2  # More frequent in production

    return (now + delta).strftime("%Y-%m-%d")


def _make_recommendation(
    level: RiskLevel,
    dp_ratio: float,
    noise_tolerance: float,
    triggers: list[MonitoringTrigger],
) -> tuple[str, list[str]]:
    """Génère la recommandation finale."""
    triggered_critical = [t for t in triggers if t.is_triggered and "critical" in t.trigger_name]

    if triggered_critical or level == RiskLevel.CRITICAL:
        return "REJECTED", [
            "Address all critical triggers before deployment",
            "Conduct root cause analysis on fairness issues",
            "Re-submit for impact assessment after remediation",
        ]

    if level == RiskLevel.HIGH or dp_ratio < 0.6:
        return "CONDITIONAL_APPROVAL", [
            "Deploy with enhanced monitoring",
            "Mandatory weekly fairness review",
            "Implement fairness improvements within 30 days",
            "Limit deployment to pilot regions initially",
        ]

    if level == RiskLevel.MEDIUM or dp_ratio < 0.8:
        return "APPROVED_WITH_MONITORING", [
            "Maintain human oversight for team composition",
            "Monitor regional fairness metrics quarterly",
            "Retrain model if PSI exceeds 0.25",
            "Review if demographic parity drops below 0.6",
            "Schedule fairness improvement for next release",
        ]

    return "APPROVED", [
        "Standard monitoring applies",
        "Quarterly impact re-assessment",
        "Annual comprehensive review",
    ]


def save_report(report: ISO42005EnhancedReport, output_path: Path) -> None:
    """Sauvegarde le rapport en JSON."""
    from dataclasses import asdict

    data = asdict(report)

    # Convert enums to strings
    data["assessment_phase"] = report.assessment_phase.value
    data["overall_impact_level"] = report.overall_impact_level.value
    for dim in data["impact_dimensions"]:
        dim["risk_level"] = dim["risk_level"].value

    output_path.write_text(json.dumps(data, indent=2, default=str))
