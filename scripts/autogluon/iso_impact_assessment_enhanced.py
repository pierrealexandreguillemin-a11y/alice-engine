"""Impact Assessment amélioré - ISO 42005:2025.

Ce module implémente une évaluation d'impact AI complète selon ISO/IEC 42005:2025
avec monitoring triggers, lifecycle integration, et processus en 10 étapes.

ISO Compliance:
- ISO/IEC 42005:2025 - AI System Impact Assessment
- ISO/IEC 42001:2023 - AI Management System (Annex A integration)
- ISO/IEC 5055:2021 - Code Quality (<300 lignes, SRP)

Document ID: ALICE-SCRIPT-ISO42005-002
Version: 2.1.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.autogluon.iso_impact_helpers import (
    analyze_all_dimensions,
    assess_transparency,
    define_mitigations,
    define_monitoring_triggers,
)
from scripts.autogluon.iso_impact_types import (
    AssessmentPhase,
    ImpactDimension,
    ISO42005EnhancedReport,
    Mitigation,
    MonitoringTrigger,
    RiskLevel,
)

# Re-export for backward compatibility
__all__ = [
    "AssessmentPhase",
    "ImpactDimension",
    "ISO42005EnhancedReport",
    "Mitigation",
    "MonitoringTrigger",
    "RiskLevel",
    "assess_impact_enhanced",
    "save_report",
]


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
    ----
        fairness_report_path: Chemin vers le rapport ISO 24027
        robustness_report_path: Chemin vers le rapport ISO 24029
        model_card_path: Chemin vers la Model Card ISO 42001
        phase: Phase du lifecycle

    Returns:
    -------
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
    impact_dimensions = analyze_all_dimensions(
        fairness, robustness, model_card, _extract_demographic_parity
    )
    overall_level = _determine_overall_impact(impact_dimensions, dp_ratio)

    # Step 6: Documentation (transparency)
    transparency = assess_transparency(model_card, fairness, robustness)

    # Step 7: Oversight chain
    accountability = [
        "Data Scientist: Model development and validation",
        "ML Engineer: Deployment and monitoring",
        "Product Owner: Business decisions",
        "Compliance Officer: Regulatory oversight",
    ]

    # Step 8: Monitoring triggers
    triggers = define_monitoring_triggers(dp_ratio, noise_tolerance)

    # Step 9: Mitigations
    mitigations = define_mitigations(impact_dimensions, dp_ratio, noise_tolerance)

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
