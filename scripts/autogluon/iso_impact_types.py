"""Types pour Impact Assessment - ISO 42005.

Ce module contient les enums et dataclasses spécifiques à l'évaluation d'impact.

ISO Compliance:
- ISO/IEC 42005:2025 - AI System Impact Assessment
- ISO/IEC 5055:2021 - Code Quality (SRP)

Document ID: ALICE-SCRIPT-ISO42005-TYPES-001
Version: 1.0.0
Author: ALICE Engine Team
Last Updated: 2026-02-12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
