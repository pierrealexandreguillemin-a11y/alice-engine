"""Semantic Memory - Base de connaissance ISO pour ALICE.

Ce module implémente la mémoire sémantique du pipeline ALICE,
inspirée de l'architecture AG-A/MLZero (NeurIPS 2025).

La mémoire sémantique enrichit le système avec:
- Connaissance des normes ISO applicables
- Seuils de conformité et règles de décision
- Stratégies de mitigation recommandées

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management
- ISO/IEC TR 24027:2021 - Bias in AI
- ISO/IEC 24029:2021 - Neural Network Robustness
- ISO/IEC 42005:2025 - AI Impact Assessment

Author: ALICE Engine Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ComplianceStatus(Enum):
    """Statut de conformité ISO."""

    COMPLIANT = "compliant"
    CAUTION = "caution"
    CRITICAL = "critical"


@dataclass
class ISOThreshold:
    """Seuil ISO avec niveaux de conformité."""

    metric: str
    compliant: float
    caution: float
    critical: float
    direction: str = "higher_is_better"  # or "lower_is_better"

    def evaluate(self, value: float) -> ComplianceStatus:
        """Évalue la conformité d'une valeur."""
        if self.direction == "higher_is_better":
            if value >= self.compliant:
                return ComplianceStatus.COMPLIANT
            if value >= self.caution:
                return ComplianceStatus.CAUTION
            return ComplianceStatus.CRITICAL
        # lower_is_better
        if value <= self.compliant:
            return ComplianceStatus.COMPLIANT
        if value <= self.caution:
            return ComplianceStatus.CAUTION
        return ComplianceStatus.CRITICAL


@dataclass
class MitigationStrategy:
    """Stratégie de mitigation ISO."""

    name: str
    phase: str  # pre-processing, in-processing, post-processing
    description: str
    effectiveness: str  # low, medium, high
    implementation: str  # code snippet or reference


@dataclass
class ISOStandard:
    """Définition d'une norme ISO."""

    code: str
    name: str
    version: str
    thresholds: list[ISOThreshold]
    mitigations: list[MitigationStrategy]
    reference_url: str = ""


class ISOSemanticMemory:
    """Mémoire sémantique des normes ISO pour ALICE.

    Cette classe centralise la connaissance des normes ISO applicables
    au système ALICE, permettant une évaluation automatique de conformité.

    Example:
    -------
        memory = ISOSemanticMemory()
        status = memory.evaluate_fairness(0.72)
        # Returns: ComplianceStatus.CAUTION

        mitigations = memory.get_mitigations("fairness", status)
        # Returns list of MitigationStrategy
    """

    def __init__(self) -> None:
        """Initialise la memoire semantique avec les normes ISO."""
        from scripts.agents.iso_knowledge import load_iso_knowledge

        self._standards: dict[str, ISOStandard] = load_iso_knowledge()

    def evaluate_fairness(self, demographic_parity: float) -> ComplianceStatus:
        """Évalue la conformité fairness."""
        threshold = self._standards["fairness"].thresholds[0]
        return threshold.evaluate(demographic_parity)

    def evaluate_robustness(self, noise_tolerance: float) -> ComplianceStatus:
        """Évalue la conformité robustness."""
        threshold = self._standards["robustness"].thresholds[0]
        return threshold.evaluate(noise_tolerance)

    def get_mitigations(self, domain: str, status: ComplianceStatus) -> list[MitigationStrategy]:
        """Retourne les mitigations recommandées pour un statut donné."""
        if domain not in self._standards:
            return []

        mitigations = self._standards[domain].mitigations

        if status == ComplianceStatus.CRITICAL:
            return mitigations  # Toutes les mitigations
        if status == ComplianceStatus.CAUTION:
            return [m for m in mitigations if m.effectiveness in ("high", "medium")]
        return []  # Compliant - pas de mitigation nécessaire

    def get_standard(self, domain: str) -> ISOStandard | None:
        """Retourne la norme ISO pour un domaine."""
        return self._standards.get(domain)

    def get_all_standards(self) -> dict[str, ISOStandard]:
        """Retourne toutes les normes ISO."""
        return self._standards.copy()

    def generate_compliance_report(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Génère un rapport de conformité complet.

        Args:
        ----
            metrics: Dict avec les métriques ISO
                - demographic_parity_ratio
                - noise_tolerance
                - consistency_rate

        Returns:
        -------
            Rapport de conformité avec statuts et recommandations
        """
        report: dict[str, Any] = {"standards": {}, "overall_status": "COMPLIANT"}

        # Évaluer fairness
        if "demographic_parity_ratio" in metrics:
            dp = metrics["demographic_parity_ratio"]
            status = self.evaluate_fairness(dp)
            report["standards"]["fairness"] = {
                "value": dp,
                "status": status.value,
                "mitigations": [m.name for m in self.get_mitigations("fairness", status)],
            }
            if status == ComplianceStatus.CRITICAL:
                report["overall_status"] = "CRITICAL"
            elif status == ComplianceStatus.CAUTION and report["overall_status"] != "CRITICAL":
                report["overall_status"] = "CAUTION"

        # Évaluer robustness
        if "noise_tolerance" in metrics:
            nt = metrics["noise_tolerance"]
            status = self.evaluate_robustness(nt)
            report["standards"]["robustness"] = {
                "value": nt,
                "status": status.value,
                "mitigations": [m.name for m in self.get_mitigations("robustness", status)],
            }
            if status == ComplianceStatus.CRITICAL:
                report["overall_status"] = "CRITICAL"
            elif status == ComplianceStatus.CAUTION and report["overall_status"] != "CRITICAL":
                report["overall_status"] = "CAUTION"

        return report
