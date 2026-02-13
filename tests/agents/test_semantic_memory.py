"""Tests Semantic Memory - ISO 29119.

Document ID: ALICE-TEST-AGENTS-SEMANTIC
Version: 1.0.0
Tests: 26

Classes:
- TestComplianceStatus: Tests enum statut (2 tests)
- TestISOThreshold: Tests seuils ISO (4 tests)
- TestISOSemanticMemory: Tests mémoire sémantique (20 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)
- ISO/IEC 42001:2023 - AI Management

Author: ALICE Engine Team
Last Updated: 2026-02-13
"""

from __future__ import annotations

import pytest

from scripts.agents.semantic_memory import (
    ComplianceStatus,
    ISOSemanticMemory,
    ISOThreshold,
    MitigationStrategy,
)


class TestComplianceStatus:
    """Tests pour l'enum ComplianceStatus."""

    def test_status_values(self):
        """Vérifie les valeurs de statut."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.CAUTION.value == "caution"
        assert ComplianceStatus.CRITICAL.value == "critical"

    def test_status_count(self):
        """Vérifie le nombre de statuts."""
        assert len(ComplianceStatus) == 3


class TestISOThreshold:
    """Tests pour ISOThreshold."""

    def test_evaluate_higher_is_better_compliant(self):
        """Test évaluation compliant (higher is better)."""
        threshold = ISOThreshold(
            metric="test",
            compliant=0.80,
            caution=0.60,
            critical=0.50,
            direction="higher_is_better",
        )
        assert threshold.evaluate(0.85) == ComplianceStatus.COMPLIANT
        assert threshold.evaluate(0.80) == ComplianceStatus.COMPLIANT

    def test_evaluate_higher_is_better_caution(self):
        """Test évaluation caution (higher is better)."""
        threshold = ISOThreshold(
            metric="test",
            compliant=0.80,
            caution=0.60,
            critical=0.50,
            direction="higher_is_better",
        )
        assert threshold.evaluate(0.70) == ComplianceStatus.CAUTION
        assert threshold.evaluate(0.60) == ComplianceStatus.CAUTION

    def test_evaluate_higher_is_better_critical(self):
        """Test évaluation critical (higher is better)."""
        threshold = ISOThreshold(
            metric="test",
            compliant=0.80,
            caution=0.60,
            critical=0.50,
            direction="higher_is_better",
        )
        assert threshold.evaluate(0.55) == ComplianceStatus.CRITICAL
        assert threshold.evaluate(0.40) == ComplianceStatus.CRITICAL

    def test_evaluate_lower_is_better(self):
        """Test évaluation lower is better."""
        threshold = ISOThreshold(
            metric="risk",
            compliant=1.0,
            caution=2.0,
            critical=3.0,
            direction="lower_is_better",
        )
        assert threshold.evaluate(0.5) == ComplianceStatus.COMPLIANT
        assert threshold.evaluate(1.0) == ComplianceStatus.COMPLIANT
        assert threshold.evaluate(1.5) == ComplianceStatus.CAUTION
        assert threshold.evaluate(2.0) == ComplianceStatus.CAUTION
        assert threshold.evaluate(2.5) == ComplianceStatus.CRITICAL
        assert threshold.evaluate(4.0) == ComplianceStatus.CRITICAL


class TestISOSemanticMemory:
    """Tests pour ISOSemanticMemory."""

    @pytest.fixture
    def memory(self) -> ISOSemanticMemory:
        """Fixture mémoire sémantique."""
        return ISOSemanticMemory()

    def test_initialization(self, memory: ISOSemanticMemory):
        """Test initialisation de la mémoire."""
        standards = memory.get_all_standards()
        assert "fairness" in standards
        assert "robustness" in standards
        assert len(standards) >= 2

    def test_evaluate_fairness_compliant(self, memory: ISOSemanticMemory):
        """Test évaluation fairness compliant."""
        status = memory.evaluate_fairness(0.85)
        assert status == ComplianceStatus.COMPLIANT

    def test_evaluate_fairness_caution(self, memory: ISOSemanticMemory):
        """Test évaluation fairness caution."""
        status = memory.evaluate_fairness(0.70)
        assert status == ComplianceStatus.CAUTION

    def test_evaluate_fairness_critical(self, memory: ISOSemanticMemory):
        """Test évaluation fairness critical."""
        status = memory.evaluate_fairness(0.55)
        assert status == ComplianceStatus.CRITICAL

    def test_evaluate_robustness_compliant(self, memory: ISOSemanticMemory):
        """Test évaluation robustness compliant."""
        status = memory.evaluate_robustness(0.96)
        assert status == ComplianceStatus.COMPLIANT

    def test_evaluate_robustness_caution(self, memory: ISOSemanticMemory):
        """Test évaluation robustness caution."""
        status = memory.evaluate_robustness(0.92)
        assert status == ComplianceStatus.CAUTION

    def test_evaluate_robustness_critical(self, memory: ISOSemanticMemory):
        """Test évaluation robustness critical."""
        status = memory.evaluate_robustness(0.87)
        assert status == ComplianceStatus.CRITICAL

    def test_get_mitigations_compliant(self, memory: ISOSemanticMemory):
        """Test mitigations pour statut compliant."""
        mitigations = memory.get_mitigations("fairness", ComplianceStatus.COMPLIANT)
        assert mitigations == []

    def test_get_mitigations_caution(self, memory: ISOSemanticMemory):
        """Test mitigations pour statut caution."""
        mitigations = memory.get_mitigations("fairness", ComplianceStatus.CAUTION)
        assert len(mitigations) > 0
        # Only high and medium effectiveness
        for m in mitigations:
            assert m.effectiveness in ("high", "medium")

    def test_get_mitigations_critical(self, memory: ISOSemanticMemory):
        """Test mitigations pour statut critical."""
        mitigations = memory.get_mitigations("fairness", ComplianceStatus.CRITICAL)
        assert len(mitigations) > 0
        # All mitigations
        standard = memory.get_standard("fairness")
        assert standard is not None
        assert len(mitigations) == len(standard.mitigations)

    def test_get_mitigations_invalid_domain(self, memory: ISOSemanticMemory):
        """Test mitigations pour domaine invalide."""
        mitigations = memory.get_mitigations("invalid", ComplianceStatus.CRITICAL)
        assert mitigations == []

    def test_get_standard(self, memory: ISOSemanticMemory):
        """Test récupération d'une norme."""
        standard = memory.get_standard("fairness")
        assert standard is not None
        assert standard.code == "ISO/IEC TR 24027:2021"
        assert len(standard.thresholds) > 0
        assert len(standard.mitigations) > 0

    def test_get_standard_not_found(self, memory: ISOSemanticMemory):
        """Test récupération norme inexistante."""
        standard = memory.get_standard("nonexistent")
        assert standard is None

    def test_generate_compliance_report_all_compliant(self, memory: ISOSemanticMemory):
        """Test rapport de conformité avec toutes métriques compliant."""
        metrics = {
            "demographic_parity_ratio": 0.85,
            "noise_tolerance": 0.96,
        }
        report = memory.generate_compliance_report(metrics)

        assert report["overall_status"] == "COMPLIANT"
        assert "fairness" in report["standards"]
        assert "robustness" in report["standards"]

        fairness = report["standards"]["fairness"]
        assert fairness["value"] == 0.85
        assert fairness["status"] == "compliant"
        assert fairness["mitigations"] == []

    def test_generate_compliance_report_with_caution(self, memory: ISOSemanticMemory):
        """Test rapport de conformité avec avertissement."""
        metrics = {
            "demographic_parity_ratio": 0.70,
            "noise_tolerance": 0.96,
        }
        report = memory.generate_compliance_report(metrics)

        assert report["overall_status"] == "CAUTION"
        assert "fairness" in report["standards"]

        fairness = report["standards"]["fairness"]
        assert fairness["value"] == 0.70
        assert fairness["status"] == "caution"
        assert len(fairness["mitigations"]) > 0

    def test_generate_compliance_report_with_critical(self, memory: ISOSemanticMemory):
        """Test rapport de conformité avec critique."""
        metrics = {
            "demographic_parity_ratio": 0.55,
            "noise_tolerance": 0.87,
        }
        report = memory.generate_compliance_report(metrics)

        assert report["overall_status"] == "CRITICAL"
        assert "fairness" in report["standards"]
        assert "robustness" in report["standards"]

        fairness = report["standards"]["fairness"]
        assert fairness["value"] == 0.55
        assert fairness["status"] == "critical"
        assert len(fairness["mitigations"]) > 0

    def test_generate_compliance_report_partial_metrics(self, memory: ISOSemanticMemory):
        """Test rapport avec métriques partielles."""
        metrics = {
            "demographic_parity_ratio": 0.85,
        }
        report = memory.generate_compliance_report(metrics)

        assert "fairness" in report["standards"]
        assert "robustness" not in report["standards"]
        assert report["overall_status"] == "COMPLIANT"

    def test_generate_compliance_report_empty_metrics(self, memory: ISOSemanticMemory):
        """Test rapport avec métriques vides."""
        metrics = {}
        report = memory.generate_compliance_report(metrics)

        assert report["standards"] == {}
        assert report["overall_status"] == "COMPLIANT"

    def test_add_domain_evaluation_via_report(self, memory: ISOSemanticMemory):
        """Test _add_domain_evaluation via generate_compliance_report."""
        metrics = {
            "demographic_parity_ratio": 0.72,
        }
        report = memory.generate_compliance_report(metrics)

        # Verify _add_domain_evaluation was called correctly
        assert "fairness" in report["standards"]
        fairness = report["standards"]["fairness"]
        assert fairness["value"] == 0.72
        assert fairness["status"] == "caution"
        assert isinstance(fairness["mitigations"], list)

    def test_mitigation_strategy_structure(self, memory: ISOSemanticMemory):
        """Test structure des stratégies de mitigation."""
        mitigations = memory.get_mitigations("fairness", ComplianceStatus.CRITICAL)
        assert len(mitigations) > 0

        for m in mitigations:
            assert isinstance(m, MitigationStrategy)
            assert isinstance(m.name, str)
            assert m.phase in ("pre-processing", "in-processing", "post-processing")
            assert isinstance(m.description, str)
            assert m.effectiveness in ("low", "medium", "high")
            assert isinstance(m.implementation, str)
