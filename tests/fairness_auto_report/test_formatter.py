"""Tests Fairness Report Formatter - ISO 29119.

Document ID: ALICE-TEST-FAIRNESS-AUTO-FORMATTER
Version: 1.1.0
Tests count: 13

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC TR 24027:2021 - Bias in AI systems

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from pathlib import Path

from scripts.fairness.auto_report.formatter import (
    _format_attribute_table,
    _format_ci,
    _format_group_details,
    _format_summary,
    format_markdown_report,
)
from scripts.fairness.auto_report.types import (
    AttributeAnalysis,
    ComprehensiveFairnessReport,
    GroupMetrics,
)


def _make_report(
    analyses: list[AttributeAnalysis],
    status: str = "fair",
) -> ComprehensiveFairnessReport:
    """Helper: cree un rapport pour les tests."""
    return ComprehensiveFairnessReport(
        model_name="TestModel",
        model_version="v1.0",
        timestamp="2026-02-10T12:00:00",
        total_samples=500,
        analyses=analyses,
        overall_status=status,
        recommendations=["Continue monitoring."],
        iso_compliance={"iso_24027": True, "nist_ai_100_1": True},
    )


class TestFormatMarkdownReport:
    """Tests pour le formatage Markdown."""

    def test_contains_executive_summary(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le rapport contient un executive summary."""
        report = _make_report([mock_analysis_fair])
        md = format_markdown_report(report)
        assert "Executive Summary" in md or "Summary" in md

    def test_contains_metrics_tables(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le rapport contient des tableaux de metriques."""
        report = _make_report([mock_analysis_fair])
        md = format_markdown_report(report)
        assert "Demographic Parity" in md or "demographic_parity" in md

    def test_contains_recommendations_section(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le rapport contient une section recommandations."""
        report = _make_report([mock_analysis_fair])
        md = format_markdown_report(report)
        assert "Recommendation" in md or "recommendation" in md

    def test_contains_iso_checklist(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le rapport contient une checklist ISO."""
        report = _make_report([mock_analysis_fair])
        md = format_markdown_report(report)
        assert "ISO" in md

    def test_writes_to_file(
        self,
        mock_analysis_fair: AttributeAnalysis,
        tmp_path: Path,
    ) -> None:
        """Le rapport peut etre ecrit dans un fichier."""
        report = _make_report([mock_analysis_fair])
        output_path = tmp_path / "fairness_report.md"
        format_markdown_report(report, output_path=output_path)
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "TestModel" in content


class TestFormatAttributeTable:
    """Tests pour le formatage d'un tableau d'attribut."""

    def test_markdown_table_format(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le tableau est au format Markdown."""
        table = _format_attribute_table(mock_analysis_fair)
        assert "|" in table

    def test_includes_all_metrics(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le tableau inclut toutes les metriques."""
        table = _format_attribute_table(mock_analysis_fair)
        assert "Demographic Parity" in table
        assert "Predictive Parity" in table

    def test_status_column_present(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le tableau inclut la colonne status."""
        table = _format_attribute_table(mock_analysis_fair)
        assert "fair" in table.lower()


class TestFormatGroupDetails:
    """Tests pour le tableau disaggrege par groupe."""

    def test_shows_per_group_table(self) -> None:
        """Affiche un tableau par groupe (EU AI Act Art.13)."""
        analysis = AttributeAnalysis(
            attribute_name="ligue_code",
            sample_count=300,
            group_count=3,
            demographic_parity_ratio=0.85,
            equalized_odds_tpr_diff=0.05,
            equalized_odds_fpr_diff=0.04,
            predictive_parity_diff=0.03,
            min_group_accuracy=0.78,
            status="fair",
            group_details=[
                GroupMetrics(
                    group_name="IDF",
                    sample_count=100,
                    positive_rate=0.6,
                    tpr=0.7,
                    fpr=0.3,
                    precision=0.65,
                    accuracy=0.80,
                ),
                GroupMetrics(
                    group_name="ARA",
                    sample_count=100,
                    positive_rate=0.55,
                    tpr=0.65,
                    fpr=0.28,
                    precision=0.62,
                    accuracy=0.78,
                ),
            ],
        )
        table = _format_group_details(analysis)
        assert "IDF" in table
        assert "ARA" in table
        assert "Disaggregated" in table

    def test_group_details_in_full_report(self) -> None:
        """Les group_details apparaissent dans le rapport complet."""
        analysis = AttributeAnalysis(
            attribute_name="test",
            sample_count=100,
            group_count=2,
            demographic_parity_ratio=0.9,
            equalized_odds_tpr_diff=0.05,
            equalized_odds_fpr_diff=0.04,
            predictive_parity_diff=0.03,
            min_group_accuracy=0.80,
            status="fair",
            group_details=[
                GroupMetrics(
                    group_name="A",
                    sample_count=50,
                    positive_rate=0.5,
                    tpr=0.6,
                    fpr=0.2,
                    precision=0.7,
                    accuracy=0.80,
                ),
            ],
        )
        report = _make_report([analysis])
        md = format_markdown_report(report)
        assert "Disaggregated" in md


class TestFormatCI:
    """Tests pour le formatage des intervalles de confiance."""

    def test_shows_ci_section(self) -> None:
        """Affiche la section CI (NIST AI 100-1)."""
        analysis = AttributeAnalysis(
            attribute_name="test",
            sample_count=100,
            group_count=2,
            demographic_parity_ratio=0.9,
            equalized_odds_tpr_diff=0.05,
            equalized_odds_fpr_diff=0.04,
            predictive_parity_diff=0.03,
            min_group_accuracy=0.80,
            status="fair",
            confidence_intervals={
                "demographic_parity_ratio": [0.82, 0.96],
            },
        )
        ci_text = _format_ci(analysis)
        assert "Bootstrap" in ci_text
        assert "0.82" in ci_text


class TestFormatSummary:
    """Tests pour le formatage du summary."""

    def test_includes_model_info(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le summary inclut les infos du modele."""
        report = _make_report([mock_analysis_fair])
        summary = _format_summary(report)
        assert "TestModel" in summary
        assert "v1.0" in summary

    def test_includes_date(
        self,
        mock_analysis_fair: AttributeAnalysis,
    ) -> None:
        """Le summary inclut la date."""
        report = _make_report([mock_analysis_fair])
        summary = _format_summary(report)
        assert "2026" in summary
