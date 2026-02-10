"""Formateur Markdown du rapport fairness - ISO 24027.

Fonctions:
- format_markdown_report: formatage complet en Markdown
- _format_summary: section executive summary
- _format_attribute_table: tableau aggrege par attribut
- _format_group_details: tableau disaggrege par groupe (EU AI Act)

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- EU AI Act Art.13 - Transparency (disaggregated metrics)
- HuggingFace Model Card format
- ISO/IEC 5055:2021 - Code Quality (<180 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-10
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.fairness.auto_report.types import (
        AttributeAnalysis,
        ComprehensiveFairnessReport,
    )


def format_markdown_report(
    report: ComprehensiveFairnessReport,
    output_path: Path | None = None,
) -> str:
    """Formate le rapport fairness en Markdown.

    Args:
    ----
        report: Rapport complet a formater
        output_path: Fichier de sortie (optionnel)

    Returns:
    -------
        Contenu Markdown du rapport
    """
    sections = [
        _format_header(),
        _format_summary(report),
        _format_analyses(report),
        _format_recommendations(report),
        _format_iso_checklist(report),
    ]
    md = "\n\n".join(sections)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")

    return md


def _format_header() -> str:
    """Formate l'en-tete du rapport."""
    return "# Comprehensive Fairness Report\n\n_ISO 24027 + NIST AI 100-1_"


def _format_summary(report: ComprehensiveFairnessReport) -> str:
    """Formate la section Executive Summary."""
    status_emoji = {"fair": "PASS", "caution": "WARNING", "critical": "FAIL"}
    status_label = status_emoji.get(report.overall_status, report.overall_status)
    return (
        f"## Executive Summary\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Model | {report.model_name} |\n"
        f"| Version | {report.model_version} |\n"
        f"| Date | {report.timestamp} |\n"
        f"| Samples | {report.total_samples:,} |\n"
        f"| Attributes Analyzed | {len(report.analyses)} |\n"
        f"| Overall Status | **{status_label}** |"
    )


def _format_analyses(report: ComprehensiveFairnessReport) -> str:
    """Formate les analyses par attribut."""
    if not report.analyses:
        return "## Attribute Analyses\n\nNo attributes analyzed."
    parts = ["## Attribute Analyses"]
    for analysis in report.analyses:
        parts.append(f"\n### {analysis.attribute_name}")
        parts.append(_format_attribute_table(analysis))
        if analysis.confidence_intervals:
            parts.append(_format_ci(analysis))
        if analysis.group_details:
            parts.append(_format_group_details(analysis))
    return "\n".join(parts)


def _format_attribute_table(analysis: AttributeAnalysis) -> str:
    """Formate le tableau de metriques aggregees pour un attribut."""
    return (
        f"\n| Metric | Value | Threshold | Status |\n"
        f"|--------|-------|-----------|--------|\n"
        f"| Demographic Parity Ratio | {analysis.demographic_parity_ratio:.4f} "
        f"| >= 0.80 | {_metric_status(analysis.demographic_parity_ratio, 0.80)} |\n"
        f"| Equalized Odds TPR Diff | {analysis.equalized_odds_tpr_diff:.4f} "
        f"| <= 0.10 | {_diff_status(analysis.equalized_odds_tpr_diff, 0.10)} |\n"
        f"| Equalized Odds FPR Diff | {analysis.equalized_odds_fpr_diff:.4f} "
        f"| <= 0.10 | {_diff_status(analysis.equalized_odds_fpr_diff, 0.10)} |\n"
        f"| Predictive Parity Diff | {analysis.predictive_parity_diff:.4f} "
        f"| <= 0.10 | {_diff_status(analysis.predictive_parity_diff, 0.10)} |\n"
        f"| Min Group Accuracy | {analysis.min_group_accuracy:.4f} "
        f"| >= 0.60 | {_metric_status(analysis.min_group_accuracy, 0.60)} |\n"
        f"| Max Calibration Gap | {analysis.max_calibration_gap:.4f} "
        f"| <= 0.10 | {_diff_status(analysis.max_calibration_gap, 0.10)} |\n"
        f"| Groups | {analysis.group_count} | - | - |\n"
        f"| Samples | {analysis.sample_count:,} | - | - |\n"
        f"| **Overall** | **{analysis.status}** | - | - |"
    )


def _format_ci(analysis: AttributeAnalysis) -> str:
    """Formate les intervalles de confiance (NIST AI 100-1)."""
    lines = ["\n**95% Bootstrap Confidence Intervals:**"]
    for metric, bounds in analysis.confidence_intervals.items():
        label = metric.replace("_", " ").title()
        lines.append(f"- {label}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
    return "\n".join(lines)


def _format_group_details(analysis: AttributeAnalysis) -> str:
    """Formate le tableau disaggrege par groupe (EU AI Act Art.13)."""
    lines = [
        "\n**Disaggregated Metrics by Group:**\n",
        "| Group | N | Pos Rate | TPR | FPR | Precision | Accuracy | Cal Gap |",
        "|-------|---|----------|-----|-----|-----------|----------|---------|",
    ]
    for g in analysis.group_details:
        lines.append(
            f"| {g.group_name} | {g.sample_count} "
            f"| {g.positive_rate:.3f} | {g.tpr:.3f} "
            f"| {g.fpr:.3f} | {g.precision:.3f} | {g.accuracy:.3f} "
            f"| {g.calibration_gap:.3f} |"
        )
    return "\n".join(lines)


def _format_recommendations(report: ComprehensiveFairnessReport) -> str:
    """Formate la section recommandations."""
    lines = ["## Recommendations"]
    for rec in report.recommendations:
        lines.append(f"\n- {rec}")
    return "\n".join(lines)


def _format_iso_checklist(report: ComprehensiveFairnessReport) -> str:
    """Formate la checklist ISO compliance."""
    lines = ["## ISO Compliance Checklist"]
    for standard, compliant in report.iso_compliance.items():
        check = "x" if compliant else " "
        lines.append(f"\n- [{check}] {standard}")
    return "\n".join(lines)


def _metric_status(value: float, threshold: float) -> str:
    """Status pour une metrique (higher is better)."""
    return "PASS" if value >= threshold else "FAIL"


def _diff_status(value: float, threshold: float) -> str:
    """Status pour une difference (lower is better)."""
    return "PASS" if value <= threshold else "FAIL"
