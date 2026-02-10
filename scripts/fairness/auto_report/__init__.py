"""Fairness Auto Report Module - ISO 24027 + NIST AI 100-1.

Generation automatique de rapport fairness multi-attributs.

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation
- EU AI Act Art.13 - Transparency

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.fairness.auto_report.formatter import format_markdown_report
from scripts.fairness.auto_report.generator import generate_comprehensive_report
from scripts.fairness.auto_report.types import (
    AttributeAnalysis,
    ComprehensiveFairnessReport,
    GroupMetrics,
)

__all__ = [
    "AttributeAnalysis",
    "ComprehensiveFairnessReport",
    "GroupMetrics",
    "format_markdown_report",
    "generate_comprehensive_report",
]
