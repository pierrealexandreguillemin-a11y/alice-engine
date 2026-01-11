"""Module: scripts/fairness/bias_detection.py - Thin re-export.

Document ID: ALICE-MOD-BIAS-001
Version: 2.0.0

DEPRECATED: Ce module est conserve pour compatibilite ascendante.
Importer directement depuis scripts.fairness.

Le module original a ete refactore en modules SRP (ISO 5055):
- types.py: BiasLevel, BiasMetrics, BiasReport
- thresholds.py: BiasThresholds, DEFAULT_THRESHOLDS
- metrics.py: compute_bias_metrics_by_group, compute_bias_by_elo_range
- checks.py: check_bias_thresholds
- report.py: generate_fairness_report

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <50 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.fairness.checks import check_bias_thresholds
from scripts.fairness.metrics import compute_bias_by_elo_range, compute_bias_metrics_by_group
from scripts.fairness.report import generate_fairness_report
from scripts.fairness.thresholds import DEFAULT_THRESHOLDS, BiasThresholds
from scripts.fairness.types import BiasLevel, BiasMetrics, BiasReport

__all__ = [
    "BiasLevel",
    "BiasThresholds",
    "BiasMetrics",
    "BiasReport",
    "DEFAULT_THRESHOLDS",
    "compute_bias_metrics_by_group",
    "compute_bias_by_elo_range",
    "check_bias_thresholds",
    "generate_fairness_report",
]
