"""Module: scripts/robustness/adversarial_tests.py - Thin re-export.

Document ID: ALICE-MOD-ROBUST-001
Version: 2.0.0

DEPRECATED: Ce module est conserve pour compatibilite ascendante.
Importer directement depuis scripts.robustness.

Le module original a ete refactore en modules SRP (ISO 5055):
- types.py: RobustnessLevel, RobustnessMetrics, RobustnessReport
- thresholds.py: RobustnessThresholds, DEFAULT_THRESHOLDS
- perturbations.py: Tests de perturbation
- metrics.py: compute_robustness_metrics
- report.py: generate_robustness_report

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <50 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from scripts.robustness.metrics import compute_robustness_metrics
from scripts.robustness.perturbations import (
    run_extreme_values_test,
    run_feature_perturbation_test,
    run_noise_test,
    run_out_of_distribution_test,
)
from scripts.robustness.report import generate_robustness_report
from scripts.robustness.thresholds import DEFAULT_THRESHOLDS, RobustnessThresholds
from scripts.robustness.types import RobustnessLevel, RobustnessMetrics, RobustnessReport

__all__ = [
    "RobustnessLevel",
    "RobustnessThresholds",
    "RobustnessMetrics",
    "RobustnessReport",
    "DEFAULT_THRESHOLDS",
    "compute_robustness_metrics",
    "run_noise_test",
    "run_feature_perturbation_test",
    "run_out_of_distribution_test",
    "run_extreme_values_test",
    "generate_robustness_report",
]
