"""Package: scripts/robustness - Robustness Testing.

Document ID: ALICE-PKG-ROBUST-001
Version: 2.0.0

Ce package implemente les tests de robustesse pour les modeles ML ALICE,
conformement a ISO/IEC 24029:2021/2023.

Modules (refactored ISO 5055):
- types.py: RobustnessLevel, RobustnessMetrics, RobustnessReport
- thresholds.py: RobustnessThresholds, DEFAULT_THRESHOLDS
- perturbations.py: Tests de perturbation (noise, features, OOD, extreme)
- metrics.py: compute_robustness_metrics
- report.py: generate_robustness_report
- adversarial_tests.py: Thin re-export (backwards compatibility)

Exports:
- RobustnessLevel: Enum (ROBUST, ACCEPTABLE, WARNING, FRAGILE)
- RobustnessMetrics: Dataclass metriques par test
- RobustnessReport: Rapport complet de robustesse
- RobustnessThresholds: Seuils configurables
- DEFAULT_THRESHOLDS: Seuils par defaut
- compute_robustness_metrics(): Suite complete de tests
- run_noise_test(): Test bruit gaussien
- run_feature_perturbation_test(): Test perturbation features
- run_out_of_distribution_test(): Test OOD
- run_extreme_values_test(): Test valeurs extremes
- generate_robustness_report(): Generation rapport ISO

Seuils par defaut (ISO 24029-2 + litterature):
- Degradation acceptable: < 3%
- Warning: 3-5%
- Critique: > 10%
- Stabilite minimum: 95%

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment
- ISO/IEC 24029-2:2023 - Robustness Testing Methodology
- ISO/IEC 42001:2023 - AI Management System (tracabilite)
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 5055:2021 - Code Quality (<300 lines per module)

See Also
--------
- docs/iso/AI_RISK_ASSESSMENT.md - Section R2: Model Performance Risks
- docs/iso/STATEMENT_OF_APPLICABILITY.md - Control B.4.5
- config/hyperparameters.yaml - metrics_thresholds

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
    # Enums
    "RobustnessLevel",
    # Dataclasses
    "RobustnessMetrics",
    "RobustnessReport",
    "RobustnessThresholds",
    # Constants
    "DEFAULT_THRESHOLDS",
    # Functions - Suite complete
    "compute_robustness_metrics",
    "generate_robustness_report",
    # Functions - Tests individuels
    "run_noise_test",
    "run_feature_perturbation_test",
    "run_out_of_distribution_test",
    "run_extreme_values_test",
]
