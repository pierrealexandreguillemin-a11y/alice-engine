"""Package: scripts/robustness - Robustness Testing.

Document ID: ALICE-PKG-ROBUST-001
Version: 1.0.0

Ce package implémente les tests de robustesse pour les modèles ML ALICE,
conformément à ISO/IEC 24029:2021/2023.

Modules:
- adversarial_tests.py: Tests adversariaux et perturbations
  - Bruit gaussien (noise injection)
  - Perturbation de features
  - Out-of-Distribution (OOD) detection
  - Valeurs extrêmes (stress testing)

Exports:
- RobustnessLevel: Enum (ROBUST, ACCEPTABLE, WARNING, FRAGILE)
- RobustnessMetrics: Dataclass métriques par test
- RobustnessReport: Rapport complet de robustesse
- RobustnessThresholds: Seuils configurables
- compute_robustness_metrics(): Suite complète de tests
- run_noise_test(): Test bruit gaussien
- run_feature_perturbation_test(): Test perturbation features
- run_out_of_distribution_test(): Test OOD
- run_extreme_values_test(): Test valeurs extrêmes
- generate_robustness_report(): Génération rapport ISO

Seuils par défaut (ISO 24029-2 + littérature):
- Dégradation acceptable: < 3%
- Warning: 3-5%
- Critique: > 10%
- Stabilité minimum: 95%

ISO Compliance:
- ISO/IEC 24029-1:2021 - Neural Network Robustness Assessment
- ISO/IEC 24029-2:2023 - Robustness Testing Methodology
- ISO/IEC 42001:2023 - AI Management System (traçabilité)
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 5055:2021 - Code Quality

See Also
--------
- docs/iso/AI_RISK_ASSESSMENT.md - Section R2: Model Performance Risks
- docs/iso/STATEMENT_OF_APPLICABILITY.md - Control B.4.5
- config/hyperparameters.yaml - metrics_thresholds

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""

from scripts.robustness.adversarial_tests import (
    RobustnessLevel,
    RobustnessMetrics,
    RobustnessReport,
    RobustnessThresholds,
    compute_robustness_metrics,
    generate_robustness_report,
    run_extreme_values_test,
    run_feature_perturbation_test,
    run_noise_test,
    run_out_of_distribution_test,
)

__all__ = [
    # Enums
    "RobustnessLevel",
    # Dataclasses
    "RobustnessMetrics",
    "RobustnessReport",
    "RobustnessThresholds",
    # Functions - Suite complète
    "compute_robustness_metrics",
    "generate_robustness_report",
    # Functions - Tests individuels
    "run_noise_test",
    "run_feature_perturbation_test",
    "run_out_of_distribution_test",
    "run_extreme_values_test",
]
