"""Tests robustesse - ISO 24029 (Re-export).

Document ID: ALICE-TEST-ROBUST-001
Version: 2.0.0

DEPRECATED: Ce fichier est conservé pour compatibilité.
Les tests ont été refactorisés en modules SRP (ISO 5055):

- test_robustness_types.py: Tests types et enums (4 tests)
- test_robustness_perturbations.py: Tests perturbations (10 tests)
- test_robustness_report.py: Tests rapport (7 tests)
- test_robustness_edge_cases.py: Tests cas limites (7 tests)
- conftest_robustness.py: Fixtures partagées

Total: 28 tests (inchangé)

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality (SRP, <50 lignes)
- ISO/IEC 29119:2022 - Software Testing

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

# Re-export all test classes for backwards compatibility
from tests.test_robustness_edge_cases import TestEdgeCases, TestIntegration
from tests.test_robustness_perturbations import (
    TestExtremeValues,
    TestFeaturePerturbation,
    TestInputNoise,
    TestOutOfDistribution,
)
from tests.test_robustness_report import (
    TestComputeRobustnessMetrics,
    TestGenerateRobustnessReport,
)
from tests.test_robustness_types import (
    TestRobustnessLevel,
    TestRobustnessMetrics,
    TestRobustnessThresholds,
)

__all__ = [
    "TestRobustnessThresholds",
    "TestRobustnessMetrics",
    "TestRobustnessLevel",
    "TestInputNoise",
    "TestFeaturePerturbation",
    "TestOutOfDistribution",
    "TestExtremeValues",
    "TestComputeRobustnessMetrics",
    "TestGenerateRobustnessReport",
    "TestEdgeCases",
    "TestIntegration",
]
