"""Configuration pytest - Optimisation mémoire.

Ce module configure pytest pour économiser la RAM:
- Garbage collection après chaque test
- Fixtures légères
- Pas de chargement des modèles ML

ISO Compliance:
- ISO/IEC 29119 - Software Testing
- ISO/IEC 23894:2023 - AI Risk Management (resource management)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

import gc

import pytest

# Load fixtures from specialized conftest modules
pytest_plugins = [
    "tests.conftest_robustness",
    "tests.conftest_drift",
    "tests.conftest_input",
    "tests.conftest_bias",
    "tests.conftest_ml",
]


@pytest.fixture(autouse=True)
def cleanup_after_test() -> None:
    """Force garbage collection après chaque test."""
    yield
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Configure l'environnement de test une seule fois."""
    # Désactiver les warnings verbeux des libs ML
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    yield

    # Cleanup final
    gc.collect()
