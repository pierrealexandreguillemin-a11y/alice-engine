"""Seuils de robustesse - ISO 24029.

Ce module contient les seuils configurables pour les tests de robustesse.

Seuils par défaut (ISO 24029-2 + littérature académique):
- Dégradation acceptable: < 3%
- Warning: 3-5%
- Critique: >= 10%
- Stabilité minimum: 95%

ISO Compliance:
- ISO/IEC 24029-2:2023 - Robustness Testing Methodology
- ISO/IEC 5055:2021 - Code Quality (<50 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobustnessThresholds:
    """Seuils de robustesse ISO 24029.

    Sources:
    - ISO 24029-2 recommandations
    - Benchmarks industrie (5% dégradation acceptable)
    - Publications académiques sur adversarial robustness
    """

    # Dégradation de performance acceptable
    degradation_acceptable: float = 0.03  # 3%
    degradation_warning: float = 0.05  # 5%
    degradation_critical: float = 0.10  # 10%

    # Stabilité des prédictions
    stability_threshold: float = 0.95  # 95% de prédictions stables

    # Niveaux de bruit pour tests
    noise_levels: tuple[float, ...] = (0.01, 0.05, 0.10)

    # Perturbation features
    perturbation_levels: tuple[float, ...] = (0.05, 0.10, 0.20)


# Seuils par défaut (ISO 24029)
DEFAULT_THRESHOLDS = RobustnessThresholds()
