"""Seuils de biais - ISO 24027.

Ce module contient les seuils configurables pour la detection de biais.

Seuils par defaut (ISO 24027 + EEOC + Fairlearn):
- SPD: |SPD| < 0.1 (acceptable), >= 0.2 (critique)
- DIR: 0.8 <= DIR <= 1.25 (regle des 4/5, EEOC guidelines)
- EOD: |EOD| < 0.1 (acceptable), >= 0.2 (critique)

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<50 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BiasThresholds:
    """Seuils de biais ISO 24027.

    Sources:
    - EEOC 4/5 rule: DIR entre 0.8 et 1.25
    - Fairlearn recommendations: SPD < 0.1
    - Academic consensus: EOD < 0.1
    """

    spd_warning: float = 0.1
    spd_critical: float = 0.2
    dir_min: float = 0.8  # EEOC 4/5 rule
    dir_max: float = 1.25
    eod_warning: float = 0.1
    eod_critical: float = 0.2


# Seuils par defaut (ISO 24027 + EEOC + Fairlearn)
DEFAULT_THRESHOLDS = BiasThresholds()
