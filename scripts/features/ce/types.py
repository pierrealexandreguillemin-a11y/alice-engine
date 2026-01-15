"""Module: types.py - Types pour le module CE (Composition Equipes).

Extrait pour éviter les imports circulaires (ISO 42010).

ISO Compliance:
- ISO/IEC 5055:2021 - Code Quality
- ISO/IEC 42010 - Architecture (no circular imports)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TeamTransferability:
    """État de transférabilité d'une équipe."""

    equipe: str
    saison: int
    ronde: int
    scenario: str
    can_donate: bool  # Peut céder des joueurs
    can_receive: bool  # Peut recevoir des renforts
    priority: int  # Priorité (1 = max, 5 = min)
    reason: str  # Justification
