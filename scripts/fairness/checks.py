"""Verification des seuils de biais - ISO 24027.

Ce module contient les fonctions de verification:
- check_bias_thresholds: Verification des seuils et alertes

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- ISO/IEC 5055:2021 - Code Quality (<100 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from scripts.fairness.thresholds import DEFAULT_THRESHOLDS, BiasThresholds
from scripts.fairness.types import BiasLevel, BiasMetrics


def check_bias_thresholds(
    metrics: list[BiasMetrics],
    thresholds: BiasThresholds | None = None,
) -> tuple[BiasLevel, list[str]]:
    """Verifie les seuils de biais et genere des alertes.

    Args:
    ----
        metrics: Liste des metriques par groupe
        thresholds: Seuils de biais

    Returns:
    -------
        Tuple (niveau_global, liste_alertes)

    ISO 24027: Detection des depassements de seuils.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    alerts = []
    worst_level = BiasLevel.ACCEPTABLE

    for m in metrics:
        group_alerts = []

        # Verifier SPD
        worst_level = _check_spd(m, thresholds, group_alerts, worst_level)

        # Verifier DIR (regle EEOC 4/5)
        worst_level = _check_dir(m, thresholds, group_alerts, worst_level)

        # Verifier EOD
        worst_level = _check_eod(m, thresholds, group_alerts, worst_level)

        if group_alerts:
            alerts.append(f"Groupe '{m.group_name}': {', '.join(group_alerts)}")

    return worst_level, alerts


def _check_spd(
    m: BiasMetrics,
    thresholds: BiasThresholds,
    group_alerts: list[str],
    worst_level: BiasLevel,
) -> BiasLevel:
    """Verifie le seuil SPD."""
    if abs(m.spd) >= thresholds.spd_critical:
        group_alerts.append(f"SPD critique ({m.spd:.3f})")
        return BiasLevel.CRITICAL
    if abs(m.spd) >= thresholds.spd_warning:
        group_alerts.append(f"SPD warning ({m.spd:.3f})")
        if worst_level != BiasLevel.CRITICAL:
            return BiasLevel.WARNING
    return worst_level


def _check_dir(
    m: BiasMetrics,
    thresholds: BiasThresholds,
    group_alerts: list[str],
    worst_level: BiasLevel,
) -> BiasLevel:
    """Verifie le seuil DIR (regle EEOC 4/5)."""
    if m.dir < thresholds.dir_min or m.dir > thresholds.dir_max:
        if m.dir < 0.6 or m.dir > 1.5:
            group_alerts.append(f"DIR critique ({m.dir:.3f})")
            return BiasLevel.CRITICAL
        group_alerts.append(f"DIR warning ({m.dir:.3f})")
        if worst_level != BiasLevel.CRITICAL:
            return BiasLevel.WARNING
    return worst_level


def _check_eod(
    m: BiasMetrics,
    thresholds: BiasThresholds,
    group_alerts: list[str],
    worst_level: BiasLevel,
) -> BiasLevel:
    """Verifie le seuil EOD."""
    if abs(m.eod) >= thresholds.eod_critical:
        group_alerts.append(f"EOD critique ({m.eod:.3f})")
        return BiasLevel.CRITICAL
    if abs(m.eod) >= thresholds.eod_warning:
        group_alerts.append(f"EOD warning ({m.eod:.3f})")
        if worst_level != BiasLevel.CRITICAL:
            return BiasLevel.WARNING
    return worst_level
