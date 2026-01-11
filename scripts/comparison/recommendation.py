"""Generation de recommandations - ISO 42001.

Ce module contient les fonctions de generation de recommandations.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System
- ISO/IEC 5055:2021 - Code Quality (<60 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-11
"""

from __future__ import annotations

from scripts.comparison.mcnemar_test import McNemarResult


def generate_recommendation(
    winner: str,
    mcnemar: McNemarResult,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    model_a_name: str,
    model_b_name: str,
    practical_significance: bool,
) -> str:
    """Genere une recommandation basee sur l'analyse.

    ISO 42001: Decision documentee et justifiee.
    """
    if winner == "tie":
        return _recommend_tie(
            mcnemar, metrics_a, metrics_b, model_a_name, model_b_name, practical_significance
        )

    if practical_significance:
        return (
            f"{winner} est significativement meilleur (p={mcnemar.p_value:.3f}) "
            f"avec une difference pratiquement significative. "
            f"Recommandation: deployer {winner}."
        )

    return (
        f"{winner} est statistiquement meilleur (p={mcnemar.p_value:.3f}) "
        f"mais la difference pratique est faible. "
        f"Considerer les couts operationnels avant decision."
    )


def _recommend_tie(
    mcnemar: McNemarResult,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    model_a_name: str,
    model_b_name: str,
    practical_significance: bool,
) -> str:
    """Recommandation en cas d'egalite statistique."""
    if practical_significance:
        better = model_a_name if metrics_a["accuracy"] > metrics_b["accuracy"] else model_b_name
        return (
            f"Pas de difference statistiquement significative (p={mcnemar.p_value:.3f}), "
            f"mais {better} montre une tendance. Considerer d'autres facteurs "
            f"(inference time, interpretabilite)."
        )
    return (
        f"Pas de difference significative entre les modeles (p={mcnemar.p_value:.3f}). "
        f"Choisir selon criteres operationnels (vitesse, maintenance)."
    )


# Alias pour compatibilite (prefixe _ historique)
_generate_recommendation = generate_recommendation
