"""Backtest quality metrics : T13 + T13b + T14 (Top-K recall + Accuracy@K + Jaccard max).

Plan 3 V2 Phase 3. ISO 25059 (quality gates), ISO 29119 (testability).

Metrics fournis
---------------
- T13 Top-K recall : fraction observed players captured in union of N scenarios
  (spec Phase 3 §6.2). Gate seuil >= 0.90.
- T13b Accuracy@K : fraction observed captured in top-weighted scenario (sports
  lineup prediction SOTA). Gate seuil >= 0.75.
- T14 Jaccard max : maximum Jaccard similarity across scenarios vs observed
  (spec Phase 3 §6.2). Gate seuil >= 0.75.

Cohérence identifiants
----------------------
Ground truth (`ObservedPlayer.joueur_nom`) provient de `echiquiers.parquet`
colonne `blanc_nom` / `noir_nom` au format "NOM Prenom" (FFE).
Scenarios exposent `PlayerCandidate(nom, prenom)` séparés : on reconstruit la
forme canonique `f"{nom} {prenom}".strip()` pour matcher le format echiquiers.

Sources SOTA
------------
- Pappalardo et al. 2019, "PlayeRank" : accuracy@K ≈ 0.75 pour lineup prediction
  sports collectifs (https://arxiv.org/abs/1902.01957).
- Oztakar & Yilmaz 2020, lineup forecasting : top-K recall >= 0.90 typique.

Document ID: ALICE-BACKTEST-METRICS
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.backtest.ground_truth import ObservedLineup
    from services.ali.scenario import ScenarioSet
    from services.ali.types import PlayerCandidate


def _canonical_name(player: PlayerCandidate) -> str:
    """Canonical player name matching echiquiers.parquet convention.

    `echiquiers.parquet` stocke `blanc_nom = "NOM Prenom"`. On reproduit
    exactement ce format pour permettre l'intersection avec
    `ObservedLineup.player_names()`.
    """
    return f"{player.nom} {player.prenom}".strip()


def top_k_recall(observed: ObservedLineup, scenario_set: ScenarioSet) -> float:
    """T13 : fraction observed players captured in union of N scenarios.

    Formule : ``|observed ∩ (∪_s scenario_s.player_names)| / |observed|``.
    Gate seuil : >= 0.90 (spec Phase 3 §6.2).

    @param observed: lineup réel du club adverse à (saison, ronde).
    @param scenario_set: ScenarioSet ALI prédit (typiquement 20 scenarios).

    @returns recall ∈ [0, 1]. Retourne 0.0 si observed est vide (edge case).
    """
    observed_names = observed.player_names()
    if not observed_names:
        return 0.0
    union_names: set[str] = set()
    for scenario in scenario_set.scenarios:
        for assignment in scenario.lineup.assignments:
            union_names.add(_canonical_name(assignment.player))
    intersection = observed_names & union_names
    return len(intersection) / len(observed_names)


def accuracy_at_k(
    observed: ObservedLineup,
    scenario_set: ScenarioSet,
    k: int | None = None,
) -> float:
    """T13b : fraction observed captured in top-weighted scenario (sports SOTA).

    Formule : ``|observed ∩ top_scenario.player_names| / k`` où
    ``top = argmax_s scenario_s.weight`` et ``k = team_size`` par défaut.
    Gate seuil : >= 0.75 (Pappalardo 2019 sports lineup prediction standard).

    @param observed: lineup réel du club adverse.
    @param scenario_set: ScenarioSet ALI prédit.
    @param k: override taille d'équipe; default = len(observed.players).

    @returns accuracy ∈ [0, 1]. Retourne 0.0 si scenario_set vide ou k_eff = 0.
    """
    if not scenario_set.scenarios:
        return 0.0
    top = max(scenario_set.scenarios, key=lambda s: s.weight)
    top_names = {_canonical_name(a.player) for a in top.lineup.assignments}
    observed_names = observed.player_names()
    k_eff = k if k is not None else len(observed.players)
    if k_eff == 0:
        return 0.0
    return len(observed_names & top_names) / k_eff


def jaccard_max(observed: ObservedLineup, scenario_set: ScenarioSet) -> float:
    """T14 : maximum Jaccard similarity across scenarios vs observed lineup.

    Formule : ``max_s |observed ∩ s_lineup| / |observed ∪ s_lineup|``.
    Gate seuil : >= 0.75 (spec Phase 3 §6.2). Au moins 1 scenario proche de
    la réalité observée garantit que le pipeline couvre le "vrai" lineup.

    @param observed: lineup réel du club adverse à (saison, ronde).
    @param scenario_set: ScenarioSet ALI prédit.

    @returns Jaccard score ∈ [0, 1]. Retourne 0.0 si observed vide ou aucun
             scenario ne partage de joueur (union vide retournerait division
             par zéro sinon).
    """
    observed_names = observed.player_names()
    if not observed_names:
        return 0.0
    best = 0.0
    for scenario in scenario_set.scenarios:
        s_names = {_canonical_name(a.player) for a in scenario.lineup.assignments}
        union = observed_names | s_names
        if not union:
            continue
        intersection = observed_names & s_names
        jac = len(intersection) / len(union)
        best = max(best, jac)
    return best
