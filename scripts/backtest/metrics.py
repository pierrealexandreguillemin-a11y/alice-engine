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
  **Note SOTA** : Pappalardo 2019 (sports lineup prediction) utilise
  accuracy@K (T13b) comme primary metric, pas Jaccard. Jaccard max peut
  sur-évaluer si MC couvre la queue mais rate le mode. Décision Plan 3 :
  garder Jaccard (spec §6.2) ET accuracy@K (T13b) complémentaires, à
  documenter explicitement dans ALI_MODEL_CARD.md §Limitations.
- T15 Brier score P(presence) : calibration per-player presence probability
  (spec Phase 3 §6.2). Gate seuil <= 0.20.
- T6 Brier skill score : BSS = 1 - (Brier_model / Brier_baseline) vs baseline
  Elo (Pappalardo 2019 sports SOTA). Gate seuil >= 0.05.

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
    # Use deduplicated set size for denominator coherence with numerator
    # (M3 audit JALON #1). Caller can override via k=team_size si besoin.
    k_eff = k if k is not None else len(observed_names)
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


def brier_presence(observed: ObservedLineup, scenario_set: ScenarioSet) -> float:
    """T15 : Brier score on per-player presence probability.

    Pour chaque player de la union (observed ∪ scenarios) :
    ``p_presence_j = Σ_s scenario_s.weight × 1[j ∈ s_lineup]``
    ``observed_flag_j = 1 si j ∈ observed sinon 0``
    ``T15 = mean((p_presence - observed_flag)²)``

    Gate seuil : <= 0.20 (spec Phase 3 §6.2, baseline Elo ~0.26).

    @param observed: lineup réel du club adverse.
    @param scenario_set: ScenarioSet ALI prédit (weights normalisés).

    @returns Brier score ∈ [0, 1]. 0 = prédiction parfaite.
    """
    observed_names = observed.player_names()
    p_presence: dict[str, float] = {}
    for scenario in scenario_set.scenarios:
        w = scenario.weight
        for assignment in scenario.lineup.assignments:
            name = _canonical_name(assignment.player)
            p_presence[name] = p_presence.get(name, 0.0) + w

    all_players = set(p_presence) | set(observed_names)
    if not all_players:
        return 0.0
    total = 0.0
    for name in all_players:
        p = min(p_presence.get(name, 0.0), 1.0)
        obs_flag = 1.0 if name in observed_names else 0.0
        total += (p - obs_flag) ** 2
    return total / len(all_players)


def brier_skill_score(
    observed: ObservedLineup,
    scenario_set: ScenarioSet,
    baseline_brier: float,
) -> float:
    """T6 : Brier skill score vs baseline (Pappalardo 2019 sports SOTA).

    Formule : ``BSS = 1 - (Brier_model / Brier_baseline)``.
    Interpretation :
    - BSS > 0 : model better than baseline
    - BSS = 0 : model == baseline
    - BSS < 0 : model worse
    - BSS = 1 : perfect model (Brier_model = 0)

    Gate seuil : >= 0.05 (model improves by >=5% over baseline).

    @param observed: lineup réel.
    @param scenario_set: ScenarioSet ALI prédit.
    @param baseline_brier: Brier score du baseline (ex. Elo-only single scenario).

    @returns BSS (unbounded below, max 1.0). Retourne 0.0 si baseline <= 0
             (évite division par zéro, fallback conservateur).
    """
    if baseline_brier <= 0:
        return 0.0
    model_brier = brier_presence(observed, scenario_set)
    return 1.0 - (model_brier / baseline_brier)
