"""Calibration metrics T7 ECE + T7b Reliability diagram.

Plan 3 V2 Phase 3. ISO 25059 (calibration quality), 29119 (testability).

Metrics fournis
---------------
- T7 ECE 10-bins (Guo 2017) : Expected Calibration Error sur P(presence).
  ``ECE = Σ_m (|B_m| / n) × |conf(B_m) - acc(B_m)|`` où
  ``B_m`` = bin m, ``conf`` = moyenne p_presence prédite, ``acc`` = fréquence
  observée. Gate seuil <= 0.10 (Guo 2017, well-calibrated model).

- T7b Reliability diagram : points (mean_p, mean_obs, n) par bin, utilisables
  pour tracer calibration curve (Bröcker 2008). Export markdown/ASCII dans
  backtest report pour ALI_MODEL_CARD.md §Calibration.

Protocole ECE
-------------
1. Pour chaque joueur de la union (predicted ∪ observed), calculer
   ``p_presence = Σ_s weight_s × 1[joueur ∈ scenario_s]`` et ``obs_flag``.
2. Binner items par ``p_presence`` dans ``n_bins`` bins equal-width [0, 1].
3. Par bin m : ``conf_m = mean(p_presence)``, ``acc_m = mean(obs_flag)``.
4. ``ECE = Σ_m (count_m / total) × |conf_m - acc_m|``.

Sources SOTA
------------
- Guo et al. 2017, "On Calibration of Modern Neural Networks"
  (https://arxiv.org/abs/1706.04599) - ECE équi-bins, threshold <= 0.10.
- Bröcker & Smith 2007, "Increasing the Reliability of Reliability Diagrams"
  (Weather & Forecasting 22) - reliability diagram methodology.
- Naeini et al. 2015 "Obtaining Well-Calibrated Probabilities Using
  Bayesian Binning" - justification equal-width vs equal-mass.

Document ID: ALICE-BACKTEST-CALIBRATION
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.backtest.ground_truth import ObservedLineup
    from services.ali.scenario import ScenarioSet
    from services.ali.types import PlayerCandidate


def _canonical_name(player: PlayerCandidate) -> str:
    """Match echiquiers.parquet 'NOM Prenom' convention (dupliqué de metrics.py).

    Duplicated pour éviter cycle import ; trivial (1 ligne).
    """
    return f"{player.nom} {player.prenom}".strip()


def _build_presence_pairs(
    observed: ObservedLineup, scenario_set: ScenarioSet
) -> list[tuple[float, float]]:
    """Compute (p_presence, obs_flag) pairs for union of players.

    @returns list of (predicted_prob, observed_binary) ∈ ([0,1], {0,1}).
             Empty si union vide (aucun joueur nulle part).
    """
    observed_names = observed.player_names()
    p_presence: dict[str, float] = {}
    for scenario in scenario_set.scenarios:
        w = scenario.weight
        for assignment in scenario.lineup.assignments:
            name = _canonical_name(assignment.player)
            p_presence[name] = p_presence.get(name, 0.0) + w

    all_players = set(p_presence) | set(observed_names)
    pairs: list[tuple[float, float]] = []
    for name in all_players:
        p = min(p_presence.get(name, 0.0), 1.0)
        obs = 1.0 if name in observed_names else 0.0
        pairs.append((p, obs))
    return pairs


def ece_presence(
    observed: ObservedLineup,
    scenario_set: ScenarioSet,
    n_bins: int = 10,
) -> float:
    """T7 : Expected Calibration Error sur P(presence), 10-bin equal-width.

    Guo et al. 2017 formulation.

    Gate seuil : <= 0.10 (well-calibrated).

    @param observed: lineup réel du club adverse.
    @param scenario_set: ScenarioSet ALI (weights normalisés).
    @param n_bins: nombre de bins equal-width sur [0, 1]. Default 10.

    @raises ValueError: n_bins < 1.
    @returns ECE ∈ [0, 1]. 0 = calibration parfaite. 0.0 si union vide.
    """
    if n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)
    pairs = _build_presence_pairs(observed, scenario_set)
    if not pairs:
        return 0.0

    total = len(pairs)
    ece = 0.0
    bin_width = 1.0 / n_bins
    for m in range(n_bins):
        lo, hi = m * bin_width, (m + 1) * bin_width
        in_bin = [(p, o) for p, o in pairs if lo <= p < hi or (m == n_bins - 1 and p == hi)]
        if not in_bin:
            continue
        conf = sum(p for p, _ in in_bin) / len(in_bin)
        acc = sum(o for _, o in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(conf - acc)
    return ece


@dataclass(frozen=True)
class ReliabilityPoint:
    """One bin of a reliability diagram (T7b).

    @param bin_low: lower edge of bin ∈ [0, 1].
    @param bin_high: upper edge ∈ [0, 1].
    @param mean_predicted: mean p_presence within bin.
    @param observed_frequency: fraction observed (= 1) within bin.
    @param count: number of items in bin.
    """

    bin_low: float
    bin_high: float
    mean_predicted: float
    observed_frequency: float
    count: int


def reliability_diagram(
    observed: ObservedLineup,
    scenario_set: ScenarioSet,
    n_bins: int = 10,
) -> list[ReliabilityPoint]:
    """T7b : Reliability diagram data points (Bröcker & Smith 2007).

    Pour chaque bin non vide, retourne (bin_low, bin_high, mean_predicted,
    observed_frequency, count). Utilisé pour tracer la calibration curve
    (identity line = perfect calibration) dans ALI_MODEL_CARD.md.

    @param observed: lineup réel.
    @param scenario_set: ScenarioSet ALI.
    @param n_bins: nombre de bins equal-width sur [0, 1]. Default 10.

    @raises ValueError: n_bins < 1.
    @returns list de ReliabilityPoint (bins non vides uniquement, ordonnés).
    """
    if n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)
    pairs = _build_presence_pairs(observed, scenario_set)
    if not pairs:
        return []

    points: list[ReliabilityPoint] = []
    bin_width = 1.0 / n_bins
    for m in range(n_bins):
        lo, hi = m * bin_width, (m + 1) * bin_width
        in_bin = [(p, o) for p, o in pairs if lo <= p < hi or (m == n_bins - 1 and p == hi)]
        if not in_bin:
            continue
        points.append(
            ReliabilityPoint(
                bin_low=lo,
                bin_high=hi,
                mean_predicted=sum(p for p, _ in in_bin) / len(in_bin),
                observed_frequency=sum(o for _, o in in_bin) / len(in_bin),
                count=len(in_bin),
            )
        )
    return points
