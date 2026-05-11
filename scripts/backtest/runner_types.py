"""Types et helpers pour BacktestRunner (T11 Plan 3 V2).

Split depuis runner.py pour respecter ISO 5055 <= 300 lignes/fichier.

Document ID: ALICE-BACKTEST-RUNNER-TYPES
Version: 1.0.0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from services.ali.types import PlayerCandidate


def _max_matches_default() -> int:
    """ALICE_MAX_MATCHES env var override (D-2026-05-11). Default 50 preserved."""
    raw = os.environ.get("ALICE_MAX_MATCHES", "50")
    try:
        v = int(raw)
    except (TypeError, ValueError) as exc:
        msg = f"ALICE_MAX_MATCHES must be int, got {raw!r}"
        raise ValueError(msg) from exc
    if v < 1:
        msg = f"ALICE_MAX_MATCHES must be >=1, got {v}"
        raise ValueError(msg)
    return v

if TYPE_CHECKING:
    from scripts.backtest.bootstrap import BootstrapCI
    from scripts.backtest.statistical import McNemarResult, WilcoxonResult

# Gate thresholds (Plan 3 V2 P3G07-P3G11)
RECALL_GATE = 0.90
JACCARD_GATE = 0.75
BRIER_GATE = 0.20
BSS_GATE = 0.05
ECE_GATE = 0.05
MAE_GATE = 1.0

# McNemar paired test seuil "correct" : RECALL_GATE strict 0.90.
# T22 review user 2026-04-28 a rejeté l'idée de baisser le seuil (=
# statistical hacking, pas SOTA). McNemar binaire dichotomise une métrique
# continue (recall ∈ [0,1]) → perte d'information. Solution SOTA = Wilcoxon
# signed-rank test paired sur recall_ali vs recall_baseline (continu).
# McNemar conservé comme outil secondaire (legacy P3G11b spec) mais le test
# de référence est Wilcoxon (cf. statistical.py::wilcoxon_paired).


@dataclass(frozen=True)
class MatchStats:
    """Per-match metrics + correctness flags ALI vs baseline."""

    saison: int
    ronde: int
    user_team: str
    opponent_team: str
    recall_ali: float
    accuracy_ali: float
    jaccard_ali: float
    brier_ali: float
    ece_ali: float
    recall_baseline: float
    brier_baseline: float
    bss: float
    e_score_predicted: float
    e_score_observed: float
    e_score_mae: float
    ali_correct: bool
    baseline_correct: bool


@dataclass(frozen=True)
class BacktestReport:
    """Aggregated backtest report with bootstrap CIs + McNemar + Wilcoxon.

    @param n_matches: number of successfully-run matches.
    @param per_match: list of MatchStats (lineage per match).
    @param ci_recall, ci_accuracy, ci_jaccard, ci_brier, ci_ece : BCa CIs.
    @param mean_bss: mean Brier skill score (ALI vs baseline).
    @param mcnemar: paired McNemar result on (ali_correct, baseline_correct)
                    binary - LEGACY P3G11b spec, dichotomise recall>=0.90.
    @param wilcoxon_recall: paired Wilcoxon signed-rank on continuous
                            recall_ali vs recall_baseline (D-P3-18 SOTA,
                            T22 finding 2026-04-28). Test PRINCIPAL pour
                            P3G11b.
    """

    n_matches: int
    per_match: list[MatchStats]
    ci_recall: BootstrapCI
    ci_accuracy: BootstrapCI
    ci_jaccard: BootstrapCI
    ci_brier: BootstrapCI
    ci_ece: BootstrapCI
    ci_mae: BootstrapCI
    mean_bss: float
    mcnemar: McNemarResult
    wilcoxon_recall: WilcoxonResult

    def gates_summary(self) -> dict[str, bool]:
        """Return dict gate_name -> pass/fail (Plan 3 V2 P3G07-P3G11).

        P3G11b utilise Wilcoxon signed-rank (D-P3-18 SOTA) sur recall continu.
        McNemar binaire conservé en metric secondaire pour conformité spec.
        """
        return {
            "P3G07_recall": self.ci_recall.passes_gate(RECALL_GATE, direction="ge"),
            "P3G08_jaccard": self.ci_jaccard.passes_gate(JACCARD_GATE, direction="ge"),
            "P3G09_brier": self.ci_brier.passes_gate(BRIER_GATE, direction="le"),
            "P3G09_bss": self.mean_bss >= BSS_GATE,
            "P3G10_ece": self.ci_ece.passes_gate(ECE_GATE, direction="le"),
            "P3G11_mae": self.ci_mae.passes_gate(MAE_GATE, direction="le"),
            "P3G11_wilcoxon_recall": self.wilcoxon_recall.passes_gate(alpha=0.05),
            "P3G11_mcnemar_legacy": self.mcnemar.passes_gate(alpha=0.05),
        }


@dataclass(frozen=True)
class MatchCandidate:
    """One sampled hold-out match candidate (pre-attempt).

    Frozen for ISO 29119 deterministic sampling lineage.

    D-2026-05-11 fix : ``groupe`` disambiguates multi-phase competitions
    (Top 16 saison 2024 = 4 groupes : "Groupe A", "Groupe B" rondes 1-7
    régulière puis "Poule Haute"/"Poule Basse" rondes 1-4 finale). Sans
    groupe, ``_select_match_rows`` mélange phase 1 et phase 2 pour une
    équipe qualifiée → trip invariant FFE A02 §3.6. Default empty pour
    backward compat (N1/N2/N3/N4 : 1 équipe = 1 groupe par saison).
    """

    saison: int
    ronde: int
    user_team: str
    opp_team: str
    opp_club: str
    groupe: str = ""


@dataclass(frozen=True)
class RunnerConfig:
    """Backtest runner configuration (immutable).

    Stratified sampling (Plan 3 V2 T22 fix-on-sight) :
    - ``type_competition`` : strict filter ('national' = SE adulte interclub).
      Exclut 'scolaire', 'national_jeunes' (D3), 'coupe_*' (D4), 'regional'.
    - ``division_filter`` : strict exact-match filter pandas
      (ex 'Nationale 3'). Distinct de ``division`` (label `N3` Phase 3
      passé au scenario_generator).
    - ``stratify_min_per_ronde`` : seuil ISO 24027 §6 minimum sample size
      per stratum pour fairness audit (default 5 = T15 default).
    """

    saison: int = 2024
    rondes: tuple[int, ...] = (5, 7, 9, 11)
    max_matches: int = field(default_factory=_max_matches_default)
    team_size: int = 8
    division: str = "N3"
    nb_rondes_total: int = 11
    seed: int = 42
    n_bootstrap: int = 1000
    bootstrap_confidence: float = 0.95
    skip_failed_matches: bool = True
    type_competition: str = "national"
    division_filter: str = "Nationale 3"
    stratify_min_per_ronde: int = 5


def df_to_candidates(df: object, club: str) -> list[PlayerCandidate]:
    """Convert joueurs DataFrame rows to PlayerCandidate list for baseline.

    @param df: pandas DataFrame with joueurs columns (nr_ffe, nom, prenom,
               elo, mute, genre, categorie).
    @param club: club name to set on each candidate.
    """
    out: list[PlayerCandidate] = []
    for _, p in df.iterrows():  # type: ignore[attr-defined]
        elo_raw = p.get("elo")
        if elo_raw is None or (isinstance(elo_raw, float) and math.isnan(elo_raw)):
            elo = 1500
        else:
            elo = int(elo_raw) if elo_raw else 1500
        out.append(
            PlayerCandidate(
                nr_ffe=str(p["nr_ffe"]),
                nom=str(p.get("nom", "")),
                prenom=str(p.get("prenom", "")),
                elo=elo,
                club=club,
                mute=bool(p.get("mute", False)),
                genre=str(p.get("genre", "M")),
                categorie=str(p.get("categorie", "SE")),
                licence_active=True,
            )
        )
    return out
