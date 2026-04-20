"""Types et helpers pour BacktestRunner (T11 Plan 3 V2).

Split depuis runner.py pour respecter ISO 5055 <= 300 lignes/fichier.

Document ID: ALICE-BACKTEST-RUNNER-TYPES
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from services.ali.types import PlayerCandidate

if TYPE_CHECKING:
    from scripts.backtest.bootstrap import BootstrapCI
    from scripts.backtest.statistical import McNemarResult

# Gate thresholds (Plan 3 V2 P3G07-P3G11)
RECALL_GATE = 0.90
JACCARD_GATE = 0.75
BRIER_GATE = 0.20
BSS_GATE = 0.05
ECE_GATE = 0.05


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
    ali_correct: bool
    baseline_correct: bool


@dataclass(frozen=True)
class BacktestReport:
    """Aggregated backtest report with bootstrap CIs + McNemar.

    @param n_matches: number of successfully-run matches.
    @param per_match: list of MatchStats (lineage per match).
    @param ci_recall, ci_accuracy, ci_jaccard, ci_brier, ci_ece : BCa CIs.
    @param mean_bss: mean Brier skill score (ALI vs baseline).
    @param mcnemar: paired McNemar result on (ali_correct, baseline_correct).
    """

    n_matches: int
    per_match: list[MatchStats]
    ci_recall: BootstrapCI
    ci_accuracy: BootstrapCI
    ci_jaccard: BootstrapCI
    ci_brier: BootstrapCI
    ci_ece: BootstrapCI
    mean_bss: float
    mcnemar: McNemarResult

    def gates_summary(self) -> dict[str, bool]:
        """Return dict gate_name -> pass/fail (Plan 3 V2 P3G07-P3G11)."""
        return {
            "P3G07_recall": self.ci_recall.passes_gate(RECALL_GATE, direction="ge"),
            "P3G08_jaccard": self.ci_jaccard.passes_gate(JACCARD_GATE, direction="ge"),
            "P3G09_brier": self.ci_brier.passes_gate(BRIER_GATE, direction="le"),
            "P3G09_bss": self.mean_bss >= BSS_GATE,
            "P3G10_ece": self.ci_ece.passes_gate(ECE_GATE, direction="le"),
            "P3G11_mcnemar": self.mcnemar.passes_gate(alpha=0.05),
        }


@dataclass(frozen=True)
class RunnerConfig:
    """Backtest runner configuration (immutable)."""

    saison: int = 2024
    rondes: tuple[int, ...] = (5, 7, 9, 11)
    max_matches: int = 50
    team_size: int = 8
    division: str = "N3"
    nb_rondes_total: int = 11
    seed: int = 42
    n_bootstrap: int = 1000
    bootstrap_confidence: float = 0.95
    skip_failed_matches: bool = True


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
