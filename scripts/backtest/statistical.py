"""Statistical tests T9 McNemar (paired binary comparison).

Plan 3 V2 Phase 3 Task 9. ISO 24029 (robustness), 25059 (quality gates).

Protocole McNemar (1947) : comparison paire de 2 classifieurs binaires sur
les MEMES observations → minimise variance nuisance vs tests indépendants.

Table 2x2 :
- b = matches où ALI "correct" mais baseline "incorrect"
- c = matches où baseline "correct" mais ALI "incorrect"
- H0 : b == c (pas de différence)
- H1 : b != c (bilateral)

Décision :
- Si ``b + c < 25`` → exact binomial test (statsmodels ``exact=True``)
- Sinon → chi2 avec continuity correction (Edwards 1948)

Gate seuil : ``p < 0.05`` bilateral → ALI significativement différent baseline
(Pappalardo 2019 sports SOTA, standard en ML comparisons).

Définition "correct" par match (T22 usage) :
Un match est "correct" pour le classifieur si recall >= threshold T13 (0.90).
Alternative : top_scenario accuracy >= 0.75 (T13b). Choix à documenter
explicitement dans ALI_QUALITY_GATES_REPORT.md.

Sources SOTA
------------
- McNemar Q. 1947, "Note on the sampling error of the difference between
  correlated proportions", Psychometrika 12(2).
- Edwards A.L. 1948, "Note on the correction for continuity in testing
  the significance of the difference between correlated proportions".
- Dietterich 1998 "Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms" (MIT J Neural Computation 10).
- Pappalardo et al. 2019 PlayeRank — sports paired comparison.

Document ID: ALICE-BACKTEST-STATISTICAL
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class McNemarResult:
    """Resultat McNemar paired test (frozen, ISO 29119 immutability).

    @param statistic: chi2 statistic (continuity-corrected si n_disc>=25) ou
                      binomial test statistic (min(b,c)) si n_disc<25.
    @param p_value: two-sided p-value ∈ [0, 1].
    @param b: count (ALI correct ∧ baseline incorrect).
    @param c: count (baseline correct ∧ ALI incorrect).
    @param n_discordant: b + c.
    @param method: "exact_binomial" si n_disc<25 sinon "chi2_continuity".
    @param significant: True si p_value < alpha (default 0.05).
    """

    statistic: float
    p_value: float
    b: int
    c: int
    n_discordant: int
    method: str
    significant: bool

    def passes_gate(self, alpha: float = 0.05) -> bool:
        """Return True si p_value < alpha (gate significativite bilaterale)."""
        return self.p_value < alpha


def mcnemar_paired(
    ali_correct: list[bool],
    baseline_correct: list[bool],
    alpha: float = 0.05,
) -> McNemarResult:
    """T9 McNemar paired test : ALI vs baseline sur N matches apparies.

    Pour chaque match ``i`` : ``ali_correct[i]`` et ``baseline_correct[i]``
    sont 2 bool indiquant si chaque modele a predit correctement.

    Sélection du test :
    - ``n_disc = b + c``. Si ``n_disc < 25`` → exact binomial (pas
      d'approximation asymptotique fiable sur petits échantillons).
    - Sinon → chi2 Yates-corrected ``(|b-c| - 1)² / (b+c)``.

    @param ali_correct: liste bool longueur N (predictions ALI par match).
    @param baseline_correct: liste bool longueur N (baseline Elo par match).
    @param alpha: seuil significativité bilatéral. Default 0.05.

    @raises ValueError: longueurs différentes, vides, ou alpha ∉ (0, 1).
    @returns McNemarResult frozen.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    if len(ali_correct) != len(baseline_correct):
        msg = (
            f"length mismatch : ali_correct={len(ali_correct)} "
            f"vs baseline_correct={len(baseline_correct)}"
        )
        raise ValueError(msg)
    if len(ali_correct) == 0:
        msg = "empty inputs : McNemar requires >= 1 paired observation"
        raise ValueError(msg)

    # Build 2x2 contingency : b = ALI+ baseline- ; c = ALI- baseline+
    b = sum(1 for a, bl in zip(ali_correct, baseline_correct, strict=True) if a and not bl)
    c = sum(1 for a, bl in zip(ali_correct, baseline_correct, strict=True) if not a and bl)
    n_disc = b + c

    # Avoid circular import with SciPy default; statsmodels handles both paths.
    from statsmodels.stats.contingency_tables import mcnemar  # noqa: PLC0415

    table = np.array([[0, b], [c, 0]])
    # statsmodels exact=True uses binomial test if b+c small ; else chi2 Yates.
    if n_disc < 25:
        result = mcnemar(table, exact=True, correction=False)
        method = "exact_binomial"
    else:
        result = mcnemar(table, exact=False, correction=True)
        method = "chi2_continuity"

    return McNemarResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        b=b,
        c=c,
        n_discordant=n_disc,
        method=method,
        significant=float(result.pvalue) < alpha,
    )
