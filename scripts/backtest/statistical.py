"""Statistical tests for paired classifier comparison (T9 McNemar + Wilcoxon).

Plan 3 V2 Phase 3 Task 9 + T22.D-P3-18 Wilcoxon (2026-04-28).
ISO 24029 (robustness), 25059 (quality gates).

Deux tests complémentaires :

**Wilcoxon signed-rank paired (TEST PRINCIPAL D-P3-18 SOTA)**
- Opère directement sur les valeurs continues `recall_ali[i]` vs
  `recall_baseline[i]`. Aucune dichotomisation arbitraire (correct/incorrect).
- Distribution-free, paired (mêmes matches). H0 : médiane des
  différences = 0. H1 bilateral : médiane ≠ 0.
- Plus puissant que McNemar quand la métrique est continue (Pratt 1959).
- Utilise scipy.stats.wilcoxon (zsplit pour 0-différences, exact si n<25).

**McNemar paired binary (TEST SECONDAIRE legacy P3G11b spec)**
- Dichotomise via `ali_correct = recall >= RECALL_GATE`. Perd l'information
  continue mais préservé pour conformité Plan 3 V2 §6.2 spec.
- Table 2x2 : b = ALI correct seul, c = baseline correct seul.
- Si b + c < 25 → exact binomial (statsmodels), sinon chi² Yates.
- Si gates absolus inatteignables (P3G07 strict), McNemar dégénère
  (n_disc petit). Wilcoxon reste valide.

Sources SOTA
------------
- Wilcoxon F. 1945, "Individual comparisons by ranking methods", Biometrics
  Bulletin 1(6), 80-83.
- Pratt J. W. 1959, "Remarks on zeros and ties in the Wilcoxon signed
  rank procedures", JASA 54(287), 655-667.
- McNemar Q. 1947, "Note on the sampling error of the difference between
  correlated proportions", Psychometrika 12(2).
- Edwards A.L. 1948, "Note on the correction for continuity in testing
  the significance of the difference between correlated proportions".
- Dietterich 1998 "Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms" (MIT J Neural Computation 10).
- Pappalardo et al. 2019 PlayeRank — sports paired comparison.
- Demšar J. 2006 "Statistical comparisons of classifiers over multiple
  datasets" JMLR 7 — Wilcoxon recommended over McNemar for continuous
  performance metrics.

Document ID: ALICE-BACKTEST-STATISTICAL
Version: 1.1.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


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


@dataclass(frozen=True)
class WilcoxonResult:
    """Resultat Wilcoxon signed-rank paired test (D-P3-18 SOTA, 2026-04-28).

    @param statistic: somme des rangs des differences positives (W+).
                      Sous H0, W+ ≈ n(n+1)/4. W+ >> implique ali > baseline.
    @param p_value: bilateral p-value (H1 : median diff != 0).
    @param n_pairs: nombre total de paires utilisees (= len(ali_values)).
    @param n_nonzero: nombre de paires apres exclusion des differences
                      egales a 0 (zero_method='wilcox' standard).
    @param median_diff: mediane des differences ali_values - baseline_values
                        (interpretation : > 0 => ali domine en mediane).
    @param mean_diff: moyenne des differences (informatif, non utilise
                      dans le test).
    @param method: "exact" si n_nonzero < 25, sinon "approx_normal" (avec
                   continuity correction).
    @param significant: True si p_value < alpha (default 0.05).
    """

    statistic: float
    p_value: float
    n_pairs: int
    n_nonzero: int
    median_diff: float
    mean_diff: float
    method: str
    significant: bool

    def passes_gate(self, alpha: float = 0.05) -> bool:
        """Return True si p_value < alpha bilateral."""
        return self.p_value < alpha


def wilcoxon_paired(
    ali_values: list[float],
    baseline_values: list[float],
    alpha: float = 0.05,
) -> WilcoxonResult:
    """T22.D-P3-18 Wilcoxon signed-rank paired test sur valeurs continues.

    Compare distribution `ali_values[i] - baseline_values[i]` vs 0
    (test bilateral H1 : median diff != 0). Distribution-free, plus
    puissant que McNemar quand la metrique est continue (Demsar 2006).

    @param ali_values: per-match ALI metric (e.g. recall, jaccard, brier).
    @param baseline_values: per-match baseline metric, MEMES indices que
                            ali_values (paired observations).
    @param alpha: seuil significativite bilateral. Default 0.05.

    @raises ValueError: longueurs differentes, vides, ou alpha hors (0, 1).
    @returns WilcoxonResult frozen.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    if len(ali_values) != len(baseline_values):
        msg = (
            f"length mismatch : ali_values={len(ali_values)} "
            f"vs baseline_values={len(baseline_values)}"
        )
        raise ValueError(msg)
    if len(ali_values) == 0:
        msg = "empty inputs : Wilcoxon requires >= 1 paired observation"
        raise ValueError(msg)

    arr_ali = np.asarray(ali_values, dtype=float)
    arr_base = np.asarray(baseline_values, dtype=float)
    diffs = arr_ali - arr_base
    n_nonzero = int((diffs != 0.0).sum())

    # Edge cases : toutes les differences sont 0 => H0 trivialement vrai,
    # p_value = 1.0, aucune evidence.
    if n_nonzero == 0:
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            n_pairs=len(arr_ali),
            n_nonzero=0,
            median_diff=0.0,
            mean_diff=0.0,
            method="degenerate_no_diff",
            significant=False,
        )

    # scipy choisit auto exact (n<25) ou approx normal (continuity).
    method_str = "exact" if n_nonzero < 25 else "approx_normal"
    res = stats.wilcoxon(
        x=arr_ali,
        y=arr_base,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return WilcoxonResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_pairs=len(arr_ali),
        n_nonzero=n_nonzero,
        median_diff=float(np.median(diffs)),
        mean_diff=float(np.mean(diffs)),
        method=method_str,
        significant=float(res.pvalue) < alpha,
    )
