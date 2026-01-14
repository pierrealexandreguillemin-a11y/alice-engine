"""Statistical Tests Drift - ISO 23894.

Tests statistiques pour la détection de drift:
- PSI (Population Stability Index)
- KS-test (Kolmogorov-Smirnov)
- Chi-squared (catégoriel)
- Jensen-Shannon divergence

ISO Compliance:
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5055:2021 - Code Quality (<150 lignes, SRP)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def compute_psi(
    baseline: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """Calcule le Population Stability Index (PSI).

    PSI < 0.1: Pas de changement
    0.1-0.2: Changement modéré
    0.2-0.25: Warning
    >= 0.25: Drift majeur
    """
    baseline = np.asarray(baseline).flatten()
    current = np.asarray(current).flatten()

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(baseline, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
    current_counts = np.histogram(current, bins=bin_edges)[0]

    baseline_pct = (baseline_counts + eps) / (len(baseline) + eps * n_bins)
    current_pct = (current_counts + eps) / (len(current) + eps * n_bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def compute_ks_test(
    baseline: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Calcule le test de Kolmogorov-Smirnov (ISO 23894)."""
    from scipy import stats

    baseline = np.asarray(baseline).flatten()
    current = np.asarray(current).flatten()

    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) < 2 or len(current) < 2:
        return 0.0, 1.0

    result = stats.ks_2samp(baseline, current)
    return float(result.statistic), float(result.pvalue)


def compute_chi2_test(
    baseline: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Calcule le test Chi-squared pour features catégorielles."""
    from scipy import stats

    baseline = np.asarray(baseline).flatten()
    current = np.asarray(current).flatten()

    all_categories = np.unique(np.concatenate([baseline, current]))

    baseline_counts = np.array([np.sum(baseline == cat) for cat in all_categories])
    current_counts = np.array([np.sum(current == cat) for cat in all_categories])

    total_baseline = np.sum(baseline_counts)
    total_current = np.sum(current_counts)

    if total_baseline == 0 or total_current == 0:
        return 0.0, 1.0

    expected = baseline_counts * (total_current / total_baseline)
    expected = np.where(expected == 0, 1e-8, expected)

    chi2 = np.sum((current_counts - expected) ** 2 / expected)
    dof = max(1, len(all_categories) - 1)
    p_value = 1 - stats.chi2.cdf(chi2, dof)

    return float(chi2), float(p_value)


def compute_js_divergence(
    baseline: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Calcule la divergence de Jensen-Shannon (0-1)."""
    from scipy.spatial.distance import jensenshannon

    baseline = np.asarray(baseline).flatten()
    current = np.asarray(current).flatten()

    all_data = np.concatenate([baseline, current])
    bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

    baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
    current_hist, _ = np.histogram(current, bins=bins, density=True)

    eps = 1e-10
    baseline_hist = (baseline_hist + eps) / (baseline_hist + eps).sum()
    current_hist = (current_hist + eps) / (current_hist + eps).sum()

    return float(jensenshannon(baseline_hist, current_hist))
