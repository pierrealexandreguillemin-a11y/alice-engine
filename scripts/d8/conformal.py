"""D8 conformal prediction split (Vovk 2024, Angelopoulos 2023).

Source SOTA :
- Vovk et al. 2024 "Algorithmic Learning in a Random World" 2nd ed. §2.3 split conformal
- Angelopoulos & Bates 2023 "Conformal Prediction: A Gentle Introduction" §4.2 efficiency
- Lei et al. 2018 "Distribution-Free Predictive Inference for Regression" JASA
- ISO/IEC TR 24029-2:2024 §5.3 robustness via prediction intervals

Marginal coverage guarantee (Vovk 2024 Theorem 2.1) :
    P(y_obs in C_alpha(y_pred)) >= 1 - alpha
under the only assumption that calibration and test data are exchangeable.

Document ID: ALICE-D8-CONFORMAL
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConformalCalibration:
    """Split conformal calibration result (Vovk 2024 §2.3).

    @attr quantile_threshold: q_hat threshold |y_obs - y_pred| <= q_hat with proba >= 1-alpha
    @attr nonconf_scores: sorted ascending nonconformity scores from calibration set
    @attr n_calibration: |D_calib|
    @attr alpha: miscoverage rate, target coverage = 1 - alpha
    """

    quantile_threshold: float
    nonconf_scores: NDArray[np.float64]
    n_calibration: int
    alpha: float


def _validate_alpha(alpha: float) -> None:
    """ISO 27034 input validation: alpha in open (0, 1)."""
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)


def _validate_finite(arr: NDArray[np.float64], name: str) -> None:
    """ISO 27034 input validation : reject NaN/inf."""
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        msg = f"{name} contains NaN or inf values"
        raise ValueError(msg)


def split_calibrate(
    y_observed: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    alpha: float = 0.10,
) -> ConformalCalibration:
    """Split conformal calibration (Vovk 2024 §2.3).

    Nonconformity score s_i = |y_obs_i - y_pred_i|.
    Quantile rank k = ceil((1-alpha)*(n+1)) - 1 (0-indexed in sorted scores).
    Threshold q_hat = sorted_scores[min(k, n-1)].

    @param y_observed: observed targets in [0, 1] (E[score] per board)
    @param y_predicted: model predictions in [0, 1]
    @param alpha: miscoverage rate, default 0.10 (90% coverage)
    @raises ValueError: alpha not in (0,1), empty input, length mismatch, NaN/inf
    """
    _validate_alpha(alpha)
    if len(y_observed) == 0:
        msg = "empty calibration input"
        raise ValueError(msg)
    if len(y_observed) != len(y_predicted):
        msg = f"length mismatch y_observed={len(y_observed)} y_predicted={len(y_predicted)}"
        raise ValueError(msg)
    _validate_finite(y_observed, "y_observed")
    _validate_finite(y_predicted, "y_predicted")

    nonconf = np.abs(y_observed - y_predicted).astype(np.float64)
    sorted_scores = np.sort(nonconf)
    n = len(sorted_scores)
    # Vovk 2024 §2.3 : finite-sample correction k = ceil((1-alpha)*(n+1)) - 1
    rank = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    rank_clipped = max(0, min(rank, n - 1))
    threshold = float(sorted_scores[rank_clipped])
    return ConformalCalibration(
        quantile_threshold=threshold,
        nonconf_scores=sorted_scores,
        n_calibration=n,
        alpha=alpha,
    )


def coverage_rate(
    y_observed: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    calibration: ConformalCalibration,
) -> float:
    """Marginal coverage P(|y_obs - y_pred| <= q_hat) on test set.

    @param y_observed: observed targets test set
    @param y_predicted: model predictions test set
    @param calibration: ConformalCalibration from split_calibrate
    @raises ValueError: length mismatch
    """
    if len(y_observed) != len(y_predicted):
        msg = f"length mismatch y_observed={len(y_observed)} y_predicted={len(y_predicted)}"
        raise ValueError(msg)
    if len(y_observed) == 0:
        return 0.0
    nonconf = np.abs(y_observed - y_predicted)
    covered = nonconf <= calibration.quantile_threshold
    return float(covered.mean())


def coverage_per_group(
    y_observed: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    groups: list[str],
    calibration: ConformalCalibration,
) -> dict[str, float]:
    """Coverage breakdown by group label (ISO 24027 §6 fairness).

    @param y_observed: observed targets test set
    @param y_predicted: model predictions test set
    @param groups: per-sample group label, len == len(y_observed)
    @param calibration: ConformalCalibration from split_calibrate
    @raises ValueError: length mismatch
    """
    if len(y_observed) == 0:
        return {}
    if len(y_observed) != len(groups):
        msg = f"length mismatch y_observed={len(y_observed)} groups={len(groups)}"
        raise ValueError(msg)
    arr_groups = np.array(groups)
    unique_groups = set(groups)
    return {
        g: coverage_rate(
            y_observed[arr_groups == g],
            y_predicted[arr_groups == g],
            calibration,
        )
        for g in unique_groups
    }


def conformal_set_size_mean(
    y_predicted: NDArray[np.float64],
    calibration: ConformalCalibration,
    support_max: float = 1.0,
    grid_resolution: float = 0.01,
) -> float:
    """Mean conformal set size = efficiency (Angelopoulos 2023 §4.2).

    For each y_pred, the conformal set is { y in [0, support_max] :
    |y - y_pred| <= q_hat }.
    Set size = min(y_pred + q_hat, support_max) - max(y_pred - q_hat, 0).

    D-2026-05-11 fix : ``support_max`` était hardcodé 1.0, masquant la vraie
    efficiency pour métriques sur [0, K] (E[score] match sum, K=team_size=8).
    Pour q_hat=4.4 sur [0,8], le clip [0,1] saturait artificiellement set_size
    à 1.0. ISO 24029 §5.3 (robustness via prediction intervals) requiert que
    l'efficiency soit mesurable, donc support_max doit refléter le vrai
    support de la prédiction.

    @param y_predicted: model predictions in [0, support_max]
    @param calibration: ConformalCalibration from split_calibrate
    @param support_max: upper bound of the prediction support (default 1.0
        pour probas. Pour E[score] match sum sur K boards, passer K=team_size).
    @param grid_resolution: granularity for clipping bounds (default 0.01)
    @raises ValueError: grid_resolution not in (0, 1], support_max <= 0
    """
    if not (0.0 < grid_resolution <= 1.0):
        msg = f"grid_resolution must be in (0, 1], got {grid_resolution}"
        raise ValueError(msg)
    if support_max <= 0.0:
        msg = f"support_max must be > 0, got {support_max}"
        raise ValueError(msg)
    if len(y_predicted) == 0:
        return 0.0
    q = calibration.quantile_threshold
    upper = np.minimum(y_predicted + q, support_max)
    lower = np.maximum(y_predicted - q, 0.0)
    sizes = upper - lower
    return float(sizes.mean())
