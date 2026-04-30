"""D8 calibration - per-group ECE + multicalibration alpha (Pleiss 2017, Hebert-Johnson 2018).

Source SOTA :
- Naeini et al. 2015 "Obtaining Well Calibrated Probabilities Using Bayesian Binning" AAAI
- Pleiss et al. 2017 "On Fairness and Calibration" NeurIPS
- Hebert-Johnson et al. 2018 "Multicalibration: Calibration for the (Computationally-Identifiable) Masses" ICML
- ISO/IEC TR 24027:2021 §6.5 calibration per protected group

Document ID: ALICE-D8-CALIBRATION
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expected_calibration_error(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_bins: int = 10,
) -> float:
    """Naeini 2015 §3 ECE - weighted absolute difference, equal-width bins.

    @param probs: predicted probabilities in [0, 1]
    @param labels: binary labels {0, 1}
    @param n_bins: number of equal-width bins
    @raises ValueError: n_bins<=0 or len mismatch
    """
    if n_bins <= 0:
        msg = f"n_bins must be > 0, got {n_bins}"
        raise ValueError(msg)
    if len(probs) != len(labels):
        msg = f"length mismatch probs={len(probs)} labels={len(labels)}"
        raise ValueError(msg)
    if len(probs) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        in_bin = (probs > edges[i]) & (probs <= edges[i + 1])
        if i == 0:
            in_bin = in_bin | (probs == edges[0])
        bin_n = int(in_bin.sum())
        if bin_n == 0:
            continue
        bin_acc = float(labels[in_bin].mean())
        bin_conf = float(probs[in_bin].mean())
        ece += (bin_n / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ece_per_group(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    groups: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """Per-group ECE (Pleiss 2017 §4).

    @param probs: predicted probabilities in [0, 1]
    @param labels: binary labels {0, 1}
    @param groups: per-sample group label, len == len(probs)
    @param n_bins: number of equal-width bins
    @raises ValueError: n_bins<=0 or len mismatch
    """
    if n_bins <= 0:
        msg = f"n_bins must be > 0, got {n_bins}"
        raise ValueError(msg)
    if len(probs) == 0:
        return {}
    if len(probs) != len(groups):
        msg = "len(probs) != len(groups)"
        raise ValueError(msg)
    arr_groups = np.array(groups)
    unique_groups = set(groups)
    return {
        g: expected_calibration_error(probs[arr_groups == g], labels[arr_groups == g], n_bins)
        for g in unique_groups
    }


def compute_multicalibration_alpha(
    probs: NDArray[np.float64],
    labels: NDArray[np.int_],
    groups: dict[str, NDArray[np.bool_]],
    n_bins: int = 10,
) -> float:
    """Multicalibration alpha (Hebert-Johnson 2018 §3) = max sub-group ECE.

    @param probs: predicted probabilities in [0, 1]
    @param labels: binary labels {0, 1}
    @param groups: dict[group_name, boolean mask len(probs)]
    @param n_bins: number of equal-width bins
    """
    if not groups:
        return 0.0
    return max(
        expected_calibration_error(probs[mask], labels[mask], n_bins) for mask in groups.values()
    )
