"""Baseline models for quality gate comparison — ISO 25059.

Document ID: ALICE-BASELINES
Version: 1.0.0

Provides naive (marginal distribution) and Elo-based draw-rate
baselines for multiclass (loss/draw/win) comparison.

ISO Compliance:
- ISO/IEC 25059:2023 - AI Quality Model (baseline comparison)
- ISO/IEC 5055:2021 - Code Quality (SRP, <300 lines)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_naive_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Always predict marginal class distribution. Returns (n_test, 3)."""
    counts = np.bincount(y_train, minlength=3)
    probs = counts / counts.sum()
    return np.tile(probs, (n_test, 1))


def compute_elo_baseline(
    blanc_elo: np.ndarray,
    noir_elo: np.ndarray,
    draw_rate_lookup: pd.DataFrame,
) -> np.ndarray:
    """Elo formula + draw rate lookup. Returns (n, 3) = [P(loss), P(draw), P(win)].

    White advantage of +35 Elo points is a standard correction
    for first-move advantage in chess.
    """
    diff = noir_elo - blanc_elo
    expected = 1 / (1 + 10 ** ((diff - 35) / 400))  # +35 white advantage
    avg = (blanc_elo + noir_elo) / 2
    abs_diff = np.abs(blanc_elo - noir_elo)
    draw_rate = _lookup_draw_rate(avg, abs_diff, draw_rate_lookup)
    p_win = np.clip(expected - 0.5 * draw_rate, 0, 1)
    p_draw = np.clip(draw_rate, 0, 1)
    p_loss = np.clip(1 - p_win - p_draw, 0, 1)
    total = p_win + p_draw + p_loss
    total = np.where(total == 0, 1, total)
    return np.column_stack([p_loss / total, p_draw / total, p_win / total])


def compute_elo_init_scores(
    elo_proba: np.ndarray,
    clip_min: float = 1e-4,
    clip_max: float = 1 - 1e-4,
) -> np.ndarray:
    """Convert Elo baseline probas to centered log-odds for residual learning.

    For multiclass: init_score = log(P_k) - mean(log(P)) per sample.
    This is the zero-sum representative of the softmax pre-image:
    softmax(init_scores) recovers the original probas. The centering is
    arbitrary (any constant offset gives the same softmax) but conventional
    for CatBoost Pool(baseline=), XGBoost base_margin, LightGBM init_score.

    Args:
    ----
        elo_proba: (n, 3) array of [P(loss), P(draw), P(win)] from Elo baseline.
        clip_min: Floor for probabilities to avoid log(0).
        clip_max: Ceiling for probabilities to avoid log(1).

    Returns:
    -------
        (n, 3) array of log-odds suitable for init_score / base_margin.
    """
    clipped = np.clip(elo_proba, clip_min, clip_max)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    log_p = np.log(clipped)
    init_scores = log_p - log_p.mean(axis=1, keepdims=True)
    return init_scores


def compute_init_scores_from_features(X: pd.DataFrame, draw_lookup: pd.DataFrame) -> np.ndarray:
    """Compute Elo baseline init scores from X features (needs blanc_elo/noir_elo columns).

    Convenience wrapper for train_kaggle.py and inference (Phase 2).
    Falls back to 1500 Elo when columns are missing.
    """
    b_elo = X["blanc_elo"].values if "blanc_elo" in X.columns else np.full(len(X), 1500)
    n_elo = X["noir_elo"].values if "noir_elo" in X.columns else np.full(len(X), 1500)
    elo_proba = compute_elo_baseline(b_elo, n_elo, draw_lookup)
    return compute_elo_init_scores(elo_proba)


def _lookup_draw_rate(
    avg_elo: np.ndarray,
    abs_diff: np.ndarray,
    lookup: pd.DataFrame,
) -> np.ndarray:
    """Map (avg_elo, abs_diff) to draw rate via the draw_priors lookup table.

    Uses the same ELO_BINS / DIFF_BINS as draw_priors.py
    and joins on the string band labels.
    """
    from scripts.features.draw_priors import DIFF_BINS, ELO_BINS  # noqa: PLC0415

    temp = pd.DataFrame({"avg": avg_elo, "diff": abs_diff})
    elo_labels = [f"{ELO_BINS[i]}-{ELO_BINS[i + 1]}" for i in range(len(ELO_BINS) - 1)]
    diff_labels = [f"{DIFF_BINS[i]}-{DIFF_BINS[i + 1]}" for i in range(len(DIFF_BINS) - 1)]
    temp["elo_band"] = pd.cut(temp["avg"], bins=ELO_BINS, labels=elo_labels, right=False).astype(
        str,
    )
    temp["diff_band"] = pd.cut(
        temp["diff"], bins=DIFF_BINS, labels=diff_labels, right=False
    ).astype(
        str,
    )
    merged = temp.merge(lookup, on=["elo_band", "diff_band"], how="left")
    global_rate = lookup["draw_rate_prior"].mean() if not lookup.empty else 0.13
    return merged["draw_rate_prior"].fillna(global_rate).values
