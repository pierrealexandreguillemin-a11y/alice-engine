"""Build 18 meta-features for MLP stacking (ISO 42001 tracability).

Document ID: ALICE-META-FEATURES
Version: 1.0.0

Input: 3 model probability arrays (n, 3) each.
Output: (n, 18) array = 9 base probas + 3 std + 3 max_prob + 3 entropy.

Ref: MODEL_SPECS.md section "Meta-features : 18 = 9 probas + 9 engineered"
"""

from __future__ import annotations

import numpy as np


def build_meta_features(
    p_xgb: np.ndarray,
    p_lgb: np.ndarray,
    p_cb: np.ndarray,
) -> np.ndarray:
    """Build 18 meta-features from 3 model probability outputs.

    Args:
    ----
        p_xgb: (n, 3) XGBoost probabilities [P(loss), P(draw), P(win)]
        p_lgb: (n, 3) LightGBM probabilities
        p_cb: (n, 3) CatBoost probabilities

    Returns:
    -------
        (n, 18) array: 9 base probas + 3 std + 3 max_prob + 3 entropy
    """
    base = np.hstack([p_xgb, p_lgb, p_cb])  # (n, 9)

    feats = [base]

    # Per-class std across 3 models (disagreement)
    for c in range(3):
        preds = np.stack([p_xgb[:, c], p_lgb[:, c], p_cb[:, c]], axis=1)
        feats.append(preds.std(axis=1, keepdims=True))

    # Max probability per model (confidence)
    for model_p in [p_xgb, p_lgb, p_cb]:
        feats.append(model_p.max(axis=1, keepdims=True))

    # Shannon entropy per model (uncertainty)
    for model_p in [p_xgb, p_lgb, p_cb]:
        p = np.clip(model_p, 1e-7, 1.0)
        feats.append((-p * np.log(p)).sum(axis=1, keepdims=True))

    return np.hstack(feats)
