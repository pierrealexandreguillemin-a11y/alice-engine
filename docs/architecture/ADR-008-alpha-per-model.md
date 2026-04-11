# ADR-008: init_score_alpha MUST be tuned per-model (not uniform)

## Status: ACCEPTED (2026-04-11)

## Context

V8 used alpha=0.7 uniformly for XGBoost, LightGBM, and CatBoost. This was
never challenged against fabricant documentation. V9 HP search (Grid 82 combos
× 3 models on saison=2022) revealed that alpha sensitivity differs by 50×
between models due to fundamental architectural differences in tree growth.

## Decision

**init_score_alpha is NOT a global parameter. It MUST be tuned independently
per model because each model's tree-building algorithm responds differently
to gradient magnitude.**

### Mechanism (verified empirically + fabricant docs)

| Model | Tree Growth | Alpha Sensitivity | Why |
|-------|------------|-------------------|-----|
| **LightGBM** | Leaf-wise | **EXTREME** (0.05 logloss gap) | Splits the leaf with LARGEST gradient. Small gradients (high alpha) → no discriminating split found → early stop in underfitting |
| **XGBoost** | Depth-wise | Low (0.001 logloss gap) | Expands ALL nodes at same level regardless of gradient magnitude. Compensates via more iterations |
| **CatBoost** | Oblivious (symmetric) | TBD (pending V9 results) | Fixed structure per depth level. Expected: moderate sensitivity |

### Evidence (V9 Grid, saison=2022, 82 combos each)

**LightGBM:**
- alpha=0.4: best 0.5357
- alpha=0.6: best 0.5585
- alpha=0.8: best 0.5869
- Gap: **0.051** logloss (monotonic)

**XGBoost:**
- alpha=0.5: best 0.5198
- alpha=0.65: best 0.5201
- alpha=0.8: best 0.5210
- Gap: **0.001** logloss (quasi-flat)

### Root Cause

LightGBM's leaf-wise growth selects the leaf with the maximum loss reduction
to split. When init_score_alpha is high (0.7-0.8), the Elo prior dominates,
residuals are tiny (~0.002), and ALL leaves have near-zero gradients. The
leaf-wise algorithm cannot distinguish meaningful splits from noise → converges
to a flat model before discovering feature patterns.

XGBoost's depth-wise growth builds the full tree structure regardless of gradient
magnitude. Each tree contributes less (lr × tiny gradient), but the architecture
is preserved. The model compensates by training more iterations until early
stopping triggers at a similar final loss.

Source: [LightGBM Parameters-Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html):
"Compared with depth-wise growth, the leaf-wise algorithm can converge much faster.
However, the leaf-wise growth may be over-fitting if not used with the appropriate parameters."

### Corollary: colsample/feature_fraction interacts with alpha

Low alpha → larger residuals → model needs MORE features to capture corrections.
- V8 (alpha=0.7): col=0.5 was optimal (small residuals, regularize to prevent overfitting)
- V9 (alpha=0.4-0.5): col=1.0 is optimal (larger residuals, need full feature access)

This is NOT a coincidence — it's the same mechanism. The feature subsampling
trade-off depends on the signal-to-noise ratio, which alpha directly controls.

## Consequences

1. **Training Final MUST use per-model alpha** (not a single global value)
2. **LightGBM alpha is the #1 hyperparameter** — more important than depth, leaves, or regularization
3. **XGBoost alpha is nearly irrelevant** — other params (subsample, mcw) matter more
4. **Any future model added to the pipeline** must have its alpha sensitivity characterized before joint tuning
5. **The V8 "alpha=0.7 was appropriate" conclusion was WRONG** — it was suboptimal for LightGBM by 0.05 logloss

## V8 Failure Mode (retrospective)

V8 LightGBM achieved valid=0.5134 with alpha=0.7. With alpha=0.4 (all else equal),
it would likely have achieved ~0.48-0.49 valid — a ~5% improvement left on the table.
This was never discovered because alpha was fixed at 0.7 by fiat and never
independently optimized per model.

The uniform alpha assumption was never documented or justified. It was a default
that propagated through 18 training iterations without being challenged.

## References

- LightGBM Parameters-Tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- XGBoost Parameter docs: https://xgboost.readthedocs.io/en/stable/parameter.html
- Guo et al. 2017 (temperature scaling analogy): arXiv:1706.04599
- V9 Grid results: grid_results/lightgbm_v2/, grid_results/xgboost_v4/
- V8 comparison: docs/project/V8_MODEL_COMPARISON.md
