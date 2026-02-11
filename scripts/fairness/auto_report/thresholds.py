"""Seuils fairness partages - ISO 24027 + EEOC + NIST.

Constantes nommees pour les seuils de fairness utilises
par le generateur (status) et le formateur (affichage).

ISO Compliance:
- ISO/IEC TR 24027:2021 - Bias in AI systems
- EEOC 80% Rule (Demographic Parity)
- NIST AI 100-1 MEASURE 2.11 - Fairness evaluation
- ISO/IEC 5055:2021 - Code Quality (DRY, SRP)

Author: ALICE Engine Team
Last Updated: 2026-02-11
"""

from __future__ import annotations

# --- Demographic Parity (EEOC 80% rule) ---
EEOC_THRESHOLD = 0.80
CAUTION_DP_THRESHOLD = 0.85

# --- TPR Diff (Equalized Odds) ---
TPR_DIFF_CRITICAL = 0.20
TPR_DIFF_CAUTION = 0.10

# --- FPR Diff (Equalized Odds) ---
FPR_DIFF_CRITICAL = 0.20
FPR_DIFF_CAUTION = 0.10

# --- Predictive Parity Diff ---
PP_DIFF_CRITICAL = 0.20
PP_DIFF_CAUTION = 0.10

# --- Min Group Accuracy ---
MIN_ACC_CRITICAL = 0.50
MIN_ACC_CAUTION = 0.60

# --- Calibration Gap (NIST) ---
CAL_GAP_CRITICAL = 0.20
CAL_GAP_CAUTION = 0.10
