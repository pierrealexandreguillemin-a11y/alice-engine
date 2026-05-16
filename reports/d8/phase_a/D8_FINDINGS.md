# D8 Findings — Phase 3.5 STRICT
**Run** : 2026-05-16T13:55:59.654906+00:00
**N matches** : 492
**Saisons** : [2024]
**Gates G-A** : 6/19 PASS, 13 FAIL, 0 INCONCLUSIVE

## Per-gate results

- **G_FAIR_01_max_gap_recall** — FAIL (measured=0.2258, threshold=0.1000, source=Mehrabi 2021 §4.1)
- **G_FAIR_02_recall_per_group_min** — FAIL (measured=0.5159, threshold=0.8500, source=P3G07 - 5pts)
- **G_FAIR_03_demographic_parity_diff** — PASS (measured=0.0000, threshold=0.1000, source=Hardt 2016)
- **G_FAIR_04_equalized_odds_diff** — PASS (measured=0.0000, threshold=0.1000, source=Hardt 2016 §3.2)
- **G_FAIR_05_calibration_ECE_per_group** — FAIL (measured=0.4958, threshold=0.0500, source=Pleiss 2017 §4)
- **G_FAIR_06_multicalibration_alpha** — FAIL (measured=0.4958, threshold=0.0500, source=Hébert-Johnson 2018)
- **G_FAIR_07_TPR_ratio_min** — PASS (measured=1.0000, threshold=0.8000, source=EEOC §1607.4D + Feldman 2015)
- **G_FAIR_08_brier_per_group** — FAIL (measured=0.4605, threshold=0.3000, source=Brier 1950 + Pappalardo 2019)
- **G_FAIR_09_BSS_per_group** — PASS (measured=0.3942, threshold=0.3000, source=Pappalardo 2019 §3.4)
- **G_FAIR_10_PSI_per_dim** — PASS (measured=0.0000, threshold=0.2000, source=Yurdakul 2020)
- **G_ROB_01_recall_drop_1pct** — FAIL (measured=0.1186, threshold=0.0200, source=Goodfellow 2015 ε=0.01)
- **G_ROB_02_recall_drop_5pct** — FAIL (measured=0.1390, threshold=0.0500, source=Madry 2018)
- **G_ROB_03_recall_drop_10pct** — FAIL (measured=0.1105, threshold=0.1000, source=Madry 2018 strict)
- **G_ROB_04_roster_5pct** — FAIL (measured=0.1443, threshold=0.0500, source=Tran 2022 §3.4)
- **G_ROB_05_roster_20pct** — FAIL (measured=0.1996, threshold=0.1500, source=Recht 2019 §5)
- **G_ROB_06_conformal_coverage_90** — PASS (measured=0.9486, threshold=0.9000, source=Vovk 2024 §2.3)
- **G_ROB_07_conformal_set_size_max** — FAIL (measured=6.0429, threshold=3.0000, source=Angelopoulos 2023 §4.2)
- **G_ROB_08_DRO_eps_005_min** — FAIL (measured=0.0000, threshold=0.7000, source=Sinha 2018 §4)
- **G_ROB_09_DRO_eps_010_min** — FAIL (measured=0.0000, threshold=0.5500, source=Duchi 2021 §6)

## Phase 4a entry gate

WARN 13 gates FAIL — see D8_FAILURE_ANALYSIS_LOG.md for case-by-case decisions.
