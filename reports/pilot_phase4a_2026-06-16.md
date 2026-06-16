# Phase 4a Pilot — N3 saison 2024 (joint-conditional early gate)

## ⚠️ Controller verdict (2026-06-16) — MARGINAL / HOLLOW PASS — do NOT proceed to Kaggle T10 yet

The machine `early_gate_decision` below prints **PASS** because the literal floor
(mean recall ≥ 0.50) is cleared by **0.0036**. Read against the *spirit* of the
gate (does the joint-conditional mirror meaningfully improve adverse prediction
toward the recall ≥ 0.65 acceptance?), this is a **strategic FAIL**:

1. **No significant lift over Phase 3.** Phase 4a mean recall **0.5036** vs Phase 3
   baseline **0.4990** = **+0.0046**. **Wilcoxon p = 0.6088 (NOT significant).**
   In most of the 70 matches `recall == recall_baseline`; where they differ, the
   drain helps and hurts in roughly equal measure (e.g. Besançon T.P.G. 2 ronde 7:
   Phase 4a 0.2857 **vs** baseline 0.7143 — the top-down single max-Elo drain
   removed players who actually played). The MVP joint-conditional (top-down single
   max-Elo allocation + C5 mute only) **does not systematically beat plain Phase 3
   sampling**.
2. **29% CP-SAT timeout error rate biases the sample.** `error=32` of the 110
   attempted-viable matches (70 ran + 32 error + 8 thin_residual) are
   `RuntimeError: adverse CE infeasible/timeout … UNKNOWN` — the CP-SAT solver hit
   `max_time_sec=2.0` (Q7 complete_or_nothing hard-fails, no Phase 3 fallback),
   concentrated on big clubs (Strasbourg, Marseille, Montpellier, Lyon, Grenoble,
   Vandœuvre, Châlons, Tour Blanche). The 70 that ran are therefore **biased toward
   easy, small-pool, n_superior=1 instances** — exactly where draining matters least
   — so 0.5036 is an **optimistic** estimate. A 492-match Kaggle Phase A run would
   inherit this ~29% failure rate.
3. **Both arms are far below the recall ≥ 0.65 acceptance gate** (T10/T11).

**Recommendation: STOP before Kaggle.** Two blockers must be resolved first:
- **(A) CP-SAT robustness** (`max_time_sec`): raise the budget and/or make `UNKNOWN`
  degrade gracefully, then re-run the pilot for an **unbiased** N=70.
  → debt `D-2026-06-16-adverse-ce-cpsat-timeout`.
- **(B) Zero lift** → the MVP mirror does not improve adverse prediction. Decide
  between escalating to the **Phase 4c mixture** (K diverse preference-scored
  allocations — the shipped-but-orphan `preference_model.py` / `diversification.py`,
  Q5 contingency `D-2026-05-26-phase4c-joint-ortools-escalation` /
  `D-2026-06-16-adverse-allocation-mixture-preference-diversification`) vs
  reconsidering the adverse-mirror approach. **User decision required.**

The raw machine summary + per-match table follow unchanged.

## Summary

- Viable matches run: **70** (target 70)
- Mean recall (Phase 4a): **0.5036**
- Mean recall (Phase 3 baseline): 0.4990 (reference ~0.57)
- Skipped: non_viable=307, no_observed=0, thin_residual=8, error=32

> `thin_residual` = matches where draining the opponent's superior teams left too few players for 20 distinct lineups (ADR-014). A HIGH count is itself a Phase 4a finding (over-draining), not an error.
- **Wilcoxon (decisive):** p=0.6088, significant=False

> **Wilcoxon signed-rank (continuous recall) is the decisive test** (D-P3-18, SOTA). McNemar below dichotomizes recall at 0.50 and is a secondary view — a near-zero n_discordant means the two models rarely cross the 0.50 line on opposite sides, NOT an absence of effect.

- McNemar (secondary): p=1.0000, n_discordant=8

### Early-gate decision

**PASS (mean recall 0.5036 >= 0.5) -> proceed to Kaggle (T10)**

## Per-match detail

| opp_team | ronde | date | n_superior | recall | recall_baseline | jaccard | brier |
|---|---|---|---|---|---|---|---|
| Echiquier Nicois 2 | 3 | 2023-11-25 | 1 | 0.3750 | 0.3750 | 0.2308 | 0.2909 |
| Grasse Echecs 3 | 7 | 2024-01-28 | 1 | 0.3750 | 0.3750 | 0.2308 | 0.3175 |
| Echiquier Nicois 2 | 8 | 2024-03-16 | 1 | 0.3750 | 0.3750 | 0.2308 | 0.2866 |
| Orsay 2 | 1 | 2023-10-15 | 1 | 0.2500 | 0.2500 | 0.0667 | 0.4641 |
| Fontainebleau-Avon 2 | 1 | 2023-10-15 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.3067 |
| Juvisy 3 | 2 | 2023-11-12 | 1 | 0.1250 | 0.1250 | 0.0667 | 0.5156 |
| Orsay 2 | 2 | 2023-11-12 | 1 | 0.5000 | 0.3750 | 0.1429 | 0.4922 |
| J.E.E.N. - Paris 2 | 3 | 2023-11-25 | 1 | 0.6250 | 0.5000 | 0.3333 | 0.3458 |
| Orsay 2 | 4 | 2023-11-26 | 1 | 0.1250 | 0.1250 | 0.0667 | 0.4498 |
| Fontainebleau-Avon 2 | 4 | 2023-11-26 | 1 | 0.7500 | 0.7500 | 0.6000 | 0.2435 |
| Juvisy 3 | 5 | 2023-12-17 | 1 | 0.2500 | 0.2500 | 0.1429 | 0.4709 |
| Fontainebleau-Avon 2 | 5 | 2023-12-17 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.3648 |
| Fontainebleau-Avon 2 | 7 | 2024-01-28 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.2969 |
| Orsay 2 | 7 | 2024-01-28 | 1 | 0.1250 | 0.2500 | 0.0667 | 0.4488 |
| Fontainebleau-Avon 2 | 8 | 2024-03-16 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.3393 |
| J.E.E.N. - Paris 2 | 8 | 2024-03-16 | 1 | 0.6250 | 0.5000 | 0.3333 | 0.3366 |
| Orsay 2 | 9 | 2024-03-17 | 1 | 0.2500 | 0.3750 | 0.1429 | 0.4407 |
| Creteil 2 | 1 | 2023-10-15 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.2128 |
| Creteil 2 | 2 | 2023-11-12 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.2997 |
| Reims Echec et Mat 2 | 2 | 2023-11-12 | 1 | 0.7500 | 0.7500 | 0.6000 | 0.1698 |
| Drancy 2 | 4 | 2023-11-26 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.2864 |
| Creteil 2 | 4 | 2023-11-26 | 1 | 0.5000 | 0.6250 | 0.3333 | 0.2204 |
| Tour Blanche - Paris 2 | 5 | 2023-12-17 | 1 | 0.2500 | 0.2500 | 0.0667 | 0.3135 |
| Creteil 2 | 5 | 2023-12-17 | 1 | 0.6250 | 0.7500 | 0.4545 | 0.2298 |
| Reims Echec et Mat 2 | 5 | 2023-12-17 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.2818 |
| Reims Echec et Mat 2 | 6 | 2024-01-14 | 1 | 0.8750 | 0.8750 | 0.7778 | 0.0516 |
| Montreuil 3 | 7 | 2024-01-28 | 2 | 0.6250 | 0.5000 | 0.4545 | 0.1918 |
| Tour Blanche - Paris 2 | 7 | 2024-01-28 | 1 | 0.2500 | 0.3750 | 0.1429 | 0.3114 |
| Drancy 2 | 7 | 2024-01-28 | 1 | 0.3750 | 0.3750 | 0.2308 | 0.2823 |
| Creteil 2 | 8 | 2024-03-16 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.1860 |
| Montreuil 3 | 9 | 2024-03-17 | 2 | 0.7500 | 0.6250 | 0.6000 | 0.1622 |
| Drancy 2 | 9 | 2024-03-17 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.2261 |
| Reims Echec et Mat 2 | 9 | 2024-03-17 | 1 | 0.8750 | 0.8750 | 0.7778 | 0.0441 |
| Lille Universite Club 3 | 1 | 2023-10-15 | 1 | 0.3750 | 0.3750 | 0.2308 | 0.3494 |
| Marcq et Lys 2 | 2 | 2023-11-12 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.1616 |
| Lille Universite Club 3 | 3 | 2023-11-25 | 2 | 0.6250 | 0.3750 | 0.4545 | 0.3435 |
| Marcq et Lys 2 | 4 | 2023-11-26 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.1597 |
| Marcq et Lys 2 | 5 | 2023-12-17 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.2138 |
| Lille Universite Club 3 | 6 | 2024-01-14 | 1 | 0.6250 | 0.3750 | 0.3333 | 0.3581 |
| Marcq et Lys 2 | 7 | 2024-01-28 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.1745 |
| Marcq et Lys 2 | 8 | 2024-03-16 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.1799 |
| Lille Universite Club 3 | 9 | 2024-03-17 | 2 | 0.3750 | 0.2500 | 0.2308 | 0.3156 |
| Metz Fischer 3 | 1 | 2023-10-15 | 1 | 0.7500 | 0.7500 | 0.6000 | 0.1820 |
| Metz Fischer 3 | 2 | 2023-11-12 | 1 | 0.6250 | 0.6250 | 0.3333 | 0.2976 |
| Nancy Stanislas 2 | 4 | 2023-11-26 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.3378 |
| Metz Fischer 3 | 5 | 2023-12-17 | 1 | 0.5000 | 0.5000 | 0.2308 | 0.3105 |
| Metz Fischer 3 | 6 | 2024-01-14 | 1 | 0.6250 | 0.5000 | 0.3333 | 0.2613 |
| Nancy Stanislas 2 | 8 | 2024-03-16 | 1 | 0.6250 | 0.6250 | 0.3333 | 0.3792 |
| Metz Fischer 3 | 8 | 2024-03-16 | 1 | 0.7500 | 0.7500 | 0.4545 | 0.2340 |
| Bischwiller 3 | 1 | 2023-10-15 | 1 | 0.6250 | 0.6250 | 0.4545 | 0.2916 |
| Bischwiller 3 | 2 | 2023-11-12 | 1 | 0.7143 | 0.7143 | 0.5000 | 0.2678 |
| Mundolsheim 2 | 3 | 2023-11-25 | 1 | 0.3750 | 0.3750 | 0.1429 | 0.4401 |
| Mundolsheim 3 | 3 | 2023-11-25 | 2 | 0.3750 | 0.0000 | 0.1429 | 0.4465 |
| Mundolsheim 3 | 4 | 2023-11-26 | 2 | 0.3750 | 0.2500 | 0.1429 | 0.4552 |
| Mulhouse Philidor 3 | 4 | 2023-11-26 | 1 | 0.5000 | 0.5000 | 0.2308 | 0.2844 |
| Bischwiller 3 | 5 | 2023-12-17 | 1 | 0.3750 | 0.6250 | 0.2308 | 0.4018 |
| Bischwiller 3 | 6 | 2024-01-14 | 1 | 0.6250 | 0.6250 | 0.3333 | 0.3421 |
| Mundolsheim 2 | 7 | 2024-01-28 | 1 | 0.8750 | 0.8750 | 0.4545 | 0.2832 |
| Mundolsheim 3 | 7 | 2024-01-28 | 2 | 0.3750 | 0.2500 | 0.2308 | 0.4095 |
| Mulhouse Philidor 3 | 7 | 2024-01-28 | 1 | 0.2500 | 0.2500 | 0.1429 | 0.3232 |
| Mundolsheim 3 | 8 | 2024-03-16 | 2 | 0.3750 | 0.2500 | 0.2308 | 0.3132 |
| Mundolsheim 2 | 8 | 2024-03-16 | 1 | 0.5000 | 0.3750 | 0.3333 | 0.3194 |
| Bischwiller 3 | 8 | 2024-03-16 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.3597 |
| Mulhouse Philidor 3 | 8 | 2024-03-16 | 1 | 0.5000 | 0.5000 | 0.3333 | 0.2895 |
| Besancon T.P.G. 2 | 4 | 2023-11-26 | 1 | 0.3750 | 0.3750 | 0.2308 | 0.4847 |
| Besancon T.P.G. 2 | 7 | 2024-01-28 | 1 | 0.2857 | 0.7143 | 0.1538 | 0.4691 |
| Besancon T.P.G. 2 | 9 | 2024-03-17 | 1 | 0.3750 | 0.7500 | 0.2308 | 0.4896 |
| Nimes 2 | 7 | 2024-01-28 | 1 | 0.5000 | 0.6250 | 0.3333 | 0.3424 |
| Nimes 2 | 8 | 2024-03-16 | 1 | 0.3750 | 0.5000 | 0.2308 | 0.3604 |
| C.E.I. Toulouse 2 | 3 | 2023-11-25 | 1 | 0.2500 | 0.2500 | 0.1429 | 0.3472 |
