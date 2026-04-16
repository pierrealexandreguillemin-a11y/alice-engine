# V8 MultiClass 3-Way (Win/Draw/Loss) — Design Spec

**Date**: 2026-03-21
**Status**: Draft
**Replaces**: V7 binary classification (leakage, target bug, calibration failure)
**ISO Compliance**: 5055, 5259, 24027, 24029, 25059, 42001
**Standards**: Pawnalyze, ChessEngineLab, penaltyblog 2025, Niculescu-Mizil & Caruana

## Problem

V7 binary model (win vs draw+loss) had 4 critical bugs and 8 feature logic errors:

### Critical bugs
1. **Leakage**: `score_dom`/`score_ext` = final match score in features (AUC inflated 0.8275 → real ~0.72). Fixed commit 05b19a7.
2. **Calibration**: eval_metric=AUC → log_loss worse than baseline (0.8067 > 0.6785). Fixed same commit.
3. **Target bug**: `resultat_blanc=2.0` (forfaits FFE, 4.2% data) coded as losses, are actually wins.
4. **Architecture mismatch**: CE (`composer.py:36-39`) expects `(win_prob, draw_prob, loss_prob)` but binary model only produces P(win).

### Feature logic errors
1. `clutch_factor` uses `score_dom` (circular leakage with match score)
2. `score_blancs/noirs` confounds color with home advantage (board-number-determined)
3. Player features mix ALL competitions (national 20.9% draws ≠ regional 9.6%)
4. Player features use global career instead of rolling window
5. `echiquier_moyen` global → misleading for players who progressed
6. Forfaits (2.0) counted in ALL rate calculations
7. H2H has 0.8% coverage (2867 pairs ≥3 games) — nearly useless
8. 9 features planned in docs but never implemented (vases communiquants, categorie_age)

### Business impact
Draws = 12.6% of games. Draw rate varies 4.9% (Elo<1200) to 45.8% (Elo>2400). Ignoring draws costs ~0.50 points per match (8 boards). CE cannot distinguish safe compositions (many draws) from risky ones (all-or-nothing) without P(draw).

## Architecture: Predict-Then-Optimize

Standard architecture confirmed by FPL 2025 (arxiv 2505.02170):

```
V8 (this spec):
  Feature engineering → ML MultiClass → P(win), P(draw), P(loss) per board
  Calibration → Baselines → Quality gate → CANDIDATE model

V9 (future):
  ALI predicts opponent lineup (Monte Carlo 20 scenarios)
  → V8 model evaluates each (player × board × opponent) combination
  → CE allocates players across N teams via OR-Tools (integer programming)
  → Strategy mode (aggressive/conservative/tactical) sets objective function
```

V8 builds the prediction foundation. V9 builds the optimization brain.

## Target Variable

| Value | Meaning | Encoding | Count (train) | % |
|-------|---------|----------|---------------|---|
| 0.0 | Loss (black wins) | 0 (loss) | 469,841 | 43.1% |
| 0.5 | Draw | 1 (draw) | 155,585 | 14.3% |
| 1.0 | Win (white wins) | 2 (win) | 464,724 | 42.6% |
| 2.0 | Forfait (white wins administratively) | **EXCLUDED** | 49,649 | — |

Forfaits excluded from target AND all feature computations. No predictive signal (administrative event).

After exclusion: 1,090,150 train rows. 3-class ordinal: loss=0, draw=1, win=2.

## Models

| Framework | Objective | eval_metric | Params | GPU |
|-----------|-----------|-------------|--------|-----|
| CatBoost | `MultiClass` (softmax) | `MultiClass` (log loss) | depth=8, iter=3000, border_count=254, lr=auto, use_best_model=True | ✓ Native CUDA |
| XGBoost | `multi:softprob` | `mlogloss` | max_depth=8, n_estimators=3000 | ✓ cuda |
| LightGBM | `multiclass` | `multi_logloss` | num_leaves=255, n_estimators=3000 | CPU only |

All produce (n_samples, 3) matrix: P(loss), P(draw), P(win).

No class weights (degrades calibration in GBMs — Niculescu-Mizil & Caruana 2005).

CatBoost has built-in `probability_calibration=True` for MultiClass (catboost.ai docs).

## Calibration

Isotonic regression per class + renormalization (sum=1).

Justification: isotonic cuts ECE by 50% for GBMs (vs temperature scaling 33%, Platt 25%). Non-parametric, handles non-linear calibration curves. 70K valid set = no overfit risk.

Pipeline:
1. Train CatBoost with `loss_function='MultiClass'` (native `probability_calibration=True` is ON by default during training)
2. After training: get raw probas from `predict_proba(X_valid)` → shape (n, 3)
3. Fit `IsotonicRegression` per class (3 calibrators) on valid set
4. At evaluation: apply isotonic → renormalize each row to sum=1 → save `calibrators.joblib`

## Evaluation Metrics & Quality Gate

### Baselines
- **Naïve**: always predict marginal distribution (43.1%/14.3%/42.6%)
- **Elo + draw rate model**: Elo formula + Pawnalyze draw rate lookup (elo_band × diff_band)

### Metrics (ordered by priority)

| # | Metric | Type | Why |
|---|--------|------|-----|
| 1 | **MultiClass log loss** | Calibration (primary) | Most sensitive, recommended penaltyblog 2025 |
| 2 | **E[score] MAE** | System metric | Direct error on CE input: E[score] = P(win) + 0.5×P(draw) |
| 3 | **Brier multiclass** | Calibration (secondary) | More robust to outliers than log loss |
| 4 | **RPS** | Ordinal quality | Captures W>D>L ordering (relevant for CE), known non-locality weakness |
| 5 | **ECE per class** | Calibration diagnostic | Reliability diagram must be quasi-diagonal |
| 6 | **Reliability diagrams** | Visual | Per-class calibration curves |

### Quality gate (ALL conditions must pass)

| Condition | Threshold |
|-----------|-----------|
| MultiClass log loss < baseline naïve AND < baseline Elo | Mandatory |
| E[score] MAE < baseline Elo MAE | Mandatory |
| Brier multiclass < baseline naïve | Mandatory |
| RPS < baseline naïve AND < baseline Elo | Mandatory |
| ECE per class < 0.05 | V8 gate. Production target: 0.015 (sports-ai.dev). Tighten when serving in V9. |
| Calibration: mean(P(draw)) ≈ observed draw rate ±2% | Systematic draw bias check |

NOT in quality gate (irrelevant for probability-based optimizer): F1, accuracy, draw recall. Reported for information only.

## Feature Engineering

### Principles
- ALL features derived from results → decomposed into (win_rate, draw_rate)
- ALL player features → stratified by `type_competition` (national ≠ regional ≠ jeunes)
- ALL player features → rolling window (3 seasons for career stats, 5 games for form, last season for board position)
- Forfaits (resultat_blanc=2.0) excluded from ALL rate computations
- Features available at inference: own team always, opponent team after ALI prediction

### Category 1 — Match context (12 features, unchanged)

`echiquier`, `est_domicile_blanc`, `saison`, `ronde`, `type_competition`, `division`, `ligue_code`, `niveau`, `phase_saison`, `ronde_normalisee`, `match_important`, `jour_semaine`

All draw-neutral. No changes needed.

### Category 2 — Player strength (10 columns: 6 existing + 4 new)

Existing (6): `blanc_elo`, `noir_elo`, `diff_elo`, `blanc_titre_num`, `noir_titre_num`, `diff_titre`

New (planned in FEATURE_SPEC §3, never implemented) (4):
- `categorie_blanc`, `categorie_noir`: FFE age category (U08-X65). Join from joueurs.parquet on NrFFE (player FFE ID).
- `k_coefficient_blanc`, `k_coefficient_noir`: FIDE K-factor from category + Elo + nb games. Juniors K=40 → more variance.

### Category 3 — Player form W/D/L (20 features, refactored from 12)

All stratified by `type_competition`, rolling window:

| Old feature | New features | Window | Stratified |
|-------------|-------------|--------|------------|
| `forme_recente_B/N` | `win_rate_recent_B/N`, `draw_rate_recent_B/N`, `expected_score_recent_B/N` | 5 games | ✓ same level |
| `forme_tendance_B/N` | `win_trend_B/N`, `draw_trend_B/N` | 5 games | ✓ |
| `score_blancs_B/N` | `win_rate_white_B/N`, `draw_rate_white_B/N` | 3 seasons | ✓ |
| `score_noirs_B/N` | `win_rate_black_B/N`, `draw_rate_black_B/N` | 3 seasons | ✓ |
| `avantage_blancs_B/N` | `win_adv_white_B/N`, `draw_adv_white_B/N` | 3 seasons | ✓ |

Color/home confound: model learns interaction via `est_domicile_blanc` × color features. No manual home/away split within color (too sparse per player).

### Category 4 — Draw priors (8 columns, all new)

| Feature | Source | Window | Columns |
|---------|--------|--------|---------|
| `avg_elo` | (blanc_elo + noir_elo) / 2 | Per game | 1 |
| `elo_proximity` | 1 - min(\|diff_elo\|, 800)/800 | Per game | 1 |
| `draw_rate_prior` | Lookup (elo_band × diff_band) on train. Bins: 200-point Elo bands, 100-point diff bands | Train only | 1 |
| `draw_rate_blanc`, `draw_rate_noir` | Player historical draw rate | 3 seasons, same level | 2 |
| `h2h_draw_rate` | H2H draw rate (≥3 confrontations, else NaN). 0.8% coverage — same as Cat 7 `h2h_draw_rate`. Kept with `h2h_exists` flag for model to learn when to trust it | Global | 1 |
| `draw_rate_equipe_dom`, `draw_rate_equipe_ext` | Team draw rate | 3 seasons | 2 |

Fallback: players <10 games at same level → use `draw_rate_prior` (Elo band). H2H <3 confrontations → NaN (CatBoost handles natively).

### Category 5 — Presence/availability (8 concepts = 16 columns _B/_N, existing code to wire up)

From `ali/presence.py` (code exists, not wired): `taux_presence_saison_B/N` (2), `derniere_presence_B/N` (2), `regularite_B/N` (2).
From `pipeline_extended.py` (existing): `rondes_manquees_consecutives_B/N` (2), `taux_presence_global_B/N` (2).
From `ali/patterns.py` (code exists, not wired): `role_type_B/N` (2), `echiquier_prefere_B/N` (2), `flexibilite_echiquier_B/N` (2).

### Category 6 — Pressure performance (6 features, corrected)

| Old | New | Fix |
|-----|-----|-----|
| `clutch_factor_B/N` (used `score_dom`) | `clutch_win_B/N`, `clutch_draw_B/N` | `is_decisive = zone_enjeu IN (montee, danger)` — no match score leakage |
| `pressure_type_B/N` | `pressure_type_B/N` | Recalculated from corrected clutch |

Rolling 3 seasons, stratified by level.

### Category 7 — Head-to-head (4 features, refactored)

`h2h_win_rate`, `h2h_draw_rate` (decomposed from h2h_avantage), `h2h_nb_confrontations`, `h2h_exists` (bool flag ≥3 games).

Coverage: 0.8% of rows. Global window (confrontations too rare for rolling). Available only after ALI at inference.

### Category 8 — Team standings (8 concepts = 16 columns _dom/_ext, existing, draw-aware)

`position` (2), `ecart_premier` (2), `ecart_dernier` (2), `points_cumules` (2), `nb_equipes` (2), `zone_enjeu` (2), `niveau_hierarchique` (2), `adversaire_niveau` (2) = 16 columns.

Already draw-aware (points_cumules: +2 win, +1 draw, +0 loss). No changes.

### Category 9 — Club behavior (8 concepts = 16 columns _dom/_ext: 6 existing to wire + 2 refactored)

Wire up existing code from `club_behavior.py`:
`nb_joueurs_utilises_dom/ext`, `rotation_effectif_dom/ext`, `noyau_stable_dom/ext`, `profondeur_effectif_dom/ext`, `club_utilise_marge_100_dom/ext`, `renforce_fin_saison_dom/ext`.

Refactored (W/D/L decomposition):
`win_rate_home_dom/ext`, `draw_rate_home_dom/ext` (from `avantage_dom_club`, rolling 3 seasons).

### Category 10 — Vases communiquants (8 concepts = 16 columns, all new)

Planned in TRAINING_PROGRESS §5.2 and REGLES_FFE_ALICE.md §4, never implemented:

| Feature | Computation | Suffix | Columns | Doc source |
|---------|-------------|--------|---------|------------|
| `joueur_promu` | Playing for higher-level team than primary | _B/_N | 2 | TRAINING_PROGRESS §5.2 |
| `joueur_relegue` | Playing for lower-level team (reinforcement) | _B/_N | 2 | TRAINING_PROGRESS §5.2 |
| `player_team_elo_gap` | Player Elo − team average Elo | _B/_N | 2 | New |
| `stabilite_effectif` | % identical players vs season N-1 | _dom/_ext | 2 | TRAINING_PROGRESS §5.2 |
| `elo_moyen_evolution` | Team Elo delta R1 → current round | _dom/_ext | 2 | TRAINING_PROGRESS §5.2 |
| `team_rank_in_club` | Rank in club hierarchy (1=fanion) | _dom/_ext | 2 | New |
| `reinforcement_rate` | % rounds with inter-team reinforcement | _dom/_ext | 2 | New |
| `club_nb_teams` | Number of teams in club | _dom/_ext | 2 | New |

Uses `get_niveau_equipe()` from `REGLES_FFE_ALICE.md` §4.2 (already specified).

### Category 11 — FFE regulatory (10 concepts = 20 columns, existing, unchanged)

Player-level (_B/_N): `est_dans_noyau` (2), `ffe_nb_equipes` (2), `ffe_niveau_max` (2), `ffe_niveau_min` (2), `ffe_multi_equipe` (2), `joueur_fantome` (2) = 12 columns.
Team-level (_dom/_ext): `pct_noyau_equipe` (2), `taux_forfait` (2), `taux_non_joue` (2), `fiabilite_score` (2) = 8 columns.
Total: 20 columns.

All draw-neutral. No changes.

### Category 12 — Elo trajectory (4 features, existing, unchanged)

`elo_trajectory_B/N`, `momentum_B/N`. Rolling 6 games. Draw-neutral.

### Category 13 — Composition strategy (8 features, existing, minor fix)

`decalage_position_B/N`, `joueur_decale_haut/bas_B/N`. Unchanged.
`echiquier_moyen_B/N`, `echiquier_std_B/N`: **change from global to rolling last season**.

### Feature count summary (exact column count)

Counting actual model input columns (each _B/_N or _dom/_ext suffix = 1 column):

| Category | Concepts | Columns | Status |
|----------|----------|---------|--------|
| 1. Match context | 12 | 12 | Unchanged |
| 2. Player strength | 5+4 new | 10 | +4 new (categorie, K) |
| 3. Player form W/D/L | 10 concepts | 20 (_B/_N) | Refactored from 12 old columns |
| 4. Draw priors | 8 | 8 | All new |
| 5. Presence/availability | 8 concepts | 16 (_B/_N) | Wire existing code |
| 6. Pressure | 3 concepts | 6 (_B/_N) | Fixed (zone_enjeu) |
| 7. H2H | 4 | 4 | Refactored W/D/L |
| 8. Standings | 8 concepts | 16 (_dom/_ext) | Unchanged |
| 9. Club behavior | 8 concepts | 16 (_dom/_ext) | Wire + refactor W/D/L |
| 10. Vases communiquants | 8 concepts | 16 (_B/_N or _dom/_ext) | All new |
| 11. FFE regulatory | 10 concepts | 20 (_B/_N or _dom/_ext) | Unchanged |
| 12. Elo trajectory | 2 concepts | 4 (_B/_N) | Unchanged |
| 13. Composition strategy | 4 concepts | 8 (_B/_N) | Minor fix (rolling) |
| **Total** | | **~156 columns** | |

Before V8: ~128 columns (incl. 2 leaky removed). After V8: ~156 columns. Net +28 (new features > replaced features). The increase comes from W/D/L decomposition (+8), draw priors (+8), vases communiquants (+16), strength (+4), minus replaced features (-8).

### Temporal split

- **Train**: saisons ≤ 2022 (~1,090,150 rows after forfait exclusion)
- **Valid**: saison 2023 (~67,824 rows)
- **Test**: saisons ≥ 2024 (~221,807 rows)

Features computed from data STRICTLY BEFORE each split (no temporal leakage). Confirmed in `feature_engineering.py`.

### score_dom/score_ext removal

`score_dom`/`score_ext` (final match scores) are removed at the `_split_xy()` level in `kaggle_trainers.py` (commit 05b19a7, `LEAKY_COLUMNS`). They remain in the raw data for standings computation (which is correct — standings use cumulative scores from PREVIOUS rounds, not the current match). The pressure fix (Category 6) additionally removes the `|score_dom - score_ext| <= 1` condition from `is_decisive`.

## Scope

### IN scope (V8)

| Layer | Files | Change |
|-------|-------|--------|
| Features | `scripts/features/draw_priors.py` | NEW — 8 draw features |
| Features | `scripts/features/club_level.py` | NEW — 10 vases communiquants features |
| Features | `scripts/features/recent_form.py` | REFACTOR — W/D/L + stratification + rolling |
| Features | `scripts/features/color_perf.py` | REFACTOR — W/D/L + rolling 3 seasons |
| Features | `scripts/features/advanced/pressure.py` | FIX — zone_enjeu instead of score_dom |
| Features | `scripts/features/advanced/h2h.py` | REFACTOR — W/D/L + h2h_exists flag |
| Features | `scripts/features/club_behavior.py` | REFACTOR — W/D/L + wire up |
| Features | `scripts/features/performance.py` | FIX — echiquier_moyen rolling last season |
| Features | `scripts/feature_engineering.py` | INTEGRATE — all new features in pipeline |
| Features | `scripts/features/pipeline.py` | INTEGRATE — wire ali/presence, ali/patterns |
| Training | `scripts/kaggle_trainers.py` | MultiClass target, loss, eval, predict_proba 3-col |
| Training | `scripts/cloud/train_kaggle.py` | Baselines (naïve + Elo), RPS, system metrics |
| Training | `scripts/cloud/train_autogluon_kaggle.py` | MultiClass config (ELIMINE -- ADR-011) |
| Diagnostics | `scripts/kaggle_diagnostics.py` | RPS, ECE, reliability diagrams, calibration 3-class |
| Config | `config/hyperparameters.yaml` | MultiClass params, quality gate thresholds |
| Tests | `tests/test_cloud_*.py` | Adapt to new metrics |
| Upload | `scripts/cloud/upload_all_data.py` | Re-upload after changes |

### OUT of scope (V9)

| What | Why |
|------|-----|
| `services/composer.py` | CE already expects W/D/L. Wiring = V9 |
| `services/inference.py` | ALI = V9 |
| `app/api/routes.py` | API stubs = V9 |
| OR-Tools multi-team allocation | V9 |
| Strategy modes (aggressive/conservative) | V9 |
| `scripts/parse_dataset/` | Parsing unchanged |
| `scripts/sync_data/` | Data sync unchanged |

## V9 Preview (for documentation)

V9 builds on V8's P(W/D/L) predictions to solve the club-level multi-team allocation problem:

```
Input: Club with N teams, ~30-50 available players, N opponent teams
ALI: For each match, predict 20 opponent lineup scenarios
ML (V8): For each scenario × each (player, board, team), compute P(W/D/L)
CE: Solve allocation via integer programming under FFE constraints
Output: Recommended composition for EACH team + alternatives
```

Strategy modes (user-selectable):
- **Aggressive**: maximize E[score] for priority team, minimum threshold for others
- **Conservative**: maximize min(E[score]) across all teams (maximin)
- **Tactical**: maximize P(match victory) for team in promotion/relegation zone
- **Risk-adjusted**: maximize E[score] - λ×Var[score] (P(draw) vs P(win) critical here)

FFE constraints:
- Each player in ONE team per weekend
- Noyau locked after first game (A02 3.7.f)
- Max 3 mutés per team per season (A02 3.7.g)
- Elo order within team (100pts, A02 3.6.e)

## Standards & Sources

| Standard | Application | Source |
|----------|-------------|--------|
| Predict-then-optimize | Architecture | FPL arxiv 2505.02170 |
| Three-outcome W/D/L | Target encoding | lichess jk_182, ChessEngineLab |
| Draw rate = f(Elo level, Elo diff) | Draw priors | Pawnalyze (5M OTB games) |
| Isotonic calibration for GBMs | Post-hoc calibration | Niculescu-Mizil & Caruana (Cornell) |
| Log loss primary > RPS | Metric hierarchy | penaltyblog 2025 |
| ECE < 0.015 (production) | Calibration monitoring | sports-ai.dev |
| CatBoost MultiClass calibration | Built-in probability_calibration | catboost.ai docs |
| No class weights for GBMs | Degrades calibration | Niculescu-Mizil & Caruana |
| Springer 2024 Soccer | MultiClass 3-way standard | ScienceDirect |
| RPS ordinal metric | Captures W>D>L ordering | penaltyblog docs |

## Data Facts (verified on train set)

| Fact | Value |
|------|-------|
| Train rows (after forfait exclusion) | ~1,090,150 |
| Valid rows | ~67,824 |
| Test rows | ~221,807 |
| Draw rate by Elo: <1200 vs >2400 | 4.9% vs 45.8% (×9.4) |
| Draw rate by competition: national vs regional | 20.9% vs 9.6% (×2.2) |
| Draw rate by Elo diff: <50 vs >400 | 18.4% vs 4.3% (×4.3) |
| Draw rate by board: board 1 vs board 8 | 16.3% vs 9.5% |
| White HOME vs AWAY win rate | 0.570 vs 0.409 |
| H2H pairs ≥3 confrontations | 0.8% coverage |
| Multi-team player-seasons | 30.7% |
| Same player national vs regional score | 0.435 vs 0.607 (corr 0.196) |
| Clubs with 3+ teams | 10% |
| Player movers avg Elo vs non-movers | 1538 vs 1189 (+349) |
