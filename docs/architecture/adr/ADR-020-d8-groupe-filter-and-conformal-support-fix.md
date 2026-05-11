# ADR-020 : D8 match identity extension `groupe` + conformal `support_max` fix

**Date** : 2026-05-11
**Statut** : Accepté (user explicit 2026-05-11 SOTA ML + ISO conforme)
**Standards** : ISO/IEC TR 5259:2024 (data lineage), ISO/IEC TR 24029-2:2024
(robustness via prediction intervals), ISO/IEC 27034 (input validation),
ISO/IEC 42010 (architecture decision record)

---

## Contexte

D8 Phase A push v12 (2026-05-10) sur 5 divisions saison 2024 (Top 16 + N1-N4).
Résultat empirique :
- 3/5 RUNNING → COMPLETE (N1/N2/N3)
- 2/5 ERROR (Top 16, N4)

Investigation 2026-05-11 a identifié 3 bugs structurels distincts qui compromettent
la validité de Phase A entière (y compris les 3 COMPLETE) :

### Bug 1 — Identité match incomplète (Top 16 multi-phase)

**Régulation FFE Top 16** = 2 phases séquentielles :
- Phase 1 (rondes 1-7) : Groupes A + B parallèles, qualification
- Phase 2 (rondes 1-4) : Poule Haute (titre) + Poule Basse (maintien)

Une équipe qualifiée (ex Bischwiller saison 2024) apparaît dans 2 groupes
distincts (`Groupe B` rondes 1-7 + `Poule Haute` rondes 1-4). Les rondes
restart à 1 dans la Phase 2. Donc le tuple `(saison, ronde, equipe_dom,
equipe_ext)` n'est **PAS unique** pour Top 16.

`scripts/backtest/ground_truth.py::_select_match_rows` filtrait sur
`(saison, ronde, club)` sans `groupe`. Conséquence : 16 boards mergées
provenant de 2 matches distincts (Phase 1 + Phase 2) → `odd_blancs` contient
3 équipes au lieu de 2 → `_validate_ffe_color_invariant` trip
`FFEDataQualityError` (D-P3-14) → match skipped → Top 16 kernel n'a que 12
valid matches < 31 conformal threshold → ERROR.

**Le bug n'est PAS dans la donnée** (parquet `data/echiquiers.parquet` est
correct, distingue les 4 groupes via colonne `groupe`). **Le bug est dans le
code** (filtre incomplet).

### Bug 2 — Conformal `set_size` saturé par clip [0, 1.0]

`scripts/d8/conformal.py::conformal_set_size_mean` calculait :
```python
upper = np.minimum(y_predicted + q, 1.0)   # clip à 1.0
lower = np.maximum(y_predicted - q, 0.0)
sizes = upper - lower
```

Or pour D8 audit, `y_predicted = e_score_predicted` ∈ [0, K] où K = `team_size`
= 8 (8 boards par match Interclubs Open). Le clip [0, 1.0] saturait
artificiellement `set_size_mean = 1.0` pour toute valeur `q_hat ≥ 1.0`.

Outputs Phase A N1/N2/N3 saison 2024 : `q_hat` ∈ [4.4, 5.1] → `set_size_mean
= 1.00` partout (faux). Gate G_ROB_07 (set_size ≤ 3.0) PASS trivialement
non-discriminant.

**Conséquence ISO 24029 §5.3** : robustness validation via prediction
intervals compromise. Les 3 outputs COMPLETE ne reflètent pas la vraie
efficiency Angelopoulos & Bates 2023 §4.2.

### Bug 3 — `RunnerConfig.max_matches=50` hardcoded trop bas

Pour divisions à taux de validation post-filter faible (~24% N4 saison 2024),
le cap 50 × 0.24 ≈ 12-19 valid < 31 conformal threshold → RuntimeError.

Pas un bug strict mais limite non-paramétrable. Phase A demande override
sans changer le default (préserver R-PRE-PUSH-01 hook fast tests).

---

## Décision

### Fix A — Match identity étendue `(saison, division, groupe, ronde, dom, ext)`

`scripts/backtest/runner_types.py::MatchCandidate` :
- Ajouter `groupe: str = ""` (default empty pour backward compat).

`scripts/backtest/runner_sampling.py::enumerate_candidates` :
- Inclure `groupe` dans dedup key.
- Propager depuis `echiquiers.parquet` (colonne `groupe`).
- Resilient à colonne absente (tests fixtures + données historiques).

`scripts/backtest/runner.py::run_single` :
- Accepter `groupe: str = ""` param.
- `run()` propage `cand.groupe` depuis `MatchCandidate`.

`scripts/backtest/ground_truth.py::extract_observed_lineup` + `_select_match_rows` :
- Accepter `groupe: str = ""` param.
- Si non-empty, filter `sub[sub["groupe"] == groupe]` avant invariant check.

### Fix B — Conformal `support_max` paramétrable

`scripts/d8/conformal.py::conformal_set_size_mean` :
- Ajouter `support_max: float = 1.0` param.
- Validation ISO 27034 : `support_max > 0`.
- Clip à `[0, support_max]` au lieu de `[0, 1.0]`.

`scripts/d8/run.py::_compute_conformal_stage` :
- Accepter `support_max: float = 8.0` (Phase A team_size=8).
- Output JSON nouveau field `set_size_relative = set_size / support_max`
  + `support_max` pour traçabilité cross-division.

### Fix C — `ALICE_MAX_MATCHES` env var

`scripts/backtest/runner_types.py` :
- `_max_matches_default()` lit `ALICE_MAX_MATCHES` (default 50, validation
  int ≥ 1, ValueError sinon).
- `RunnerConfig.max_matches: int = field(default_factory=_max_matches_default)`.

Wrappers Phase A `scripts/d8/run_2024_*.py` :
- `os.environ.setdefault("ALICE_MAX_MATCHES", "200")` (5 wrappers).

---

## Conséquences

### Code

| Fichier | Δ Lignes | Type |
|---------|----------|------|
| `scripts/backtest/runner_types.py` | +18 -1 | MatchCandidate.groupe + env var helper |
| `scripts/backtest/runner_sampling.py` | +15 -3 | enumerate_candidates groupe propagation |
| `scripts/backtest/runner.py` | +5 -2 | run_single + run() groupe plumbing |
| `scripts/backtest/ground_truth.py` | +6 -2 | extract_observed_lineup groupe filter |
| `scripts/d8/conformal.py` | +14 -7 | support_max parameter |
| `scripts/d8/run.py` | +6 -2 | _compute_conformal_stage support_max |
| `scripts/d8/run_2024_{5 divisions}.py` | +3×5 | ALICE_MAX_MATCHES=200 |
| `tests/d8/test_conformal.py` | +50 | 4 nouveaux tests support_max |
| `tests/backtest/test_ground_truth.py` | +65 | 3 nouveaux tests groupe filter |
| `tests/backtest/test_max_matches_env.py` | +44 NEW | 5 nouveaux tests env var |

**Total** : ~225 lignes ajoutées, 17 supprimées.

### Tests

- 49 tests scope ciblé PASS : 32 conformal + 5 env var + 12 runner_stratified
- 9 ground_truth (slow, requires parquets) PASS dont 3 nouveaux
- 3 test_runner E2E PASS (signature change `run_single(groupe=)` compatible)
- Ruff PASS, mypy --strict PASS sur les 6 fichiers modifiés

### Backward compatibility

- `MatchCandidate.groupe: str = ""` → tests legacy continuent à passer
- `extract_observed_lineup(groupe="")` → comportement identique pre-fix
- `_max_matches_default()` returns 50 par défaut (sans env var)
- `conformal_set_size_mean(support_max=1.0)` → comportement identique pre-fix

### Re-execution Phase A

Les 3 outputs COMPLETE actuels (N1/N2/N3) sont à **invalider** :
- `set_size_mean = 1.0` artificiellement saturé
- Gate G_ROB_07 non-discriminant

Phase A v3 doit re-pusher les 5 kernels après commit des fixes.

### Source de vérité

- Régulation Top 16 : règlement FFE A02 §3.7 + Top 16 saison 2024
- Conformal SOTA : Vovk 2024 *Algorithmic Learning in a Random World* §2.3 +
  Angelopoulos & Bates 2023 *Gentle Intro CP* §4.2 (efficiency relative)
- Data lineage : ISO/IEC TR 5259:2024 §4.2 (identité unique observation)
- Robustness : ISO/IEC TR 24029-2:2024 §5.3 (prediction intervals)

---

## Alternatives évaluées

| Alternative | Verdict | Raison |
|-------------|---------|--------|
| Skip Top 16 saison 2024 (exclude) | Rejeté | Phase A spec §1.3 exige 5 divisions multi-niveau (R-ALI-01 capture). Top 16 = niveau ≥2300 critical. |
| Warn-and-skip ronde corrompue | Rejeté | ISO 5259 data lineage : silently dropping data = produit non-trustworthy commercial. |
| Bypass conformal pour <31 valid | Rejeté | ISO 24029 §5.3 robustness guarantee perdue. SOTA Vovk 2024 finite-sample valable. |
| Jackknife+ (Barber 2021) | Reporté | Plus complexe que split conformal. Phase B si N reste serré post-fix. |
| Hardcode max_matches=200 dans RunnerConfig | Rejeté | Casse R-PRE-PUSH-01 (slow tests). Env var pattern = clean isolation. |
| **Match identity étendue + support_max + env var** | **Accepté** | Trois fixes orthogonaux, ISO conforme, backward compat, testable. |

---

## Sources

- Vovk, V. et al. 2024 *Algorithmic Learning in a Random World* 2nd ed., §2.3 (split conformal marginal coverage)
- Angelopoulos, A. & Bates, S. 2023 *A Gentle Introduction to Conformal Prediction* (Foundations and Trends in ML 16:4) §4.2 (efficiency)
- ISO/IEC TR 5259:2024 — Data quality + lineage
- ISO/IEC TR 24029-2:2024 — Robustness of neural networks §5.3
- ISO/IEC 27034 — Input validation guidelines
- ISO/IEC 42010 — Architecture description (ADR format)
- Règlement FFE Top 16 saison 2024 (4 groupes : A/B + Poule Haute/Basse)
