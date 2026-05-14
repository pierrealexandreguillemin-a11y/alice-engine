# ADR-021 : D8 rondes_default division-specific override (Top 16 full 7 rondes)

**Date** : 2026-05-14
**Statut** : Accepté (user explicit 2026-05-14, démarche guidée ISO + SOTA ML)
**Standards** : ISO/IEC 5055 (SRP single-source-of-truth), ISO/IEC 42010
(architecture decision record), ISO/IEC 27034 (input validation),
ISO/IEC TR 24029-2:2024 (robustness via prediction intervals)

---

## Contexte

D8 Phase A push v3 (2026-05-12, post-ADR-020 fixes) :
- 4/5 kernels Nationale 1-4 → COMPLETE (N1=56, N2=101, N3=168, N4=85 matches valides)
- **1/5 ERROR : Top 16**, log Kaggle 0-byte (RuntimeError uncaught avant flush)

Investigation 2026-05-14 par lecture code + inspection parquet :

### Root cause

`scripts/d8/run.py:209` (pré-ADR-021) :
```python
rondes_default = (5, 7, 9, 11) if saison >= 2022 else (1, 3, 5, 7, 9)
```

Ce filtre était calibré pour les **formats Nationale 1-4 FFE** : championnat
saison fin-année reporting sur rondes 5/7/9/11 (équilibrage avant pivot période).

**Top 16 a un format différent** :
- Régulière (Groupe A + Groupe B parallèles) : rondes 1-7
- Finale (Poule Haute + Poule Basse parallèles) : rondes 1-4
- Total candidates Top 16 saison 2024 : **88 matches uniques** (post-ADR-020 dedup)

Avec `rondes_default=(5,7,9,11)`, seules les rondes 5 et 7 régulières passent :
- 2 rondes × 2 groupes × 4 matches = **16 matches** (vs 88 candidates)

Le fail-fast invariant `run.py:350-368` requiert `n_matches ≥ CONFORMAL_CALIB_N + 1 = 31`
(Vovk 2024 §2.3 + Lei 2014 split-conformal minimum). Top 16 16 < 31 →
**`raise RuntimeError(msg)` uncaught** (le `try/except ValueError` ligne 333
ne catch pas RuntimeError) → kernel ERROR.

### Vérification empirique

```python
>>> df = pd.read_parquet('data/echiquiers.parquet')
>>> sub = df[(df.saison==2024) & (df.division=='Top 16') & (df.type_competition=='national')]
>>> sub['ronde'].unique()  # array([1, 2, 3, 4, 5, 6, 7])
>>> sub[['ronde','groupe','equipe_dom','equipe_ext']].drop_duplicates().shape[0]  # 88
>>> sub[sub.ronde.isin([5,7,9,11])][['ronde','groupe','equipe_dom','equipe_ext']].drop_duplicates().shape[0]  # 16
```

ADR-020 (D-2026-05-11) avait correctement anticipé le bump `ALICE_MAX_MATCHES=200`
pour Top 16 (88 candidates + buffer post-filter), mais n'avait pas traité le
filter `rondes` qui invalide le bump. **ADR-021 complète ADR-020** sur la
dimension temporelle (rondes) après que ADR-020 a corrigé la dimension
identitaire (groupe).

---

## Décision

Introduire `DIVISION_RONDES_DEFAULT: dict[str, tuple[int, ...]]` table dans
`scripts/d8/run.py`, override division-specific consulté AVANT le fallback
saison-based :

```python
DIVISION_RONDES_DEFAULT: dict[str, tuple[int, ...]] = {
    "Top 16": (1, 2, 3, 4, 5, 6, 7),
}

# Dans _run_backtest :
if division in DIVISION_RONDES_DEFAULT:
    rondes_default = DIVISION_RONDES_DEFAULT[division]
else:
    rondes_default = (5, 7, 9, 11) if saison >= 2022 else (1, 3, 5, 7, 9)
```

### Pourquoi cette forme (vs alternatives rejetées)

| Option | Verdict | Raison |
|--------|---------|--------|
| **B Mapping dict run.py** (retenu) | ✓ | ISO 5055 single-source-of-truth, ISO 42010 traceable, ISO 27034 exact-match lookup |
| A env var `ALICE_RONDES` wrapper | rejeté | Split logique wrapper+run.py, attack surface parsing env var |
| C adaptive `CONFORMAL_CALIB_N=min(30,n//3)` | rejeté | Patch downstream symptom, dilue robustness SOTA conformal (Lei 2014 N≥30) |

### Top 16 = (1, 2, 3, 4, 5, 6, 7) — pourquoi `range(1,8)` plutôt que limiter

Post-ADR-020, le dedup `(saison, ronde, equipe_dom, equipe_ext, groupe)` distingue
correctement les matches Régulière vs Finale (même ronde nominale, groupe différent).
Inclure toutes les rondes 1-7 capture les 88 matches sans collision identité,
soit 88 ≥ CONFORMAL_CALIB_N+10 buffer = 40 → conformal robuste.

Notons que la **Finale Poule Haute/Basse** n'utilise que rondes 1-4. Le filtre
`isin([1,2,3,4,5,6,7])` retient automatiquement les 4 rondes finale + les 7
rondes régulière, soit Régulière 56 + Finale 32 = **88 candidates uniques** (avec
dedup `groupe`).

---

## Conséquences

### Positives
- Top 16 v4 attendu : **88 candidates** post-dedup → 88 matches → conformal robuste
- Mapping extensible : ajout futur de divisions (Top 12, Top 8, formats Coupes)
  triviale via une entrée additionnelle, sans toucher `_run_backtest`
- N1-N4 inchangé (pas dans `DIVISION_RONDES_DEFAULT`) → backward-compat strict
- Test unitaire (`tests/d8/test_run.py`) garde-fou anti-régression

### Négatives / limites
- Hardcoded mapping ne couvre que **Top 16** (saisons 2022+). Saisons antérieures
  (2018-2021) avec autre format historique non vérifiées — pas dans Phase A scope.
- `DIVISION_RONDES_DEFAULT` est un dict static, pas un module config YAML.
  Acceptable Phase A (≤5 divisions) ; Phase 4+ multi-tenant SaaS demandera
  externalisation config (référencé à `config/hyperparameters.yaml` pattern).

### Risques résiduels
- Si une saison future change le format Top 16 (e.g. 8 rondes régulière),
  mapping doit être saison-keyed (`(saison, division) → rondes`). Pas urgent
  Phase A, à tracer dans dette `D-FUTURE-top16-saison-aware-rondes` si évolution.

---

## Implémentation

Diff applicable sur `scripts/d8/run.py` :

```python
# +14 lignes (DIVISION_RONDES_DEFAULT mapping + docstring)
DIVISION_RONDES_DEFAULT: dict[str, tuple[int, ...]] = {
    "Top 16": (1, 2, 3, 4, 5, 6, 7),
}

# Dans _run_backtest, remplacer ligne unique :
# -    rondes_default = (5, 7, 9, 11) if saison >= 2022 else (1, 3, 5, 7, 9)
# +    if division in DIVISION_RONDES_DEFAULT:
# +        rondes_default = DIVISION_RONDES_DEFAULT[division]
# +    else:
# +        rondes_default = (5, 7, 9, 11) if saison >= 2022 else (1, 3, 5, 7, 9)
```

Test garde-fou `tests/d8/test_run.py` :
- `test_top_16_has_full_7_rondes_mapping` → assert `(1,2,3,4,5,6,7)`
- `test_nationale_divisions_use_saison_default_fallback` → assert absence N1-N4
- `test_division_rondes_default_values_are_tuples_of_ints` → ISO 27034 shape

---

## Validation empirique attendue

Re-push v4 Top 16 :
1. Pre-push hook 5 gates PASS (ruff, mypy, xenon, pip-audit, ruff-format-check)
2. Tests d/8 + tests backtest PASS sur HEAD nouveau commit
3. CI ALL 5 jobs GREEN
4. Re-upload alice-d8-code (CODE_SHA = nouveau commit ADR-021)
5. Re-push kernel d8-2024-top-16 v4 → COMPLETE attendu, n_matches=88

---

## Liens

- ADR-020 (groupe match identity + conformal support_max) — préalable structurel
- `feedback_diagnostic_first_doctrine.md` — diagnostic-first via lecture code + parquet
- `feedback_no_silent_debt.md` — D-2026-05-14-top16-rondes tracée + résolue même commit
- Postmortem (incident in-line) : Top 16 v3 ERROR 2026-05-12 → diagnostic + fix 2026-05-14
- Vovk et al. 2024 "Algorithmic Learning in a Random World" 2nd ed. §2.3 split-conformal
- Lei et al. 2014/2018 minimum N=30 calibration set for split-conformal
