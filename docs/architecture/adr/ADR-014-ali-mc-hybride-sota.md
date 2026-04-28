# ADR-014 : ALI Monte Carlo hybride SOTA — Copule + LHS + TopK

**Date** : 2026-04-19
**Status** : ACCEPTED
**Context** : Phase 3 brainstorming 2026-04-18/19, option B SOTA maximum

## Contexte

Plan 1 a livré les fondations (RuleEngine, data infra, F2/F3/F7 features).
Plan 2 doit livrer le générateur ALI (Adversarial Lineup Inference) qui produit
20 scénarios pondérés de composition adverse pour alimenter le Composition Engine.

L'utilisateur a confirmé l'option B (qualité > temps) : intégrer **dès Plan 2**
les composants SOTA (F1 copule, F5 LHS/antithetic, F6 wrapper) plutôt que les
différer en Plan 3.5.

## Décision

**Architecture : 10 TopK déterministe + 10 Monte Carlo stochastique pondérés.**

### TopKEnumerator (10 scénarios)
Branch-and-bound priorisé énumère les 10 lineups les plus probables (mode dominant
de la distribution), respectant les contraintes RuleEngine PUBLIC. Déterministe,
testable, reproductible.

### MonteCarloSampler (10 scénarios)
- **Latin Hypercube Sampling** (McKay 1979) : couverture stratifiée de l'espace
- **Antithetic variates** (Hammersley & Morton 1956) : 5 paires négativement corrélées
- **Inverse-transform** via copule gaussienne → presence vector binaire
- Conditionnement F3 streak features
- RuleEngine.validate_lineup → reject/resample max 50 retries

### CopulaJointSampler (joint distribution)
- **Copule gaussienne** (Sklar 1959, Genest & Favre 2007, Nelsen 2006)
- Fit Spearman rank correlation matrix sur co-presence empirique
- Transform marginales → N(0,1) via empirical CDF rangs
- Sample N(0, Σ) via Cholesky → inverse-CDF binaire selon F2 taux_presence_effectif

## Conséquences

**Positives :**
- ALI niveau SOTA dès première version production (pas de réécriture Plan 3.5)
- Joint distribution capture corrélation joueurs (n°1 absent → n°9 +probable)
- LHS + antithetic réduit variance MC 30-50% vs IID naïf
- TopK garantit couverture du mode (90% du signal pour CE)
- ISO 42001 SOTA documented satisfait

**Négatives :**
- ~600 lignes nouvelles (3 samplers + orchestrateur)
- Dependency scipy (déjà présent)
- Backtest complexité accrue (Plan 3)

## Alternatives rejetées

### PtO vs DFL (Predict-then-Optimize vs Decision-Focused Learning)
- PtO retenu : explicabilité (chaque scénario interprétable)
- DFL différé : couplage trop serré ML+CE, harder à auditer ISO 42001
- Source : Elmachtoub & Grigas 2022 "Smart predict, then optimize" + Wilder 2019

### Gibbs sampler vs Copule gaussienne
- Gibbs : O(N²) × iterations, mal conditionné pools >30 joueurs
- Copule : O(N) sample, ajustement Spearman rank-based, extensible temporel
- Mesure attendue Plan 3 backtest : copule gagne sur convergence + variance

### IID Monte Carlo vs LHS + antithetic
- IID : 20 scénarios peuvent être quasi-identiques (variance gaspillée)
- LHS + antithetic : couverture stratifiée garantie + paires neg corrélées
- Source : Owen 2013 ch. 10

### Point estimate E[score] vs Conformal prediction (intervalles)
- Conformal différé Plan 4+ : nécessite CE multi-objectif (worst-case)
- Plan 2 livre point E[score] suffisant pour CE Phase 4 OR-Tools de base

### Adaptive Importance Sampling (AIS) en prod
- AIS différé Plan 5+ (D9) : nécessite feedback loop volume prod
- Sans drift observé, gain marginal vs MIS fixe

## Implémentation

- **P2-Task 3** : Scenario types (frozen dataclasses)
- **P2-Task 4** : CopulaJointSampler
- **P2-Task 5** : TopKEnumerator
- **P2-Task 6** : MonteCarloSampler + LHS + antithetic
- **P2-Task 7** : ScenarioGenerator orchestrateur
- **P2-Task 8** : Wire `/compose` + lifespan
- **P2-Task 9** : Suppression `services/ffe_rules.py` legacy
- **P2-Task 10** : Smoke E2E

## Invariants ScenarioSet (D-P2-03 résorbée 2026-04-28)

**Cardinalité = 20 (strict)** : `_EXPECTED_SCENARIOS = 20` dans `services/ali/scenario.py` est un invariant de design, pas un paramètre configurable. Décomposition fixe :

- **10 TopK** déterministes (mode dominant, ~90% du signal CE)
- **10 Monte Carlo** = 5 paires LHS × 2 antithetic (queue de distribution + variance reduction)

Modifier ce 20 = breaking change ADR. Les paramètres `n_topk=10` et `n_mc_pairs=5` du `ScenarioGenerator.generate()` sont calibrés pour respecter `n_topk + 2 * n_mc_pairs = 20`.

**Comportement si `len(scenarios) < 20`** : `ScenarioSet.validate()` raise `ValueError`. Cause typique : pool adversaire trop petit pour générer 20 lineups distincts (`_merge_and_pad` épuise ses 5 rounds de retry sur dedup). Les callers (ex. `BacktestRunner`) doivent skip ce match avec `skip_failed_matches=True` plutôt que tolérer un set partiel (qualité ML dégradée silencieusement).

**Pourquoi pas configurable ?** Variances LHS/antithetic et couverture mode/queue sont calibrées pour 10+10. Un set 5+5 sous-échantillonne la queue ; un 15+15 dilue les poids du mode. Le 20 est un compromis SOTA ; le rendre configurable inviterait des appels mal calibrés sans gain métier.

## Determinism & Reproducibility (D-P2-04 résorbée 2026-04-28)

`ScenarioGenerator.generate(seed=42)` est **déterministe par default** : même `(opponent_club_id, round_date, context, saison, ronde, nb_rondes_total, n_topk, n_mc_pairs, seed)` → même `lineage_hash` SHA-256 → même `tuple[Scenario, ...]`. Vérifié par `tests/backtest/test_determinism.py` (T18, 4/4 PASS).

**Audit ISO 5259** : le default `seed=42` est volontaire pour garantir l'**idempotence observable** des appels `/compose` successifs (deux requêtes identiques produisent même `lineage_hash` → audit log MongoDB cohérent).

**API `ComposeRequest.seed`** : champ optionnel `int | None`. Sémantique :

- `seed=None` (default) → generator utilise `seed=42` (audit-stable, recommandé prod)
- `seed=<int>` (custom) → variations RNG pour exploration alternative (ex. capitaine veut 20 alternatives au lieu d'un set unique)

Le `seed` est inclus dans le `lineage_hash` → la chaîne de traçabilité ISO 42001 reste intacte quel que soit le seed choisi.

## Composer legacy supprimé (D5 résorbée 2026-04-28)

`services/composer.py::ComposerService` (legacy, 207 lignes) supprimé. Le vrai flow CE est `app/api/routes.py::/compose` qui invoque `ScenarioGenerator` + `StackingInferenceService` + `aggregate_from_scenarios`. Le composer legacy implémentait une formule Elo simpliste rendue obsolète par le pipeline ML stacking (Phase 2). Tests `tests/test_composer.py` également supprimés (couverture redondante avec `tests/test_inference_pipeline.py`).

## Fail-fast MC fallback (D-P3-12 résorbée 2026-04-28)

`MonteCarloSampler._u_to_presence` raise `RuntimeError` si copula non-fittée ou `n_players` mismatch (au lieu d'un fallback threshold marginal indépendant qui produirait des samples non-corrélés → biais conservateur silencieux). ISO 24029 fail-fast : mieux vaut crash que biais caché. Vérifié par `tests/test_monte_carlo.py::test_mc_unfit_copula_raises_runtime_error` + `test_mc_copula_size_mismatch_raises_runtime_error`.

## Références

- Sklar 1959 — Fonctions de répartition à n dimensions
- McKay, Beckman, Conover 1979 — Latin Hypercube Sampling
- Hammersley & Morton 1956 — antithetic variates
- Genest & Favre 2007 — Everything you always wanted to know about copula modeling
- Nelsen 2006 — An introduction to copulas
- Owen 2013 — Monte Carlo theory, methods and examples
- Elmachtoub & Grigas 2022 — Smart predict, then optimize
- Wilder, Dilkina, Tambe 2019 — Melding the data-decisions pipeline
