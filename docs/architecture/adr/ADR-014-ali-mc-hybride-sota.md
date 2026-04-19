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

## Références

- Sklar 1959 — Fonctions de répartition à n dimensions
- McKay, Beckman, Conover 1979 — Latin Hypercube Sampling
- Hammersley & Morton 1956 — antithetic variates
- Genest & Favre 2007 — Everything you always wanted to know about copula modeling
- Nelsen 2006 — An introduction to copulas
- Owen 2013 — Monte Carlo theory, methods and examples
- Elmachtoub & Grigas 2022 — Smart predict, then optimize
- Wilder, Dilkina, Tambe 2019 — Melding the data-decisions pipeline
