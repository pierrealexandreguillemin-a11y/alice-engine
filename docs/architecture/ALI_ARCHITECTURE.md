# ALI — Architecture (Phase 3)

**Last updated** : 2026-04-19 (Plan 1 Foundations complète)
**Status** : Plan 1 livré, Plan 2 Générateur SOTA à venir
**Référence ADR** : ADR-013 (RuleEngine JSON-driven replaces Python rules)

## Vue d'ensemble Plan 1

```
config/ffe_rules/
  ├── a02.json                   (vendored from chess-app, 14 rules)
  └── alice_verifiability.json   (ALICE local, 10 public + 4 private)
       │
       ▼
services/ffe/
  ├── schemas.py (Pydantic RuleModel, RulesDocument, MetadataModel)
  └── rule_engine.py (Rule dataclass + RuleEngine loader + dispatchers)
       │
       ▼
services/ali/
  ├── types.py (PlayerCandidate, CompetitionContext, RuleViolation)
  ├── verifiability.py (VerifiabilityClassifier)
  ├── cache.py (ALIDataCache — SHA-256 lineage)
  ├── pool_loader.py (PlayerPoolLoader — F7 survivor filter)
  └── history.py (HistoryEnricher — F2 recency + F3 streak)

data/
  ├── joueurs.parquet      → cache.joueurs_by_club (index O(1))
  └── echiquiers.parquet   → cache.echiquiers_by_player (index O(1))
```

## Responsabilités par composant

| Composant | SRP | Entrée | Sortie |
|-----------|-----|--------|--------|
| RuleEngine | Loader + dispatch règles | JSON path | Rule[] + violations |
| VerifiabilityClassifier | Partition public/private | Rule[] | (public[], private[]) |
| ALIDataCache | I/O parquets + index | paths | DataFrames + dicts |
| PlayerPoolLoader | Filter + F7 survivor | club_id, date | PlayerCandidate[] |
| HistoryEnricher | F2 recency + F3 streak | PlayerCandidate[] | enriched PlayerCandidate[] |

## F2 + F3 — features ALI (sources littérature)

### F2 — Recency exponential decay
- Formule : `taux_effectif = Σ_r λ^(age_r) × 1[player plays r] / Σ_r λ^(age_r)`
- λ = 0.9 par défaut (tuné backtest Plan 3)
- Sources : Brown 1959 exponential smoothing, Silver 2012 (FiveThirtyEight methodology)

### F3 — Autoregressive streak lag 1-3
- Features : `played_lag1`, `played_lag2`, `played_lag3` (bool)
- Capture l'effet "joueur qui a joué les 3 dernières rondes → P(jouer prochaine) > taux moyen"
- Sources : Box & Jenkins 1970 (ARIMA), Pappalardo 2019 (soccer squad rotation)

### F7 — Survivor bias filter
- Filter `licence_active == True` dans PlayerPoolLoader
- Évite les joueurs qui ont quitté le club (pollue proba présence)
- Source : Brown, Goetzmann, Ross, Ibbotson 1992 (finance → sport transposition)

## Classification PUBLIC/PRIVATE (A02)

10 règles PUBLIC (appliquées comme contraintes dures dans les scénarios adversaires) :
- 3.7.a team_size
- 3.6.e ordre Elo
- 3.7.c brûlé
- 3.7.d same_group
- 3.7.e match count
- 3.7.g max_mutes
- 3.7.h foreign_quota
- 3.7.i FR gender
- 3.7.j elo_max
- 3.7.k inscriptions (meta-rule réappliquant 3.7)

4 règles PRIVATE (supposées respectées par l'adversaire, non vérifiables ex-ante) :
- 3.7.b force équipes (décision CTF/Ligue)
- 3.2 désignation titulaires
- 3.7.f noyau (déclaré début saison, privé)
- 3.7 arbitrage

## Dépendances externes

Aucune nouvelle dépendance ajoutée en Plan 1. Réutilise pandas, pydantic, pathlib, hashlib.

## ISO compliance Plan 1

| Norme | Mécanisme |
|-------|-----------|
| ISO 5055 | Tous modules < 300 lignes, SRP strict, xenon ≤ B, ruff clean |
| ISO 5259 | Lineage SHA-256 sur rules JSON + parquets (ALIDataCache, RuleEngine.lineage_hash) |
| ISO 27034 | Pydantic validation JSON externes au load |
| ISO 29119 | Coverage ≥ 75% via tests unitaires + smoke integration |
| ISO 42001 | Traceability UUID RFC4122 + source_ref (Rule.uuid_rfc4122) |
| ISO 42010 | ADR-013 RuleEngine JSON-driven |
| ISO 24027 | F7 survivor filter assumption documentée |
| ISO 15289 | MkDocs build --strict pass, documentation ALI_ARCHITECTURE.md livrée |

## Suite (Plans 2-5)

- **Plan 2** : CopulaJointSampler (F6 SOTA) + TopKEnumerator + MonteCarloSampler (F5 LHS/antithetic) + ScenarioGenerator + wire /compose
- **Plan 3** : Walk-forward backtest + 10 gates T13-T22 + Model Card
- **Plan 4** : ScenarioExplainer + ConfidenceLevel + FeedbackCollector + Artefacts ISO 19 normes
- **Plan 5** : Observability DIY + KillSwitch + STRIDE + Capacity + Property-based testing + Multi-tenant-ready
