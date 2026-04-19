# ALI — Architecture (Phase 3)

**Last updated** : 2026-04-19 (Plan 2 Generator SOTA livré)
**Status** : Plans 1+2 livrés, Plan 3 backtest validation à venir
**Référence ADR** : ADR-013 (RuleEngine JSON) + ADR-014 (ALI MC hybride SOTA)

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

### F7 — Survivor bias filter (interprétation FFE)

`joueurs.parquet` est la base FFE des licences actives au moment du scraping.
Les joueurs ayant quitté un club n'apparaissent plus dans `joueurs_by_club[club_id]`,
donc F7 est enforcé **implicitement** via la composition du parquet.

`_row_licence_active` retourne True pour toutes les lignes — pas de flag
ARCHIVE/INACTIVE observé sur 83K joueurs (valeurs `elo_type` ∈ {E, F, N, ''}).

**Override** : un capitaine peut passer `licence_active: false` dans `overrides`
pour signaler une licence en cours de renouvellement (cas exceptionnel hors
fenêtre de scraping).

Source : audit schema réel 2026-04-19 (Plan 2 Task 1, finding D-P3-04).
Référence théorique : Brown, Goetzmann, Ross, Ibbotson 1992 (finance → sport).

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

## Plan 2 — Générateur SOTA (livré 2026-04-19)

### Composants ajoutés

```
services/ali/
  ├── scenario.py          (frozen types: BoardAssignment, Lineup, Scenario, ScenarioSet)
  ├── joint_sampler.py     (CopulaJointSampler — F6 Sklar 1959)
  ├── topk.py              (TopKEnumerator — branch-and-bound déterministe)
  ├── monte_carlo.py       (MonteCarloSampler + LHS + antithetic — F5 McKay 1979)
  └── generator.py         (ScenarioGenerator — orchestrateur 10 TopK + 10 MC = 20)

app/
  ├── main.py              (lifespan : _init_ali_generator)
  └── api/
      ├── routes.py        (compose_route wired ScenarioGenerator)
      └── schemas.py       (ComposeRequest +6 fields ALI optional)
```

### Pipeline /compose avec ALI

```
POST /compose request
  ├── club_id + joueurs_disponibles + ronde + division (Phase 2)
  ├── opponent_club_id + round_date + saison + current_round + nb_rondes_total (Plan 2)
  └── player_overrides? (optional)
       │
       ▼
ScenarioGenerator.generate(opponent_club_id, ...)
  1. PlayerPoolLoader.load_pool (F7 implicit via membership)
  2. HistoryEnricher.enrich (F2 recency + F3 streak)
  3. CopulaJointSampler.fit (Spearman rank correlation matrix)
  4. RuleEngine.filter_candidates (10 PUBLIC rules A02)
  5. TopKEnumerator.enumerate (10 lineups deterministic)
  6. MonteCarloSampler.sample (10 lineups LHS + antithetic via copule)
  7. _merge_and_pad : dedup distincts (T20), boucle si < 20
  8. _renormalize : sum(weights) = 1.0 (T18)
  9. ScenarioSet.validate (T18, T19)
  10. lineage_hash = SHA-256(opp + date + ctx + saison + round + lambda + rules_sig + parquet_sigs)
       │
       ▼
ScenarioSet (20 scenarios, lineage_hash)
       │
       ▼
Boucle inférence Phase 2 × 20 scenarios → CE moyenne pondérée
       │
       ▼
ComposeResponse {
  lineup, expected_score, per_board_probabilities,
  metadata: { lineage_hash, ali_mode: "scenario_generator", n_scenarios: 20, rule_uuids: [...] }
}
```

### F1/F5/F6 — Composants SOTA Plan 2

| Composant | Source littérature |
|-----------|---------------------|
| F1 — Joint distribution sampling (CopulaJointSampler) | Sklar 1959, Genest & Favre 2007, Nelsen 2006 |
| F5 — Latin Hypercube + antithetic variates (MonteCarloSampler) | McKay, Beckman, Conover 1979 ; Hammersley & Morton 1956 ; Owen 2013 |
| F6 — Copule gaussienne (extension F1, méthode `transform_uniform_to_presence`) | Sklar 1959, Cholesky |

### Quality gates Plan 2 — 16 P2G + 7 structural

Voir `docs/superpowers/plans/2026-04-19-phase3-plan2-generator-sota.md` §DoD pour la table complète. Gates clés :
- T18 sum(weights)=1±1e-4
- T19 len(scenarios)=20
- T20 distincts garantis (ScenarioGenerator._merge_and_pad)
- T21 MC rejection_rate ≤ 30%
- Latence p95 /compose ≤ 2000ms (mesuré)
- Coverage ≥ 75% (Plan 1+2 combinés)

### Backward compatibility Phase 2

Si `opponent_club_id` absent dans la request, `_try_generate_scenarios` retourne None → fallback Phase 2 inchangé (Elo synthétique + tri Elo). Tests `tests/test_compose_e2e.py` 11/11 PASS sans régression.

## Suite (Plans 3-5)

- **Plan 3** : Walk-forward backtest + 10 gates T13-T22 + Model Card SOTA + suppression `services/ffe_rules.py` legacy (D-P3-11)
- **Plan 4** : ScenarioExplainer + ConfidenceLevel + FeedbackCollector + AI_RISK_REGISTER (R-ALI-*)
- **Plan 5** : Observability DIY + KillSwitch + STRIDE + Capacity benchmark + Property-based testing + Multi-tenant-ready
