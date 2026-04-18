# Phase 2 : Feature Store + API Wiring — Design Spec

**Date:** 2026-04-18
**Status:** DRAFT
**Prerequis:** Phase 1 COMPLETE (champion MLP(32,16) 0.5530, models on HF)
**Scope:** Pipeline inference E2E temps reel, fallback ALI+CE

---

## Objectif

Le capitaine ouvre chess-app, voit son pool de joueurs dispos, lance ALICE,
recoit les compositions de ses N equipes + adversaires predits, peut ajuster
et relancer. Tout le pipeline ML V9 tourne en arriere-plan.

---

## Architecture inference

```
POST /api/v1/compose
  |
  +- DataLoader : joueurs club (MongoDB) + adversaires (parquet FFE)
  |
  +- ALI fallback : pour chaque match (N matchs du club)
  |    -> top K adversaires par Elo ranking (K = team_size, 4-16 selon division)
  |    -> 1 scenario deterministe (Phase 3 = 20 Monte Carlo)
  |
  +- ML stacking : pour chaque (joueur_candidat x echiquier x adversaire_predit x equipe)
  |    -> feature_store.assemble(joueur, adversaire, contexte_match_equipe)  # 201 cols
  |    -> init_scores = elo_baseline(joueur.elo, adversaire.elo) x alpha_per_model
  |    -> LGB predict_with_init(init_scores * 0.1)
  |    -> XGB predict_with_init(init_scores * 0.5)
  |    -> CB  predict_with_init(init_scores * 0.3)
  |    -> 18 meta-features (std, max_prob, entropy)
  |    -> MLP(32,16) -> temp scaling (T=1.02)
  |    -> P(W/D/L) calibrees par cellule
  |
  +- CE fallback : allocation par tri Elo + E[score]
  |    -> contraintes FFE verifiees :
  |       ordre Elo 100pts (A02 3.6.e)
  |       1 joueur = 1 equipe
  |       noyau 50% (A02 3.7.f)
  |       max 3 mutes (A02 3.7.g)
  |       joueur brule (A02 3.7.c)
  |    -> PAS d'OR-Tools (Phase 4), mais contraintes RESPECTEES
  |
  +- Reponse :
       -> N compositions (1 par equipe)
       -> adversaires predits visibles par le capitaine
       -> E[score] par board + par equipe
       -> alternative remplacement si joueur annule
```

**Fallback** : si un des 3 GBMs fail au chargement, bascule sur LGB + Dirichlet
(0.5541 log_loss, 0.0042 ECE_draw). Log warning, pas d'erreur.

---

## Composants et fichiers

### Modifier (existant)

| Fichier | Modification |
|---------|-------------|
| `app/main.py` | Model loading au startup (3 GBMs + MLP + calibrateurs depuis HF Hub) |
| `app/config.py` | Ajouter HF_REPO_ID, FEATURE_STORE_PATH, FALLBACK_MODE |
| `app/api/routes.py` | Cabler POST /compose, POST /recompose, supprimer /train |
| `app/api/schemas.py` | ComposeRequest/Response (N equipes, pool joueurs, adversaires predits) |
| `services/inference.py` | Reecrire : stacking pipeline (3 GBM -> meta-features -> MLP -> temp) |
| `services/composer.py` | Verification contraintes FFE (pas OR-Tools, validation + tri Elo) |

### Creer (nouveau)

| Fichier | Role |
|---------|------|
| `services/feature_store.py` | Assemble 201 features depuis parquets pre-calcules |
| `scripts/serving/model_loader.py` | Download HF Hub -> cache local, load 4 modeles en memoire |
| `scripts/serving/meta_features.py` | build_meta_features(p_xgb, p_lgb, p_cb) -> 18 cols |

### Garder tel quel (reutilise)

| Fichier | Utilisation |
|---------|------------|
| `scripts/kaggle_metrics.py` | predict_with_init() |
| `scripts/baselines.py` | compute_elo_baseline(), compute_init_scores_from_features() |
| `scripts/features/draw_priors.py` | build_draw_rate_lookup() |
| `services/data_loader.py` | MongoDB + parquet (deja fonctionnel) |

---

## Feature Store

Piece manquante principale. Le FE pipeline Kaggle produit des features par PARTIE
(1 ligne = 1 partie jouee). Le feature store production produit des features par
JOUEUR (stats agregees) qu'on combine avec le contexte match.

```python
class FeatureStore:
    """Assemble 201 features pour une prediction (joueur x adversaire x contexte).

    Lookup dans parquets pre-calcules :
    - joueur_features.parquet  (stats rolling 3 saisons par joueur)
    - equipe_features.parquet  (stats club, noyau, classement)
    - draw_rate_lookup.parquet (prior draw par bande Elo)

    Pas de recalcul — les parquets sont refresh par cron hebdo.
    Meme code FE que le kernel Kaggle, execute localement.
    """
    def __init__(self, store_path: Path): ...
    def load(self) -> None: ...
    def assemble(self, joueur_elo, adversaire_elo, contexte_match) -> pd.DataFrame: ...
```

### Refresh cron (hebdomadaire)

```
1. scrape_ffe.py --refresh        # nouvelles rondes
2. parse -> echiquiers.parquet    # update donnees
3. FE pipeline local              # recalcul features joueur/equipe
4. -> joueur_features.parquet     # stats rolling mises a jour
5. -> equipe_features.parquet     # classements, noyau, historique
```

---

## Endpoints API

### POST /api/v1/compose

```
Request: {
    club_id: str,                    # club FFE du capitaine
    joueurs_disponibles: [str],      # liste ffe_id des joueurs dispos
    mode_strategie: str = "agressif" # agressif|conservateur
}

Response: {
    compositions: [{
        equipe: str,                 # "Marseille 1"
        division: str,               # "N3"
        adversaire: str,             # club adverse
        adversaire_predit: [{        # compo adverse (visible capitaine)
            board: int,
            joueur: str,
            elo: int
        }],
        boards: [{
            board: int,
            joueur: str,
            elo: int,
            adversaire: str,
            adversaire_elo: int,
            p_win: float,
            p_draw: float,
            p_loss: float,
            e_score: float
        }],
        e_score_total: float,
        contraintes_ok: bool
    }],
    metadata: {
        model_version: str,
        ali_mode: str,               # "elo_fallback" ou "monte_carlo"
        timestamp: str
    }
}
```

### POST /api/v1/recompose

```
Request: {
    club_id: str,
    joueurs_disponibles: [str],      # pool mis a jour
    composition_precedente: [...],   # compo a ajuster
    mode: "remplacement" | "global"  # swap simple ou recomposition totale
}

Response: meme format + diff avec compo precedente
```

### GET /api/v1/health

```
Response: { status, models_loaded, feature_store_age, version }
```

---

## Ordre d'implementation

1. **Feature store** — assemble les 201 cols (bloquant pour tout le reste)
2. **Model loader** — charge les 4 modeles depuis HF (bloquant pour inference)
3. **Inference service** — stacking pipeline complet
4. **Composer service** — contraintes FFE + allocation tri Elo
5. **API routes** — cablage endpoints
6. **Tests E2E**

---

## Tests et criteres de succes

### Tests E2E

| Test | Input | Verification |
|------|-------|-------------|
| Smoke test | 10 joueurs, 1 equipe | 200 OK, 8 boards, P(W/D/L) sum=1, E[score] coherent |
| Multi-equipe | 30 joueurs, 3 equipes | 1 joueur = 1 equipe, ordre Elo, 3 compos |
| Contraintes FFE | joueur brule dans pool | joueur brule NON assigne equipe inferieure |
| Recomposition | 1 joueur retire | nouvelle compo valide, diff retournee |
| Fallback modele | XGB manquant sur HF | LGB+Dirichlet fonctionne, log warning |

### Critere de succes Phase 2

Le capitaine peut appeler /compose avec son pool de joueurs et recevoir des
compositions valides avec des P(W/D/L) calculees par le pipeline ML V9 complet.
Pas optimales (pas OR-Tools, pas ALI ML), mais valides (contraintes FFE respectees)
et utiles (probas calibrees ECE_draw 0.0016).

---

## ISO Deliverables Phase 2

| Norme | Artefact |
|-------|---------|
| ISO 42001 | Model Card updated (serving endpoint documente) |
| ISO 5259 | Feature Store lineage (hash, date, coverage par joueur) |
| ISO 27034 | Input validation (Pydantic schemas, rate limiting) |
| ISO 25059 | Serving metrics (latence P50/P99, throughput) |
| ISO 42010 | ADR feature store architecture |
| ISO 29119 | Integration tests (POST /compose E2E avec vrai modele) |

---

## Hors scope Phase 2

| Element | Phase |
|---------|-------|
| ALI ML (Monte Carlo 20 scenarios) | Phase 3 |
| CE OR-Tools (optimisation multi-objectif) | Phase 4 |
| Modes strategie (tactique, risk-adjusted) | Phase 4 |
| Batch predict cron | Phase 5 |
| Deploy Oracle VM | Phase 5 |
| UI chess-app | Apres API stable |
| NN base model (diversite stacking) | Phase 5+ |
| Decision-Focused Learning | Phase 5+ |
| Conformal prediction / copules | Phase 4+ |

---

## Sources

- MODEL_SPECS.md : pipeline inference, alpha per-model, stacking
- REGLES_FFE_ALICE.md : contraintes composition (A02 3.6-3.7)
- docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md : roadmap 5 phases
- memory/project_batch_architecture.md : batch ML + CE on-demand
- Kull 2019 (NeurIPS) : Dirichlet calibration (fallback)
- Guo 2017 (ICML) : temperature scaling (champion)
