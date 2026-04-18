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
  +- FFE eligibility filter (ALICE AUTONOME, pas de dependance chess-app) :
  |    -> PRE-FILTRE sur le pool de joueurs AVANT composition :
  |       joueur brule (A02 3.7.c) -> exclu du pool pour equipe faible
  |       match count (A02 3.7.e) -> exclu si matchs >= ronde
  |       same group (A02 3.7.d) -> exclu si deja joue pour autre equipe du groupe
  |    -> POST-CHECK sur la composition :
  |       ordre Elo 100pts (A02 3.6.e)
  |       1 joueur = 1 equipe
  |       noyau 50% (A02 3.7.f)
  |       max 3 mutes (A02 3.7.g)
  |       quota etrangers (A02 3.7.h)
  |    -> Code : services/ffe_rules.py (8 regles bloquantes)
  |    -> Source : REGLES_FFE_ALICE.md (verifie contre reglements FFE 2025-26)
  |    -> Chess-app Flat-Six (30 validateurs) = optionnel, double-check si cable
  |
  +- CE fallback : allocation par tri Elo + E[score]
  |    -> PAS d'OR-Tools (Phase 4), mais contraintes FFE RESPECTEES via pre-filtre
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
| `services/ffe_rules.py` | 8 regles FFE bloquantes — pre-filtre joueurs + post-check compo |
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

## Conformite ISO — 14 normes (contraintes de conception)

### Code quality (applique a CHAQUE fichier Phase 2)

| Norme | Contrainte | Verification |
|-------|-----------|-------------|
| **ISO 5055** | Max 300 lignes/fichier, 50/fonction, complexite <= B, SRP | `radon`, `xenon`, `wc -l`. feature_store.py, inference.py, model_loader.py DOIVENT respecter. Si >300 lignes → split. |
| **ISO 27001** | Secrets en env vars (HF_TOKEN, MONGODB_URI). Audit logs sur chaque appel /compose. SHA-256 checksum des parquets feature store. | Gitleaks pre-commit. HF_TOKEN JAMAIS dans le code. Audit via data_loader._audit_log(). |
| **ISO 27034** | Validation Pydantic sur TOUTES les entrees. Sanitization ffe_id (regex). Rate limiting /compose (30/min). Erreurs structurees (ErrorResponse). | schemas.py validators. Rejet 422 si format invalide. Pas de stack trace en production. |
| **ISO 25010** | Latence /compose < 5s (10 joueurs x 8 boards = 80 predictions). Disponibilite : fallback LGB+Dirichlet si GBM fail. Memory < 4GB (modeles + feature store). | Benchmark pytest. health endpoint reporte memory usage. |
| **ISO 29119** | Coverage >70%. Docstring structure (ID, Version, Count) sur chaque module. Fixtures pytest pour feature store mock, model mock. Classes de test par composant. | `make test-cov`. Tests unitaires + integration. |
| **ISO 42010** | ADR pour feature store architecture (nouvelle decision). Mise a jour ARCHITECTURE.md avec le pipeline serving. | `docs/architecture/DECISIONS.md` ADR-012. |
| **ISO 15289** | Mise a jour MkDocs avec la doc API (endpoints, schemas). CHANGELOG.md entry Phase 2. | `mkdocs build --strict`. |

### ML serving (specifique au pipeline inference)

| Norme | Contrainte | Verification |
|-------|-----------|-------------|
| **ISO 42001** | Model Card du pipeline stacking complet : 3 GBMs + MLP + temp scaling. Documenter : versions modeles, alpha per-model (LGB=0.1, XGB=0.5, CB=0.3), T=1.02, 18 meta-features. Documenter le fallback LGB+Dirichlet. | metadata.json servi par /health. Model card dans MODEL_SPECS.md deja fait. |
| **ISO 42005** | Impact assessment MISE A JOUR : le modele guide maintenant de VRAIES decisions de composition. Nouveau risque : feature store stale → predictions basees sur donnees obsoletes. Risque : fallback silencieux → capitaine ne sait pas que la qualite a baisse. | docs/iso/AI_RISK_ASSESSMENT.md mise a jour. |
| **ISO 23894** | Risk register MISE A JOUR. Nouveaux risques : (1) feature store stale >14j (2) modele HF corrompu au download (3) fallback silencieux sans notification (4) MongoDB indisponible → pas de joueurs club. Mitigations documentees. | docs/iso/AI_RISK_REGISTER.md mise a jour. |
| **ISO 5259** | Feature store lineage : hash SHA-256 de chaque parquet, date de derniere mise a jour, nombre de joueurs couverts, % NaN par feature. Alerte si age >7j (warning) ou >14j (critical). | feature_store.py log le hash + age au chargement. /health reporte feature_store_age. |
| **ISO 25059** | Quality gates du serving : (1) P(W/D/L) sum=1 par prediction (2) aucun NaN en sortie (3) mean_p_draw > 1% (4) latence < 5s. Monitoring continu, pas juste au training. | Assertions dans inference.py. /health reporte les metriques. |
| **ISO 24029** | Robustesse serving : que se passe-t-il si Elo manquant (→ 1500 default) ? Si joueur inconnu du feature store (→ features par defaut) ? Si HF Hub down au startup (→ cache local) ? Chaque cas documente et teste. | Tests unitaires : inputs degrades → output valide (pas crash). |
| **ISO 24027** | Fairness serving : le pipeline ne doit pas discriminer par club (petit vs grand), par division, par genre. Verifier que les compositions sont equitables. Note : la fairness du MODELE est validee Phase 1. Phase 2 verifie que le SERVING ne degrade pas. | Test : meme joueurs, meme adversaires, clubs differents → memes probas. |

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
