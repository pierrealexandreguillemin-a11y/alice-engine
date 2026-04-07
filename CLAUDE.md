# Alice-Engine - Guide Claude

## BUT DU PROJET

Alice Engine = **recommandation de composition d'équipe** interclubs FFE.
Pipeline : ALI (prédire adversaire) → ML (P(win/draw/loss) par joueur×échiquier) → CE (optimiser E[score]) → API FastAPI.
Le modèle DOIT produire 3 probabilités. Le CE calcule `E[score] = P(win) + 0.5×P(draw)`.
Compositions soumises simultanément (A02 Art. 3.6.a) — capitaine ne connaît PAS l'adversaire.

## ÉTAT ACTUEL (avril 2026)

**V9 Optuna EN COURS.** V8 modèles passent quality gates mais hyperparams manuels.
Optuna Bayesian optimization lancée (11 étapes, 9 kernels). XGBoost canary RUNNING.
Spec : `docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md`

| Couche | Statut |
|--------|--------|
| API FastAPI + schemas | COMPLET (stubs) |
| ML Training V8 | 3 modèles convergés, V9 Optuna en cours |
| Câblage routes→services | MANQUANT (après V9) |
| ALI prédiction adverse | MANQUANT (Phase 3) |
| CE multi-équipe OR-Tools | MANQUANT (Phase 4) |

## RÈGLES ABSOLUES

### Code (ISO 5055)
- **Max 300 lignes/fichier**, 50 lignes/fonction, complexité ≤ B
- SRP : 1 fichier = 1 responsabilité. Refactorer si > 300 lignes.

### Sécurité (ISO 27034)
- Pydantic pour toute validation. Jamais de secrets en clair.

### Démarche de rigueur
- **WebSearch si doute** — vérifier doc officielle AVANT utilisation
- **Audit domaine AVANT push** — standards ML + logique échecs/FFE
- **Post-mortem** chaque échec. Documenter dans `docs/postmortem/`
- **"Je ne sais pas"** > affirmation fausse

### Process Kaggle (9 étapes, voir `memory/feedback_verify_before_push.md`)
1. Recherches web (standards ML, littérature)
2. Corrections
3. Audit skill ml-training-pipeline
4. Corrections
5. Audit skill kaggle-deployment
6. Corrections
7. Quality gates F1-F12 / T1-T12
8. Présenter bilan → ATTENDRE validation user
9. Push

## TRAINING RULES (V8/V9)

- **NE JAMAIS** entraîner sans residual learning (baseline forte = Elo)
- **NE JAMAIS** déclarer champion sans Optuna — hyperparams manuels ≠ optimisés
- **NE JAMAIS** utiliser `optuna.integration` — v4.0+ = `optuna_integration`
- **NE JAMAIS** TreeSHAP sur test complet — subsample 20K
- **NE JAMAIS** écrire budget temps sans calcul
- **NE JAMAIS** CatBoost init_model + Pool(baseline=) — erreur fatale
- **TOUJOURS** init_scores AVANT filtrage features
- **TOUJOURS** rsm=0.3-0.5 pour CatBoost >50 features
- **TOUJOURS** quality gates AVANT SHAP
- **TOUJOURS** vérifier dataset Kaggle contient fichiers importés avant push
- **TOUJOURS** SQLite storage Optuna (pas pickle)

Inference : `compute_elo_baseline → init_scores → *= alpha → predict_with_init`
Alpha dans `metadata.json`. `draw_rate_lookup.parquet` requis.

## COMMANDES

```bash
make all           # Validation complète (lint+test+coverage+complexity)
make refresh-data  # Sync + parse + validate + features
make test-cov      # Tests + coverage >70%
```

### Kaggle push (pattern)
```bash
python -m scripts.cloud.upload_all_data    # TOUJOURS avant push si code modifié
cp scripts/cloud/kernel-metadata-{NAME}.json scripts/cloud/kernel-metadata.json
kaggle kernels push -p scripts/cloud/
git checkout -- scripts/cloud/kernel-metadata.json
```

## DONNÉES

HuggingFace : `Pierrax/ffe-history` (public). Compte : Pierrax.
Local : `data/echiquiers.parquet` (~35 MB), `data/joueurs.parquet` (~3 MB).
Scraping : repo `C:\Dev\ffe_scrapper`.

## ISO

14 normes (7 générales + 7 ML/AI). Détails : `docs/iso/ISO_STANDARDS_REFERENCE.md`

## DOCUMENTATION

**Index complet : `.claude/rules/docs-index.md`** (65 docs)

Priorité :
- `docs/superpowers/specs/2026-04-07-optuna-v9-pipeline-design.md` — V9 Optuna
- `docs/project/V8_MODEL_COMPARISON.md` — Comparaison modèles
- `docs/requirements/FEATURE_DOMAIN_LOGIC.md` — Logique métier features
- `docs/requirements/QUALITY_GATES.md` — Gates F1-F12 / T1-T12

## RULES FILES

- `.claude/rules/docs-index.md` — Index 65 docs par domaine
- `.claude/rules/v8-training-findings.md` — Historique campagne V8 (bugs, découvertes, chronologie)

## ARCHITECTURE CIBLE

```
Vercel (chess-app) → HTTPS → Oracle VM (FastAPI + ML, 24GB ARM)
```
Spec : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md`
V9 CE multi-équipe : `CLAUDE.md` historique dans `.claude/rules/v8-training-findings.md`
