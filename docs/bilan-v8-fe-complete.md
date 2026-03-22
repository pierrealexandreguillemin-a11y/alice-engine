# Bilan V8 MultiClass — Feature Engineering COMPLETE

> Date: 2026-03-22 | Branch: `feat/v8-multiclass-features` | 27 commits

---

## 1. Statut Pipeline

| Etape | Statut | Artefact | Localisation |
|-------|--------|----------|--------------|
| **Plan A: Feature Engineering** | COMPLETE | 3 parquets | Kaggle kernel output |
| **Plan B: Training MultiClass** | CODE READY | Scripts locaux | A pousser sur Kaggle |
| **Kaggle Kernel 1 (FE)** | COMPLETE | `alice-fe-v8` | `pguillemin/alice-fe-v8` |
| **Kaggle Kernel 2 (Training)** | A LANCER | `alice-training-v8` | `pguillemin/alice-training-v8` |
| **Dataset alice-code** | A METTRE A JOUR | Scripts Python | `pguillemin/alice-code` |

---

## 2. Outputs FE — Inventaire des artefacts

### 2.1 Parquets features (Kaggle kernel output `alice-fe-v8`)

| Fichier | Lignes | Colonnes | Période | Taille estimée |
|---------|--------|----------|---------|----------------|
| `features/train.parquet` | 1,139,799 | 196 | 2002-2022 | ~800 MB |
| `features/valid.parquet` | 70,647 | 196 | 2023 | ~50 MB |
| `features/test.parquet` | 231,532 | 196 | 2024-2026 | ~160 MB |

### 2.2 Décomposition des 196 colonnes

| Catégorie | #cols | Type | Nouveauté V8 |
|-----------|-------|------|-------------|
| Match context (saison, ronde, echiquier, division...) | ~15 | input | — |
| Player Elo + titre | ~6 | input | — |
| Enrichissement joueurs (elo_type, categorie, K-coeff) | 6 | enrichi | V8 |
| Recent form W/D/L (×blanc/noir) | 10 | rolling | Refactoré V8 |
| Color perf W/D/L (×blanc/noir) | 16 | rolling 3 saisons | Refactoré V8 |
| Board position (×blanc/noir) | 4 | rolling last season | Fix V8 |
| Club reliability (×dom/ext) | 6 | historique | — |
| Player reliability (×blanc/noir) | 4 | historique | — |
| Standings + zone_enjeu (×dom/ext) | 16 | real-time standings | — |
| Club behavior (×dom/ext) | ~16 | historique | W/D/L V8 |
| Noyau (×blanc/noir) | 2 | réglementaire | — |
| FFE regulatory (×blanc/noir) | 8 | réglementaire | — |
| **Draw priors** (avg_elo, proximity, prior) | **3** | lookup table | **NEW V8** |
| **Draw rate player** (×blanc/noir) | **2** | historique | **NEW V8** |
| **Draw rate equipe** (×dom/ext) | **2** | historique | **NEW V8** |
| **Club level / vases** (×dom/ext) | **10** | inter-équipes | **NEW V8** |
| **Player team context** (×blanc/noir) | **6** | promu/relegue/gap | **NEW V8** |
| ALI presence (×blanc/noir) | 6 | historique | — |
| ALI patterns (×blanc/noir) | 6 | historique | — |
| ALI absence (×blanc/noir) | 4 | rolling 3 saisons | — |
| Composition strategy (×blanc/noir) | 6 | historique | — |
| Trajectory + momentum (×blanc/noir) | 4 | rolling window=6 | — |
| Pressure/clutch W/D/L (×blanc/noir) | 14 | historique | W/D/L V8 |
| H2H W/D/L (×blanc/noir) | 8 | historique | W/D/L V8 |
| Temporal (phase_saison, ronde_normalisee) | 2 | direct | — |
| match_important, adversaire_niveau | 3 | derived | — |
| Domicile | 1 | direct | — |
| Identifiers + target (noms, equipes, résultat) | ~18 | metadata | — |
| **TOTAL** | **~196** | | **+23 new V8** |

### 2.3 Qualité données (signaux du log FE)

| Métrique | Valeur | Verdict |
|----------|--------|---------|
| Draw rate global | 0.142 (train/valid/test ±0.001) | Stable |
| null_priors (draw_rate_prior) | 0 | Couverture 100% |
| Joueurs fantômes | 404 / 104,136 (0.4%) | Normal |
| Color perf coverage | 5,537 / 74,819 (7.4%) | Faible — NaN 93% |
| H2H pairs ≥3 confrontations | 8,684 | Sparse — NaN >99% |
| Noyau entries | 2,188,220 | Dense |
| Forfaits exclus | Oui (filter + exclude_forfeits) | ISO 5259 |

---

## 3. Git — 27 commits conventionnels

```
b65d8b8 fix(hooks): enable_internet check only for training/autogluon kernels
596d76e feat(kaggle): simplify train_kaggle.py to pure Kernel 2 + wire FE kernel_source
463f0f2 perf(features): vectorize club_level extract + reinforcement_rate (O(n²) → O(n))
85bbdc7 feat(kaggle): split into 2-kernel architecture (FE + training)
194b562 feat(kaggle): FE checkpoint — skip feature engineering if parquets already exist
419e10c refactor(trainers): extract constants to kaggle_constants.py (ISO 5055 310->300)
90db988 chore(kaggle): slug alice-training-v7 -> alice-training-v8
d9a279f fix(training): quality gate test metrics + lineage 3-class + review fixes
45339bb fix(artifacts): model card best_auc -> best_log_loss (MultiClass gate)
074157e feat(config): multiclass thresholds + AutoGluon 3-class + test suite adapted
f5a5e2e feat(training): quality gate 8 conditions + baselines (naive + Elo)
a5a1038 feat(training): MultiClass 3-way — target, configs, metrics, calibration
14e1306 fix(plan): Plan A Task 12 — no Kaggle upload until Plan B complete
b5dba44 fix(plans): feature engineering on Kaggle, not local (15GB OOM)
982927c fix(features): recent_form row explosion + K-coefficient FFE categories
64a7241 fix(features): K-coefficient young categories — FFE names not FIDE codes
d057d29 feat(features): integrate all V8 modules into pipeline — draw priors + club level
ac92763 fix(features): echiquier_moyen rolling last season instead of global
e804593 feat(features): add club_level — vases communiquants (joueur_promu/relegue)
d610e75 refactor(features): club_behavior W/D/L home rates + forfait exclusion
8b0df97 refactor(features): h2h W/D/L + h2h_exists + draw_rate_h2h
963d22a fix(features): restore recent_form W/D/L + fix draw_priors diff_elo drop
08dd428 fix(features): pressure uses zone_enjeu (leakage fix) + W/D/L
14a9342 refactor(features): recent_form W/D/L decomposition + competition stratification
088b5a1 refactor(features): color_perf W/D/L decomposition + rolling 3 seasons
0dc9ac6 feat(features): add draw priors — avg_elo, elo_proximity, draw_rate
f15ae24 feat(features): add forfait filter + W/D/L rate helpers (ISO 5259)
```

### Catégories de commits

| Type | Count | Description |
|------|-------|-------------|
| `feat` | 12 | Nouvelles features, baselines, architecture 2-kernel |
| `fix` | 9 | Leakage, K-coeff, quality gate, row explosion |
| `refactor` | 4 | W/D/L decomposition, SRP extraction |
| `perf` | 1 | Vectorisation club_level O(n²)→O(n) |
| `chore` | 1 | Slug renaming |

---

## 4. Tracking des outputs (DVC-style)

> DVC installé (v3.67.0) mais non initialisé dans ce repo.

### 4.1 Artefacts Kaggle (remote)

| Artefact | Slug Kaggle | Type | Statut | Date |
|----------|-------------|------|--------|------|
| Features train/valid/test | `pguillemin/alice-fe-v8` | kernel output | COMPLETE | 2026-03-22 17:05 |
| Code Python + data | `pguillemin/alice-code` | dataset | A METTRE A JOUR | 2026-03-22 15:48 |
| Training models | `pguillemin/alice-training-v8` | kernel output | PAS ENCORE LANCÉ | — |

### 4.2 Artefacts attendus du Training (Kernel 2)

```
/kaggle/working/{version}/
├── CatBoost_model.cbm           # Modèle CatBoost MultiClass
├── XGBoost_model.ubj            # Modèle XGBoost multi:softprob
├── LightGBM_model.txt           # Modèle LightGBM multiclass
├── calibrators.joblib            # Isotonic 3-class (per model × 3 classes)
├── metadata.json                 # Model card + lineage + quality gate
├── catboost_info/                # CatBoost training logs
├── *_feature_importance.csv      # Feature importance per model
├── *_test_predictions.parquet    # Predictions brutes + calibrées
├── *_valid_predictions.parquet   # Idem validation
├── *_learning_curve.csv          # Loss per iteration
├── *_roc_loss/draw/win.csv       # ROC curves 3-class OvR
├── *_calibration_loss/draw/win.csv # Calibration curves
├── classification_reports.json   # Precision/recall/F1 per class
└── train_feature_distributions.csv # Drift baseline (ISO 5259)
```

### 4.3 Pipeline de données end-to-end

```
FFE HTML (85k fichiers) ──scrape──▶ HuggingFace Pierrax/ffe-history
                                        │
                                   parse_dataset
                                        │
                                        ▼
                              echiquiers.parquet (1.75M lignes)
                              joueurs.parquet (83k joueurs)
                                        │
                                  upload_all_data
                                        │
                                        ▼
                              Kaggle pguillemin/alice-code
                                        │
                               ┌────────┴────────┐
                               ▼                  ▼
                    Kernel 1: alice-fe-v8    (code Python)
                    (P100 CPU, 74 min)
                               │
                               ▼
                    features/{train,valid,test}.parquet
                    (196 cols, 1.44M lignes)
                               │
                               ▼
                    Kernel 2: alice-training-v8
                    (T4 GPU, ~30 min estimé)
                               │
                               ▼
                    3 modèles + calibrators + diagnostics
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              Quality Gate   Model Card   HF Push
              (8 conditions)  (ISO 42001)  (si gate OK)
```

---

## 5. Etapes préliminaires avant Kernel 2

### BLOCKERS (à faire avant push)

| # | Action | Pourquoi | Commande |
|---|--------|----------|----------|
| 1 | **Re-upload alice-code dataset** | `train_kaggle.py` modifié (simplifié Kernel 2), `club_level.py` vectorisé — le dataset date de 15:48, avant ces commits | `python -m scripts.cloud.upload_all_data` |
| 2 | **Vérifier kernel_sources monte le FE output** | `kernel_sources: ["pguillemin/alice-fe-v8"]` doit monter les parquets dans `/kaggle/input/alice-fe-v8/features/` | Vérifiable uniquement au runtime |

### PUSH COMMAND

```bash
# Copier metadata training
cp scripts/cloud/kernel-metadata-train.json scripts/cloud/kernel-metadata.json

# Push avec T4 GPU explicite (pas P100 par défaut)
kaggle kernels push -p scripts/cloud/ --accelerator NvidiaTeslaT4

# Restaurer metadata
git checkout -- scripts/cloud/kernel-metadata.json

# Surveiller
kaggle kernels status pguillemin/alice-training-v8
```

### RISQUES

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| kernel_sources ne monte pas les parquets FE | Faible | Fatal | Le script fail-fast avec FileNotFoundError |
| RAM insuffisante (3 modèles × 1.1M lignes × 196 features) | Moyen | Fatal | Sequential training + gc.collect() entre modèles |
| CatBoost GPU OOM sur T4 (15GB VRAM) | Faible | Partiel | Fallback CPU dans les params si besoin |
| Quality gate trop strict (ECE < 0.05 per class) | Moyen | Soft fail | Modèles sauvés localement même si gate échoue |
| HF push silencieux fail (pas de secrets batch) | Connu | Soft fail | Promotion manuelle post-kernel |

---

## 6. Métriques attendues (quality gate)

Le training doit battre les 2 baselines ET passer la calibration :

| Condition | Seuil | Métrique |
|-----------|-------|----------|
| log_loss < naive | Distribution marginale (class freq) | `test_log_loss` |
| log_loss < Elo | Elo formula + draw_rate lookup | `test_log_loss` |
| RPS < naive | Ranked Probability Score ordinal | `test_rps` |
| RPS < Elo | idem | `test_rps` |
| Brier < naive | Multiclass Brier score | `test_brier` |
| E[score] MAE < Elo | P(win)+0.5*P(draw) vs actual | `test_es_mae` |
| ECE < 0.05 per class | Expected Calibration Error | `ece_class_loss/draw/win` |
| draw calibration bias < 0.02 | mean P(draw) - observed draw rate | `draw_calibration_bias` |
