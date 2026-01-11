# Analyse Initiale ALICE - Documentation de Départ

> **Date** : 3 Janvier 2026
> **Contexte** : Analyse du dataset FFE et choix technologiques pour ALICE
> **Source** : Conversation Claude Code - Analyse approfondie

---

## ANALYSE CRITIQUE DU DATASET ET DES CONSEILS

### 1. État RÉEL du dataset

**Ce que tu as :** `C:/Dev/ffe_data_backup/`

| Élément | Valeur | État |
|---------|--------|------|
| Format | **HTML brut** | Pas prêt pour ML |
| Taille | ~2.15 GB | Correct |
| Fichiers | 85,672 | Énorme travail de parsing |
| Période | 2002-2026 (25 saisons) | Excellent |
| Échiquiers estimés | ~650,000-850,000 | Largement suffisant |

**Problème critique :** Le dataset n'est **PAS** un CSV/Parquet prêt à l'emploi. C'est du HTML brut qu'il faut parser avant tout entraînement.

---

### 2. Critique rigoureuse des conseils reçus

#### Ce qui est **CORRECT** :

| Conseil | Verdict | Commentaire |
|---------|---------|-------------|
| XGBoost pour données tabulaires | **Valide** | Confirmé par benchmarks |
| `tree_method='hist'` | **Valide** | Plus rapide pour gros datasets |
| `early_stopping` | **Valide** | Best practice standard |
| `subsample` pour accélérer | **Valide** | 0.8 est un bon défaut |
| scikit-learn pour preprocessing | **Valide** | Standard |
| pandas avec chunks | **Valide** | Nécessaire pour 2GB |
| PuLP ou OR-Tools pour optimisation | **Valide** | OR-Tools mieux pour assignment |

#### Ce qui est **INCORRECT ou INCOMPLET** :

| Conseil | Problème | Correction |
|---------|----------|------------|
| "Charge mon dataset (CSV)" | **FAUX** - C'est du HTML ! | Il faut **parser d'abord** |
| "700k parties, 2 GB" | **Imprécis** - C'est 85k fichiers HTML | ~750k échiquiers **après parsing** |
| `pd.read_csv(..., chunksize=100000)` | **Inapplicable** | Le parsing HTML vient avant |
| "15-90 min d'entraînement" | **Sous-estimé** | Parsing = des heures, training = 15-90 min |
| "XGBClassifier" seul | **Incomplet** | CatBoost peut être meilleur (catégories) |

#### Ce qui **MANQUE** dans les conseils :

1. **Phase de parsing HTML** (plusieurs heures de travail)
2. **Normalisation des noms de joueurs** (accents, casse, doublons)
3. **Gestion des Elo = 0** (imputation ou exclusion)
4. **Période COVID 2020** à traiter spécialement
5. **Validation croisée temporelle** (pas random split !)

---

### 3. Choix du modèle ML - Analyse approfondie

D'après les benchmarks 2025 ([Neptune.ai](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm), [TabArena](https://valeman.medium.com/catboost-vs-xgboost-why-catboost-is-stealing-the-crown-in-tabular-data-2d1c2f96afbc)) :

| Critère | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Données catégorielles** | Encodage requis | Limité | **Natif (meilleur)** |
| **Vitesse entraînement** | Modérée | **Rapide** | Modérée |
| **Vitesse inférence** | Modérée | Modérée | **30-60x plus rapide** |
| **Tuning requis** | Beaucoup | Modéré | **Minimal** |
| **Accuracy benchmarks 2025** | Bon | Bon | **+20% vs XGBoost** |
| **Petits datasets** | Risque overfit | Risque overfit | **Robuste** |

**Ton dataset a des catégories importantes :**
- `licence` : A, B, C, J, P
- `categorie` : Sen, Vet, Jun, Cad, Min...
- `division` : N1, N2, N3, R1, D1...
- `ligue_code` : IDF, PACA, HDF...
- `titre_fide` : GM, IM, FM...

**Recommandation :**

```
1er choix : CatBoost (gestion native catégories, moins de tuning)
2ème choix : XGBoost (plus documenté, communauté plus large)
3ème choix : LightGBM (si contraintes mémoire fortes)
```

**Librairies à installer :**
```
catboost>=1.2
xgboost>=2.0
lightgbm>=4.0  # optionnel, backup
```

---

### 4. Étapes d'entraînement RÉELLES (corrigées)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET ALICE                           │
└─────────────────────────────────────────────────────────────────────┘

PHASE 0 : PRÉPARATION (tu es ici)
═══════════════════════════════════
 0.1 Copier dataset dans Alice-Engine
 0.2 Installer dépendances Python
 0.3 Vérifier parse.py existant

PHASE 1 : PARSING HTML → CSV/Parquet (Durée: 2-4 heures)
═════════════════════════════════════════════════════════
 1.1 Parser calendrier.html → matchs avec dates, lieux, scores
 1.2 Parser ronde_N.html → échiquiers avec joueurs, Elo, résultats
 1.3 Parser players/ → référentiel joueurs FFE
 1.4 Normaliser noms (majuscules, accents, doublons)
 1.5 Gérer Elo = 0 (imputation ou flag)
 1.6 Exporter en Parquet (plus efficace que CSV pour 750k lignes)

     Résultat: echiquiers.parquet (~150-300 MB)
               joueurs.parquet (~5 MB)

PHASE 2 : FEATURE ENGINEERING (Durée: 1-2 heures)
══════════════════════════════════════════════════
 2.1 Features directes (extraites du parsing):
     - saison, ronde, echiquier
     - elo_blanc, elo_noir, diff_elo
     - division, niveau, ligue_code
     - titre_fide_blanc, titre_fide_noir

 2.2 Features dérivées (à calculer):
     - taux_presence_joueur (% matchs joués / matchs équipe)
     - forme_recente (score sur 5 derniers matchs)
     - echiquier_habituel (position moyenne)
     - historique_face_a_face (si disponible)
     - domicile (bool)
     - enjeu_match (maintien/titre/milieu)

 2.3 Features de fiabilité (extraites des non_joue/forfaits):
     ─────────────────────────────────────────────────────────
     Ces données "vides" révèlent des patterns d'indisponibilité
     exploitables pour ALI (prédiction lineup adverse).

     Par club:
     - taux_forfait_club : % forfaits historiques du club
     - taux_non_joue_club : % matchs non joués (capacité à aligner)
     - fiabilite_club : score composite de régularité

     Par joueur:
     - taux_presence_joueur : % présence sur matchs possibles
     - pattern_dispo_mois[1-12] : taux présence par mois
       (détecte vacances, contraintes saisonnières)
     - pattern_dispo_jour[Lun-Dim] : taux présence par jour semaine
       (détecte contraintes professionnelles)
     - derniere_presence : nombre de rondes depuis dernier match
     - joueur_fantome : flag si < 20% présence sur 2 saisons

     Exemple d'extraction:
     ```python
     # Fiabilité club
     df_club = df.groupby('equipe').agg({
         'type_resultat': lambda x: x.isin([
             'forfait_blanc', 'forfait_noir', 'double_forfait'
         ]).mean()
     }).rename(columns={'type_resultat': 'taux_forfait_club'})

     # Pattern mensuel joueur
     df_player_month = df.groupby(
         ['blanc_nom', df['date'].dt.month]
     )['type_resultat'].apply(
         lambda x: (x == 'non_joue').mean()
     ).unstack(fill_value=0)
     ```

     Impact ALI:
     - Pondère les probabilités de présence joueur
     - Identifie les clubs susceptibles d'aligner équipes incomplètes
     - Affine prédictions selon période/date du match

 2.4 Encodage catégories:
     - CatBoost : rien à faire (natif)
     - XGBoost : LabelEncoder ou OrdinalEncoder

PHASE 3 : SPLIT DONNÉES (Durée: 5 min)
═══════════════════════════════════════
 ⚠️ ATTENTION : PAS de random split !

 Utiliser TimeSeriesSplit ou split temporel:
 - Train : 2002-2023 (80%)
 - Validation : 2024 (10%)
 - Test : 2025-2026 (10%)

 Pourquoi ? Éviter le data leakage temporel.

PHASE 4 : ENTRAÎNEMENT MODÈLE (Durée: 15-60 min)
═════════════════════════════════════════════════
 4.1 Modèle ALI (Adversarial Lineup Inference):
     Problème : Classification binaire multi-label
     Target : joueur_a_joue (0 ou 1)

     CatBoost:
     ─────────
     from catboost import CatBoostClassifier

     model = CatBoostClassifier(
         iterations=1000,
         learning_rate=0.05,
         depth=6,
         cat_features=['division', 'ligue_code', 'titre_fide'],
         early_stopping_rounds=50,
         eval_metric='AUC',
         task_type='CPU'  # ou 'GPU' si disponible
     )

     model.fit(X_train, y_train, eval_set=(X_val, y_val))

     XGBoost (alternative):
     ──────────────────────
     from xgboost import XGBClassifier

     model = XGBClassifier(
         n_estimators=1000,
         learning_rate=0.05,
         max_depth=6,
         tree_method='hist',
         early_stopping_rounds=50,
         eval_metric='auc',
         enable_categorical=True  # XGBoost 2.0+
     )

 4.2 Métriques à surveiller:
     - AUC-ROC (qualité des probabilités)
     - Precision@K (top K joueurs prédits)
     - Recall (ne pas rater les joueurs présents)

PHASE 5 : OPTIMISEUR COMPOSITION (CE)
═════════════════════════════════════
 Utiliser OR-Tools (meilleur que PuLP pour assignment):

 from ortools.sat.python import cp_model

 Problème : Assigner N joueurs à N échiquiers
 Contraintes :
   - Ordre Elo décroissant
   - 1 joueur par échiquier
   - Maximiser score attendu (via formule Elo)

PHASE 6 : SAUVEGARDE ET DÉPLOIEMENT
═══════════════════════════════════
 6.1 Sauvegarder modèle:
     model.save_model('models/ali_catboost_v1.cbm')
     # ou joblib.dump(model, 'models/ali_xgboost_v1.joblib')

 6.2 Sauvegarder feature importance:
     Graphique + liste des 20 features les plus importantes

 6.3 API FastAPI:
     Charger modèle au démarrage
     Endpoint /predict retourne probabilités
```

---

### 5. Résumé des dépendances Python recommandées

```python
# requirements.txt pour ALICE

# ML - Choix principal
catboost>=1.2.7          # Recommandé (catégories natives)
xgboost>=2.1.0           # Alternative solide

# Data processing
pandas>=2.0
numpy>=1.24
pyarrow>=14.0            # Pour Parquet (plus rapide que CSV)

# Preprocessing & metrics
scikit-learn>=1.4

# Optimisation lineup (CE)
ortools>=9.8             # Meilleur que PuLP pour assignment

# Parsing HTML
beautifulsoup4>=4.12
lxml>=5.0                # Parser rapide

# API
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.0

# Utils
tqdm>=4.66               # Progress bars
joblib>=1.3              # Sauvegarde modèles
```

---

### 6. Auto-evaluation ISO (docs/iso/ISO_STANDARDS_REFERENCE.md)

| Norme | Application | Statut |
|-------|-------------|--------|
| **ISO 25010** | Fiabilité - Analyse rigoureuse du dataset | ✅ |
| **ISO 25012** | Qualité données - Identifié Elo=0, COVID, doublons | ✅ |
| **ISO 29119** | Tests - Split temporel recommandé (pas random) | ✅ |
| **ISO 5055** | Qualité code - Dépendances modernes proposées | ✅ |

---

## Sources utilisées

- [Neptune.ai - When to Choose CatBoost](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm)
- [CatBoost vs XGBoost - TabArena Benchmarks](https://valeman.medium.com/catboost-vs-xgboost-why-catboost-is-stealing-the-crown-in-tabular-data-2d1c2f96afbc)
- [XGBoost Multi-Output Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)
- [OR-Tools vs PuLP Comparison](https://medium.com/operations-research-bit/mip-solvers-unleashed-a-beginners-guide-to-pulp-cplex-gurobi-google-or-tools-and-pyomo-0150d4bd3999)
- [NBA Player Prediction with XGBoost](https://medium.com/ai-builder/predicting-nba-player-performance-with-xgboost-a-time-series-approach-7affce3ef614)

---

*Document généré le 3 Janvier 2026 - ALICE Engine Bootstrap*
