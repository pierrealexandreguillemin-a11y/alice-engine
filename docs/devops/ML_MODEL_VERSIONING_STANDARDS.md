# ML Model Versioning - Industry Standards

**Document ID**: DEVOPS-ML-VERSION-001
**Version**: 1.0.0
**Last Updated**: 2026-01-10
**ISO Compliance**: ISO 42001:2023 (AI Management), ISO 5259 (Data Quality)

## TL;DR - Ne PAS pousser les modèles ML sur Git

| Artefact | Push sur Git? | Solution recommandée |
|----------|---------------|---------------------|
| Code source | ✅ OUI | Git standard |
| Hyperparamètres (YAML) | ✅ OUI | Git standard |
| Modèles entraînés (.pkl, .pt, .h5) | ❌ NON | DVC, MLflow, Cloud Storage |
| Datasets | ❌ NON | DVC, Cloud Storage |
| Checkpoints | ❌ NON | MLflow, Cloud Storage |
| Embeddings | ❌ NON | Vector DB, Cloud Storage |

## Pourquoi ne pas pousser les modèles sur Git?

1. **Taille**: Modèles = 100MB - 10GB+ (Git limit ~100MB)
2. **Binaires**: Git optimisé pour texte, pas binaires
3. **Historique**: Chaque version stockée = explosion taille repo
4. **Clone**: `git clone` devient lent (télécharge tout l'historique)
5. **Coût**: GitHub LFS payant au-delà de 1GB

## Architecture Recommandée (Industry Standard 2025)

```
┌─────────────────────────────────────────────────────────────┐
│                         GIT REPO                             │
├─────────────────────────────────────────────────────────────┤
│  ✅ Code source (.py)                                        │
│  ✅ Configuration (hyperparameters.yaml)                     │
│  ✅ DVC files (.dvc) - pointeurs vers données               │
│  ✅ MLflow config (mlflow.yaml)                             │
│  ✅ Requirements (requirements.txt)                          │
│  ✅ Documentation (docs/)                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STOCKAGE EXTERNE                          │
├─────────────────────────────────────────────────────────────┤
│  📦 DVC Remote (S3, GCS, Azure Blob)                        │
│     └── Datasets versionnés                                  │
│     └── Features transformées                                │
│                                                              │
│  🧪 MLflow Tracking Server                                   │
│     └── Expériences                                          │
│     └── Métriques                                            │
│     └── Modèles versionnés                                   │
│     └── Artefacts                                            │
│                                                              │
│  ☁️ Cloud Storage (S3, GCS)                                  │
│     └── Modèles production                                   │
│     └── Checkpoints                                          │
│     └── Embeddings                                           │
└─────────────────────────────────────────────────────────────┘
```

## Outils Recommandés

### 1. DVC (Data Version Control) - Pour Données

```bash
# Installation
pip install dvc dvc-s3  # ou dvc-gs, dvc-azure

# Initialisation
dvc init
dvc remote add -d myremote s3://bucket/path

# Versionner un dataset
dvc add data/train.parquet
git add data/train.parquet.dvc .gitignore
git commit -m "Add training data v1"
dvc push
```

**Avantages**:
- Intégration Git native
- Support multi-cloud (S3, GCS, Azure, SSH)
- Pipelines reproductibles
- Léger (fichiers .dvc = quelques bytes)

### 2. MLflow - Pour Modèles et Expériences

```python
import mlflow

# Configuration
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("chess-prediction")

# Logging
with mlflow.start_run():
    mlflow.log_params(hyperparameters)
    mlflow.log_metrics({"auc": 0.85, "f1": 0.82})
    mlflow.sklearn.log_model(model, "model")

# Récupération
model = mlflow.sklearn.load_model("runs:/abc123/model")
```

**Avantages**:
- UI web intégrée
- Model Registry
- Déploiement intégré
- Auto-logging pour frameworks ML

### 3. Git LFS - Pour Petits Binaires (<1GB)

```bash
# Installation
git lfs install

# Tracking
git lfs track "*.pkl"
git lfs track "*.pt"
git add .gitattributes

# Usage normal
git add model.pkl
git commit -m "Add model"
git push
```

**Limites**:
- 1GB gratuit sur GitHub
- Payant au-delà
- Pas de versioning sémantique
- Pas de métadonnées ML

## Configuration pour Alice-Engine

### Actuel (MLflow)

```yaml
# config/hyperparameters.yaml - Versionné dans Git
mlflow:
  tracking_uri: "file:./mlruns"  # Local
  experiment_name: "alice-chess"

# Modèles stockés dans:
# - mlruns/ (local, gitignore)
# - models/ (production, gitignore)
```

### .gitignore recommandé

```gitignore
# ML Models - NE PAS VERSIONNER
models/**/*.pkl
models/**/*.joblib
models/**/*.pt
models/**/*.h5
models/**/*.onnx

# MLflow
mlruns/

# DVC cache
.dvc/cache/
.dvc/tmp/

# Datasets
data/**/*.parquet
data/**/*.csv
!data/.gitkeep

# Checkpoints
checkpoints/
*.ckpt
```

## Traçabilité ISO 42001

Pour maintenir la conformité ISO 42001 sans stocker les modèles:

```python
# Dans le code, stocker les références
model_reference = {
    "model_id": "alice-v1.2.3",
    "git_commit": "abc123",
    "mlflow_run_id": "xyz789",
    "data_hash": "sha256:...",
    "training_date": "2026-01-10",
    "metrics": {"auc": 0.85}
}

# Sauvegarder la référence (JSON dans Git)
with open("models/production/model_card.json", "w") as f:
    json.dump(model_reference, f)
```

## Pour ton projet RAG (EmbeddingGemma + MediaPipe)

```yaml
# Structure recommandée
rag-project/
├── .dvc/                    # DVC config
├── .gitignore
├── data/
│   ├── documents.dvc        # Pointeur DVC → documents
│   └── embeddings.dvc       # Pointeur DVC → embeddings
├── models/
│   ├── model_card.json      # Métadonnées (Git)
│   └── .gitkeep
├── src/
│   ├── embedding.py         # Code (Git)
│   └── retrieval.py
└── mlruns/                   # MLflow (gitignore)
```

**Embeddings**: Stocker dans Vector DB (Pinecone, Weaviate, ChromaDB) ou DVC.

## ALICE Engine — Implémentation T24 Phase 3 (2026-04-29)

`dvc.yaml` racine définit 2 stages reproducibles :

| Stage | Cmd | Outs (cached) | Outs (git-tracked) |
|-------|-----|---------------|--------------------|
| `refit_mlp_champion` | `python -m scripts.cloud.refit_mlp_champion` | `models/cache/mlp_meta_learner.joblib`, `temperature_T.joblib` | `models/cache/mlp_champion_metadata.json` (metric) |
| `backtest_holdout_2024` | `python -m scripts.backtest.run_holdout_2024` | — | `reports/backtest/ali_holdout_2024.json` (cache: false, persist: true) |

DAG : `refit_mlp_champion → backtest_holdout_2024` (mlp deps déclarées).

`dvc.lock` versionne les hashes md5 des deps (sources + parquets OOF) et outs.
Cela garantit la traçabilité commit git ↔ artefacts (D6/D7 partial — versioning
local DVC sans remote distant). Phase 5+ : `dvc remote add -d s3://...` pour
push artefacts vers stockage cloud (cf. architecture above).

Commandes :

```bash
dvc status                # vérifier reproducibilité
dvc repro <stage>         # rerun si deps changent
dvc commit -f <stage>     # snapshot artefacts existants sans rerun
```

**Limites résolues** : lineage commit ↔ MLP champion + backtest report.
**Limites restantes (Phase 5)** :
- Pas de remote DVC (cache local seulement)
- Pas de lien commit git ↔ version Kaggle dataset ↔ kernel (D7)
- Artefacts training Kaggle (CatBoost, XGB, LGB) non DVC-tracked

## Références

- [DVC vs Git vs Git LFS](https://censius.ai/blogs/dvc-vs-git-and-git-lfs-in-machine-learning-reproducibility)
- [ML Versioning with MLflow, DVC, GitHub](https://medium.com/@amitkharche/ml-versioning-with-mlflow-dvc-github-why-it-matters-for-delivery-leaders-8311f68d648d)
- [Git LFS and DVC: The Ultimate Guide](https://medium.com/@pablojusue/git-lfs-and-dvc-the-ultimate-guide-to-managing-large-artifacts-in-mlops-c1c926e6c5f4)
- [LakeFS - Model Versioning Best Practices](https://lakefs.io/blog/model-versioning/)
- [Data Versioning: ML Best Practices 2025](https://labelyourdata.com/articles/machine-learning/data-versioning)

---

**Décision Alice-Engine**: Modèles non versionnés dans Git. Utilisation de MLflow local + Model Cards JSON pour traçabilité ISO 42001.
