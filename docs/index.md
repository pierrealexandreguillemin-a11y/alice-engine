# ALICE Engine

**Adversarial Lineup Inference & Composition Engine**

ALICE is a Machine Learning engine for chess team composition optimization and game outcome prediction.

## Features

- **ML Pipeline**: CatBoost, XGBoost, LightGBM ensemble
- **Explainability**: SHAP-based feature importance (ISO 42001)
- **Robustness**: OOD detection, drift monitoring (ISO 24029)
- **Security**: Model signing, encryption (ISO 27001)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test-cov

# Train model
python -m scripts.train_models_parallel
```

## ISO Compliance

ALICE implements multiple ISO standards:

| Standard | Domain | Status |
|----------|--------|--------|
| ISO 5055 | Code Quality | ✅ |
| ISO 27001 | Security | ✅ |
| ISO 42001 | AI Management | ✅ |
| ISO 29119 | Testing | ✅ |

See [ISO Standards Reference](iso/ISO_STANDARDS_REFERENCE.md) for details.

## Documentation

- [API Contract](api/API_CONTRACT.md) - REST API documentation
- [API Reference](api/reference.md) - Auto-generated from docstrings
- [ISO Standards](iso/ISO_STANDARDS_REFERENCE.md) - Compliance mapping
- [Deployment](operations/DEPLOIEMENT_RENDER.md) - Render deployment guide
