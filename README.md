# ALICE - Adversarial Lineup Inference & Composition Engine

> *"J'ai fait les compos avec Alice"*

Module predictif pour l'application chess-app. Predit les compositions adverses et optimise les compositions d'equipes pour les interclubs d'echecs FFE.

## Composants

| Composant | Acronyme | Fonction |
|-----------|----------|----------|
| **Adversarial Lineup Inference** | ALI | Predire la composition adverse |
| **Composition Engine** | CE | Optimiser sa propre composition |

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- Git

### Installation

```powershell
# Cloner le repo
git clone https://github.com/VOTRE_USER/alice-engine.git
cd alice-engine

# Creer environnement virtuel
python -m venv venv

# Activer (Windows PowerShell)
.\venv\Scripts\Activate

# Installer dependances
pip install -r requirements.txt
```

### Configuration

```powershell
# Copier le fichier exemple
copy .env.example .env

# Editer avec vos valeurs (MongoDB URI, API_KEY)
notepad .env
```

### Lancement

```powershell
# Demarrer le serveur
uvicorn app.main:app --reload --port 8000

# Tester
curl http://localhost:8000/health
```

### Tests

```powershell
# Lancer les tests
pytest

# Avec coverage
pytest --cov=app --cov=services --cov-report=html
```

### Qualite code (ISO 5055)

```powershell
# Linting
ruff check .

# Auto-fix
ruff check --fix .

# Type checking
mypy app services

# Audit securite
pip-audit
```

## Documentation API

| Endpoint | URL |
|----------|-----|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| OpenAPI JSON | http://localhost:8000/openapi.json |

## Architecture (ISO 42010 - SRP)

```
alice-engine/
├── app/                        # Controller layer (HTTP)
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration (ISO 27001)
│   └── api/
│       ├── __init__.py
│       ├── routes.py           # HTTP routes
│       └── schemas.py          # Pydantic validation (ISO 25012)
│
├── services/                   # Service layer (logique pure)
│   ├── __init__.py
│   ├── inference.py            # ALI - Prediction adverse
│   ├── composer.py             # CE - Optimisation composition
│   └── data_loader.py          # Repository (I/O MongoDB)
│
├── models/                     # Modeles ML entraines (.cbm, .joblib)
│   └── .gitkeep
│
├── tests/                      # Tests (ISO 29119)
│   ├── __init__.py
│   ├── test_health.py          # Tests API
│   └── test_composer.py        # Tests services
│
├── scripts/                    # Scripts utilitaires
│   └── __init__.py
│
├── docs/                       # Documentation
│   ├── ANALYSE_INITIALE_ALICE.md
│   ├── API_CONTRACT.md
│   ├── CDC_ALICE.md
│   ├── CONTEXTE_DATASET_FFE.md
│   ├── CONTEXTE_INTEGRATION.md
│   └── INSTRUCTIONS_PROJET.md
│
├── dataset_alice/              # Lien symbolique -> ffe_data_backup (2.4 GB)
│
├── .env.example                # Template configuration
├── .gitignore                  # Fichiers ignores
├── pyproject.toml              # Config ruff, mypy, pytest
├── pytest.ini                  # Config pytest
├── requirements.txt            # Dependances Python
├── render.yaml                 # Config deploiement Render
├── docs/iso/                   # Documentation ISO (normes, conformite)
└── README.md                   # Ce fichier
```

## Standards ISO

| Norme | Focus | Implementation |
|-------|-------|----------------|
| **ISO 27001** | Secrets | `.env`, `config.py` |
| **ISO 27034** | Securite code | Validation Pydantic, OWASP |
| **ISO 5055** | Qualite code | ruff, mypy |
| **ISO 25012** | Qualite donnees | Schemas Pydantic |
| **ISO 29119** | Tests | pytest, 80% unitaires |
| **ISO 42010** | Architecture | SRP, layers separes |

## Deploiement

### Render (Production)

Le fichier `render.yaml` configure le deploiement automatique.

| Environnement | URL |
|---------------|-----|
| Production | `https://alice-engine.onrender.com` |
| Health | `https://alice-engine.onrender.com/health` |
| Docs | `https://alice-engine.onrender.com/docs` |

### Variables d'environnement (Render Dashboard)

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | URI MongoDB Atlas (readonly) |
| `API_KEY` | Cle API pour /train |
| `LOG_LEVEL` | INFO ou DEBUG |

### CI/CD

- Auto-deploy depuis `main` branch
- Health check: `/health`
- Region: Frankfurt (eu-central)

## Dataset

Le dataset FFE (2002-2026) est disponible via lien symbolique :

| Metrique | Valeur |
|----------|--------|
| Taille | 2.4 GB (HTML brut) |
| Echiquiers | ~750,000 |
| Joueurs | ~55,000 |
| Saisons | 25 (2002-2026) |

**Note**: Le dataset n'est pas versionne (trop gros). Voir `docs/CONTEXTE_DATASET_FFE.md`.

## License

Proprietaire - Tous droits reserves

---

*Version 0.1.0 - 3 Janvier 2026*
