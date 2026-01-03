# Changelog - ALICE Engine

> **Format**: [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)
> **Versioning**: [Semantic Versioning](https://semver.org/lang/fr/)

---

## [Unreleased]

### Added
- Structure projet complete (ISO 42010 SRP)
- API FastAPI avec endpoints /health et /predict
- Service ALI (Adversarial Lineup Inference) - placeholder
- Service CE (Composition Engine) avec calcul Elo
- Schemas Pydantic v2 pour validation
- Configuration Render (render.yaml)
- Configuration Vercel (vercel.json)
- Pre-commit hooks complets (Gitleaks, Ruff, MyPy, Bandit)
- Pre-push hooks (Pytest, Coverage, Xenon, pip-audit)
- Scripts DevOps (graphs, architecture, ISO docs)
- Documentation ISO 15289/26514 reorganisee

### Changed
- Seuil coverage temporairement a 70% (objectif 80%)

### Fixed
- Erreurs MyPy dans schemas.py
- Erreurs Ruff (S104, N802, S603)
- Vulnerabilites dependances (starlette, urllib3, etc.)

---

## [0.1.0] - 2026-01-03

### Added
- **Initial release**
- Structure de base du projet
- README.md avec instructions
- pyproject.toml (Ruff, MyPy, Pytest, Bandit)
- requirements.txt et requirements-dev.txt
- .gitignore adapte Python/ML
- Lien symbolique vers dataset FFE (2.4 GB)

### Documentation
- ANALYSE_INITIALE_ALICE.md
- CDC_ALICE.md (cahier des charges)
- CONTEXTE_DATASET_FFE.md
- CONTEXTE_INTEGRATION.md
- API_CONTRACT.md
- DEPLOIEMENT_RENDER.md
- PYTHON-HOOKS-SETUP.md

---

## Roadmap

### [0.2.0] - Parsing Dataset
- [ ] Script parse_dataset.py
- [ ] Export echiquiers.parquet (~750k lignes)
- [ ] Export joueurs.parquet (~55k lignes)
- [ ] Nettoyage donnees (Elo=0, forfaits)

### [0.3.0] - Entrainement Modele
- [ ] Feature engineering
- [ ] Entrainement CatBoost
- [ ] Validation croisee
- [ ] Export modele .cbm

### [0.4.0] - Integration chess-app
- [ ] Connexion MongoDB Atlas
- [ ] Endpoint /predict fonctionnel
- [ ] Tests integration

### [1.0.0] - Production
- [ ] Deploiement Render
- [ ] Coverage >= 80%
- [ ] Documentation complete
- [ ] Performance benchmarks

---

*Derniere mise a jour: 3 Janvier 2026*
