# Changelog - ALICE Engine

> **Format**: [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)
> **Versioning**: [Semantic Versioning](https://semver.org/lang/fr/)

---

## [Unreleased]

### Added
- **Module `scripts/ffe_rules_features.py`** - Implementation regles FFE
  - Types stricts: TypeCompetition, NiveauCompetition, Sexe, Joueur, Equipe
  - Detection type competition (A02, F01, C01, C03, C04, J02, J03, REG, DEP)
  - Calcul joueur brule avec seuils variables (1-4 matchs selon competition)
  - Calcul noyau (50% ou 2 absolu selon niveau)
  - Validation composition complete (ordre Elo, mutes, quotas)
  - Zones d'enjeu (montee, descente, danger, mi_tableau)
- **Tests unitaires** `tests/test_ffe_rules_features.py` (66 tests)
- **Features reglementaires ML** dans `feature_engineering.py`
  - Features: nb_equipes, niveau_max, niveau_min, type_competition, multi_equipe
  - Features equipe: zone_enjeu, niveau_hierarchique
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
- **Script parse_dataset.py** pour extraction donnees FFE
  - Parsing compositions (ronde_N.html, calendrier.html)
  - Parsing joueurs licencies (page_XXXX.html)
  - Export Parquet (echiquiers.parquet, joueurs.parquet)
  - 13,935 groupes parses, 1,736,490 echiquiers extraits
  - 35,320 joueurs licencies (dataset incomplet, voir BILAN_PARSING.md)
  - Champ `mute` (transfert club) ajoute pour reglements FFE
  - Mapping categories FFE legacy → officiel (U8, U10, X20, X50, X65)

### Changed
- Seuil coverage temporairement a 70% (objectif 80%)

### Known Issues
- Dataset joueurs incomplet (~47% manquants, surtout jeunes U08-U14)
- Voir `C:\Dev\ffe_scrapper\TODO_SCRAPING_JOUEURS_COMPLET.md` pour action

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

### [0.2.0] - Parsing Dataset ✅
- [x] Script parse_dataset.py
- [x] Export echiquiers.parquet (**1,736,490 lignes** - 34 MB)
- [x] Export joueurs.parquet (**35,320 lignes** - 1.6 MB)
- [x] Documentation BILAN_PARSING.md (ISO 25012)
- [x] Champ `mute` pour reglements FFE
- [ ] Nettoyage donnees (Elo=0: 18.2%, forfaits: 5%)
- [ ] Scraping joueurs complet (~66k) - voir ffe_scrapper

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

## Documentation associee

- [BILAN_PARSING.md](./BILAN_PARSING.md) - Resultats detailles du parsing

---

*Derniere mise a jour: 3 Janvier 2026*
