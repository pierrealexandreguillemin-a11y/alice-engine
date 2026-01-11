# Alice-Engine - Guide Claude

## Setup DevOps

Hooks pre-commit Python équivalents aux hooks Husky de `C:\Dev\chess-app`.

### Conformité ISO

**Normes Générales:**
- ISO 25010 (Qualité système)
- ISO 27001 (Sécurité)
- ISO 27034 (Secure coding)
- ISO 5055 (Qualité code)
- ISO 42010 (Architecture)
- ISO 29119 (Tests)
- ISO 15289 (Documentation)

**Normes ML/AI (ALICE Engine):**
- ISO/IEC 42001:2023 - AI Management System (Model Card, Traçabilité)
- ISO/IEC 23894:2023 - AI Risk Management
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 24029 - Neural Network Robustness
- ISO/IEC TR 24027 - Bias in AI

Voir `docs/iso/ISO_STANDARDS_REFERENCE.md` pour details et mapping fichiers -> normes.

## Fichiers clés

```
.pre-commit-config.yaml    # Hooks pre-commit/commit-msg/pre-push
pyproject.toml             # Config Ruff, MyPy, Bandit, Pytest, Commitizen
requirements-dev.txt       # Dépendances dev
Makefile                   # Commandes (make help)
```

## Scripts personnalisés

```
scripts/
├── generate_graphs.py      # Graphs SVG architecture (pydeps)
├── update_iso_docs.py      # Génère docs/iso/IMPLEMENTATION_STATUS.md
└── analyze_architecture.py # Score santé architecture (coupling, cycles)
```

## Hooks actifs

### Pre-commit
- Gitleaks (secrets P0-CRITICAL)
- Ruff (lint + format)
- MyPy (types)
- Bandit (sécurité)

### Commit-msg
- Commitizen (conventional commits)

### Pre-push
- Pytest + coverage >80%
- Xenon (complexité)
- pip-audit (vulnérabilités)
- `analyze_architecture.py`
- `generate_graphs.py`
- `update_iso_docs.py`

## Artefacts générés

```
graphs/dependencies.svg
reports/complexity/index.html
reports/architecture-health.json
docs/iso/IMPLEMENTATION_STATUS.md
```

## Commandes

```bash
make install    # Installer dépendances
make hooks      # Installer git hooks
make quality    # Lint + Format + Typecheck + Security
make test-cov   # Tests + coverage
make all        # Validation complète
make graphs     # Générer graphs SVG
make iso-docs   # MAJ documentation ISO
```

## Documentation

- `docs/PYTHON-HOOKS-SETUP.md` - Setup complet avec correspondances chess-app
- `docs/iso/ISO_STANDARDS_REFERENCE.md` - Normes ISO applicables
