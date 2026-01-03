# Alice-Engine - Guide Claude

## Setup DevOps

Hooks pre-commit Python équivalents aux hooks Husky de `C:\Dev\chess-app`.
Conformité ISO 25010, 27001, 5055, 42010, 29119.

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
- `iso-standards-reference.md` - Normes ISO applicables
