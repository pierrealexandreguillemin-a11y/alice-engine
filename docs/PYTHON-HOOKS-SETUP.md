# Configuration Hooks Pre-commit Python

> Equivalents Python **COMPLETS** des hooks Husky de chess-app (TypeScript)
> Version: 2.0 - ZERO TOLERANCE + GRAPHS + ISO DOCS

---

## Correspondance des outils

| Fonction | TypeScript (chess-app) | Python (Alice-Engine) |
|----------|------------------------|----------------------|
| Hook manager | Husky | **pre-commit** |
| Linter | ESLint | **Ruff** |
| Formatter | Prettier | **Ruff format** |
| Type checker | TypeScript (`tsc`) | **MyPy** |
| Staged files | lint-staged | **pre-commit** (natif) |
| Secret scanner | Gitleaks | **Gitleaks** |
| Commit messages | Commitlint | **Commitizen** |
| Tests | Vitest | **Pytest** |
| Coverage | c8 | **pytest-cov** |
| Security audit | npm audit / Snyk | **pip-audit** / **Bandit** |
| Complexity | eslint-complexity / plato | **Radon** / **Xenon** |
| Duplication | jscpd | **pylint --duplicate-code** |
| Circular deps | Madge | **pydeps** |
| Architecture graphs | madge --image | **pydeps --cluster** |
| API docs | TypeDoc | **pdoc** |
| ISO docs auto | update-iso-docs.cjs | **update_iso_docs.py** |
| Architecture health | analyze-architecture-health.js | **analyze_architecture.py** |

---

## Conformite ISO

| Norme | Focus | Outils Python |
|-------|-------|---------------|
| **ISO 27001** | Securite info | Gitleaks, pip-audit, Bandit |
| **ISO 27034** | Secure coding | Bandit, Ruff (S rules) |
| **ISO 5055** | Qualite code | Ruff, MyPy, Radon |
| **ISO 25010** | Qualite systeme | Radon, Xenon, pytest-cov |
| **ISO 29119** | Tests logiciels | Pytest, pytest-cov |
| **ISO 42010** | Architecture | pydeps, analyze_architecture.py |
| **ISO 12207** | Cycle de vie | pre-commit, Commitizen |

---

## Installation rapide

```bash
# 1. Installer dependances
pip install -r requirements-dev.txt

# 2. Installer hooks
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# 3. (Optionnel) Graphviz pour SVG
# Windows: choco install graphviz
# Mac: brew install graphviz
# Linux: apt install graphviz

# Ou via Makefile:
make install && make hooks
```

---

## Hooks actifs

### Pre-commit (a chaque commit)

| Hook | Outil | Norme ISO |
|------|-------|-----------|
| Secret scanning | Gitleaks | ISO 27001 (P0-CRITICAL) |
| Lint + Fix | Ruff | ISO 25010, ISO 5055 |
| Format | Ruff format | ISO 25010 |
| Type check | MyPy | ISO 5055 |
| Security scan | Bandit | ISO 27034 |
| Debug statements | pre-commit-hooks | ISO 25010 |
| No commit to main | pre-commit-hooks | ISO 12207 |

### Commit-msg

| Hook | Outil | Format |
|------|-------|--------|
| Conventional commits | Commitizen | feat/fix/docs/refactor... |

### Pre-push (avant push)

| Hook | Outil | Norme ISO |
|------|-------|-----------|
| Tests unitaires | Pytest | ISO 29119 |
| Coverage >80% | pytest-cov | ISO 29119 |
| Complexite seuils | Xenon | ISO 25010 |
| Audit deps | pip-audit | ISO 27001 |
| Sante architecture | analyze_architecture.py | ISO 42010 |
| Graphs SVG | generate_graphs.py | ISO 42010 |
| ISO docs | update_iso_docs.py | ISO 12207 |

---

## Artefacts generes

```
Alice-Engine/
├── graphs/
│   ├── dependencies.svg       # Graph dependances (pydeps)
│   ├── imports.svg            # Structure imports
│   └── circular-imports.txt   # Detection cycles
├── reports/
│   ├── complexity/
│   │   ├── index.html         # Rapport HTML
│   │   ├── cyclomatic.txt     # Complexite cyclomatique
│   │   └── maintainability.txt # Indice maintenabilite
│   ├── duplication/
│   │   └── duplicates.txt     # Code duplique
│   └── architecture-health.json # Score sante
├── docs/
│   └── iso/
│       └── IMPLEMENTATION_STATUS.md  # Conformite ISO auto
└── htmlcov/
    └── index.html             # Rapport coverage
```

---

## Commandes Makefile

```bash
# === QUALITE (pre-commit) ===
make lint         # Linter Ruff
make format       # Formatter Ruff
make typecheck    # Type check MyPy
make security     # Security Bandit
make quality      # Tout ci-dessus

# === TESTS (pre-push) ===
make test         # Tests unitaires
make test-cov     # Tests + coverage

# === AUDIT (pre-push) ===
make audit        # pip-audit
make complexity   # Radon + Xenon

# === ARCHITECTURE (pre-push) ===
make graphs       # Generer SVG
make architecture # Analyse sante
make iso-docs     # MAJ docs ISO

# === TOUT ===
make all          # Validation complete
make validate     # Quality + Tests + Audit
```

---

## Scripts Python

### scripts/generate_graphs.py

Genere les graphs d'architecture :
- `graphs/dependencies.svg` - Dependances modules (pydeps)
- `graphs/imports.svg` - Structure imports
- `reports/complexity/index.html` - Rapport complexite HTML
- Detection imports circulaires

```bash
python scripts/generate_graphs.py
```

### scripts/update_iso_docs.py

Genere `docs/iso/IMPLEMENTATION_STATUS.md` :
- Score DevOps global
- Conformite par norme ISO
- Modules installes/manquants
- Metriques qualite actuelles

```bash
python scripts/update_iso_docs.py
```

### scripts/analyze_architecture.py

Analyse sante architecture :
- Metriques coupling (afferent/efferent)
- Detection hot spots
- Score sante (0-100)
- Recommandations

```bash
python scripts/analyze_architecture.py
```

---

## Comparaison chess-app vs Alice-Engine

### Pre-commit

| Fonctionnalite | chess-app | Alice-Engine |
|----------------|-----------|--------------|
| Gitleaks | OK | OK |
| Lint + Fix | ESLint | Ruff |
| Format | Prettier | Ruff format |
| Type check | tsc | MyPy |
| Type assertions check | Custom grep | MyPy strict |
| Catch vides check | Custom grep | - (Python exceptions) |
| Zero warnings | --max-warnings=0 | Ruff strict |
| Archivage sessions | Custom bash | - |

### Pre-push

| Fonctionnalite | chess-app | Alice-Engine |
|----------------|-----------|--------------|
| TypeScript check | tsc --noEmit | mypy |
| Tests unitaires | Vitest | Pytest |
| Tests E2E | Playwright | - (a ajouter) |
| Deps circulaires | Madge | pydeps |
| Duplication | jscpd | pylint |
| Complexite | analyze-complexity.cjs | Radon/Xenon |
| Architecture health | analyze-architecture-health.js | analyze_architecture.py |
| Graphs SVG | generate-all-graphs.sh | generate_graphs.py |
| ISO docs | update-iso-docs.cjs | update_iso_docs.py |
| Audit deps | npm audit | pip-audit |

---

## Seuils de qualite (ZERO TOLERANCE)

| Metrique | Seuil | Norme |
|----------|-------|-------|
| Erreurs lint | 0 | ISO 5055 |
| Warnings lint | 0 | ISO 25010 |
| Erreurs type | 0 | ISO 5055 |
| Coverage tests | >80% | ISO 29119 |
| Complexite max | B (6-10) | ISO 25010 |
| Complexite moyenne | A (1-5) | ISO 25010 |
| Vulnerabilites | 0 critical | ISO 27001 |
| Imports circulaires | 0 | ISO 42010 |
| Secrets detectes | 0 | ISO 27001 |

---

## Workflow Git

```
git add .
    |
    v
[PRE-COMMIT]
    ├── Gitleaks (secrets)
    ├── Ruff (lint + format)
    ├── MyPy (types)
    ├── Bandit (security)
    └── BLOQUE si erreur
    |
    v
git commit -m "feat: ..."
    |
    v
[COMMIT-MSG]
    ├── Commitizen (format)
    └── BLOQUE si non-conventionnel
    |
    v
git push
    |
    v
[PRE-PUSH]
    ├── Pytest (tests)
    ├── Coverage (>80%)
    ├── Xenon (complexity)
    ├── pip-audit (deps)
    ├── analyze_architecture.py
    ├── generate_graphs.py
    ├── update_iso_docs.py
    └── BLOQUE si echec
    |
    v
Push OK!
```

---

*Derniere MAJ: 2025-01-03*
*Score DevOps cible: 100/100*
