# Guide de Contribution - ALICE Engine

> **Version**: 0.1.0
> **Date**: 3 Janvier 2026
> **Norme**: ISO 12207 - Software Life Cycle

---

## 1. Prerequis

### Environnement

- Python 3.11+
- Git
- Graphviz (pour les graphs)

### Installation

```powershell
# Cloner
git clone https://github.com/pierrealexandreguillemin-a11y/alice-engine.git
cd alice-engine

# Setup complet (Windows)
.\scripts\setup-dev.ps1

# Ou manuellement
pip install -r requirements.txt
pip install -r requirements-dev.txt
make hooks
```

---

## 2. Workflow Git

### Branches

| Branche | Usage |
|---------|-------|
| `master` | Production (protegee) |
| `feature/*` | Nouvelles fonctionnalites |
| `fix/*` | Corrections de bugs |
| `docs/*` | Documentation |

### Commits (Conventional Commits)

Format: `type(scope): description`

```bash
# Types autorises
feat     # Nouvelle fonctionnalite
fix      # Correction bug
docs     # Documentation
style    # Formatage (pas de changement code)
refactor # Refactoring
test     # Ajout/modification tests
chore    # Maintenance
```

Exemples:
```bash
git commit -m "feat(api): add /predict endpoint"
git commit -m "fix(composer): correct Elo probability calculation"
git commit -m "docs(readme): update installation instructions"
```

---

## 3. Hooks Git

### Pre-commit (automatique)

- Gitleaks (secrets)
- Ruff (lint + format)
- MyPy (types)
- Bandit (securite)

### Commit-msg (automatique)

- Commitizen (format conventional commits)

### Pre-push (automatique)

- Pytest (tests)
- Coverage >= 70%
- Xenon (complexite)
- pip-audit (vulnerabilites)

---

## 4. Standards de Code

### Style

- PEP 8 (enforce par Ruff)
- Line length: 100 caracteres
- Docstrings: Google style

### Typage

```python
# Obligatoire pour fonctions publiques
def calculate_probability(player_elo: int, opponent_elo: int) -> float:
    """Calcule la probabilite de victoire."""
    ...
```

### Tests

```python
# Nommage: test_<fonction>_<scenario>
def test_calculate_probability_equal_elo():
    """Joueurs de meme Elo = 50% chacun."""
    result = calculate_probability(1500, 1500)
    assert result == pytest.approx(0.5, rel=0.01)
```

---

## 5. Commandes utiles

```bash
# Qualite
make quality      # Lint + Format + Typecheck + Security
make lint         # Ruff check
make format       # Ruff format
make typecheck    # MyPy

# Tests
make test         # Tests unitaires
make test-cov     # Tests + coverage HTML

# Audit
make audit        # pip-audit
make complexity   # Radon + Xenon

# Architecture
make graphs       # Generer SVG
make architecture # Score sante
make iso-docs     # MAJ docs ISO

# Tout
make all          # Validation complete
```

---

## 6. Structure du code

```
alice-engine/
├── app/                    # Controller layer
│   ├── main.py             # FastAPI entry point
│   ├── config.py           # Settings (Pydantic)
│   └── api/
│       ├── routes.py       # HTTP endpoints
│       └── schemas.py      # Pydantic models
│
├── services/               # Service layer
│   ├── inference.py        # ALI - Prediction
│   ├── composer.py         # CE - Optimisation
│   └── data_loader.py      # Repository - MongoDB
│
├── models/                 # ML models (.cbm)
├── tests/                  # Pytest
├── scripts/                # Utilitaires
└── docs/                   # Documentation (ISO 15289)
```

---

## 7. Pull Request

### Checklist

- [ ] Tests passes localement (`make test`)
- [ ] Qualite OK (`make quality`)
- [ ] Coverage >= 70%
- [ ] Commit messages conventionnels
- [ ] Documentation mise a jour si necessaire

### Template PR

```markdown
## Description
[Description courte du changement]

## Type
- [ ] feat
- [ ] fix
- [ ] docs
- [ ] refactor
- [ ] test

## Tests
- [ ] Tests unitaires ajoutes/modifies
- [ ] Coverage maintenu >= 70%

## Checklist
- [ ] Code lint OK
- [ ] Types OK
- [ ] Security OK
```

---

*Derniere mise a jour: 3 Janvier 2026*
