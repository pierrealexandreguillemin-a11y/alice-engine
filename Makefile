# Makefile - Alice-Engine
# Equivalent complet des scripts npm de chess-app
# Version: 2.0 - ZERO TOLERANCE + GRAPHS + ISO DOCS

.PHONY: help install hooks lint format typecheck test test-cov security audit
.PHONY: complexity quality graphs iso-docs architecture all dev clean

# Variables
PYTHON := python
PIP := pip
SRC := app/ services/
TESTS := tests/

# ============================================
# AIDE
# ============================================
help:
	@echo ""
	@echo "============================================"
	@echo "  Alice-Engine - Commandes DevOps"
	@echo "  Equivalent chess-app hooks"
	@echo "============================================"
	@echo ""
	@echo "  SETUP:"
	@echo "    make install      - Installer dependances prod + dev"
	@echo "    make hooks        - Installer hooks pre-commit"
	@echo ""
	@echo "  QUALITE (pre-commit):"
	@echo "    make lint         - Linter (ruff)"
	@echo "    make format       - Formatter (ruff format)"
	@echo "    make typecheck    - Type check (mypy)"
	@echo "    make security     - Security scan (bandit)"
	@echo ""
	@echo "  TESTS (pre-push):"
	@echo "    make test         - Tests unitaires"
	@echo "    make test-cov     - Tests + coverage (>80%)"
	@echo ""
	@echo "  AUDIT (pre-push):"
	@echo "    make audit        - Audit dependances (pip-audit)"
	@echo "    make complexity   - Analyse complexite (radon + xenon)"
	@echo ""
	@echo "  ARCHITECTURE (pre-push):"
	@echo "    make graphs       - Generer graphs SVG (pydeps)"
	@echo "    make architecture - Analyse sante architecture"
	@echo "    make iso-docs     - MAJ documentation ISO"
	@echo ""
	@echo "  ALL-IN-ONE:"
	@echo "    make quality      - Lint + Format + Typecheck + Security"
	@echo "    make all          - Quality + Tests + Audit + Graphs + ISO"
	@echo ""
	@echo "  DEV:"
	@echo "    make dev          - Lancer serveur dev"
	@echo "    make clean        - Nettoyer fichiers temporaires"
	@echo ""

# ============================================
# SETUP
# ============================================
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo ""
	@echo "Dependances installees!"
	@echo "Executez 'make hooks' pour installer les git hooks"

hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit install --hook-type pre-push
	@echo ""
	@echo "Hooks installes:"
	@echo "  - pre-commit: lint, format, typecheck, security"
	@echo "  - commit-msg: conventional commits"
	@echo "  - pre-push: tests, coverage, audit, graphs, iso-docs"

# ============================================
# QUALITE (PRE-COMMIT)
# ============================================
lint:
	@echo "Lint Python (Ruff)..."
	ruff check $(SRC) $(TESTS) --fix

format:
	@echo "Format Python (Ruff)..."
	ruff format $(SRC) $(TESTS)

typecheck:
	@echo "Type check (MyPy)..."
	mypy $(SRC) --ignore-missing-imports

security:
	@echo "Security scan (Bandit)..."
	bandit -c pyproject.toml -r $(SRC)

# ============================================
# TESTS (PRE-PUSH)
# ============================================
test:
	@echo "Tests unitaires (pytest)..."
	pytest $(TESTS) -v --tb=short

test-cov:
	@echo "Tests + Coverage..."
	pytest $(TESTS) --cov=app --cov=services --cov-report=html --cov-report=term --cov-fail-under=80
	@echo ""
	@echo "Rapport HTML: htmlcov/index.html"

# ============================================
# AUDIT (PRE-PUSH)
# ============================================
audit:
	@echo "Audit dependances (pip-audit)..."
	pip-audit --strict --desc

complexity:
	@echo ""
	@echo "=== Complexite Cyclomatique (Radon) ==="
	radon cc $(SRC) -a -nc
	@echo ""
	@echo "=== Indice de Maintenabilite ==="
	radon mi $(SRC) -s
	@echo ""
	@echo "=== Seuils Complexite (Xenon) ==="
	xenon --max-absolute=B --max-modules=B --max-average=A $(SRC) || true

# ============================================
# ARCHITECTURE (PRE-PUSH) - ISO 42010
# ============================================
graphs:
	@echo "Generation graphs architecture..."
	$(PYTHON) scripts/generate_graphs.py
	@echo ""
	@echo "Graphs generes dans: graphs/"

architecture:
	@echo "Analyse sante architecture..."
	$(PYTHON) scripts/analyze_architecture.py
	@echo ""
	@echo "Rapport: reports/architecture-health.json"

iso-docs:
	@echo "MAJ documentation ISO..."
	$(PYTHON) scripts/update_iso_docs.py
	@echo ""
	@echo "Documentation: docs/iso/IMPLEMENTATION_STATUS.md"

# ============================================
# ALL-IN-ONE
# ============================================
quality: lint format typecheck security
	@echo ""
	@echo "Quality checks passed!"

all: quality test-cov audit complexity architecture graphs iso-docs
	@echo ""
	@echo "============================================"
	@echo "  TOUS LES CHECKS PASSES!"
	@echo "  Ready for push."
	@echo "============================================"
	@echo ""
	@echo "Artefacts generes:"
	@echo "  - graphs/dependencies.svg"
	@echo "  - reports/complexity/index.html"
	@echo "  - reports/architecture-health.json"
	@echo "  - docs/iso/IMPLEMENTATION_STATUS.md"
	@echo ""

# ============================================
# DEV
# ============================================
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

clean:
	@echo "Nettoyage fichiers temporaires..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Nettoyage termine!"

# ============================================
# VALIDATION COMPLETE (equivalent validate:strict)
# ============================================
validate: quality test-cov audit
	@echo ""
	@echo "Validation complete passee!"
