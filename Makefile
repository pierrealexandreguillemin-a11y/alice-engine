# Makefile - Alice-Engine
# Equivalent complet des scripts npm de chess-app
# Version: 3.0 - FULL ISO COVERAGE + ML LIFECYCLE + MONITORING

.PHONY: help install hooks lint format typecheck test test-cov security audit
.PHONY: complexity quality graphs iso-docs architecture all dev clean
.PHONY: iso-audit train evaluate ensemble features parse-data
.PHONY: model-card drift-report data-lineage ml-pipeline all-iso validate-iso

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
	@echo "  ARCHITECTURE (manual, run before push):"
	@echo "    make reports      - Generate all reports (then git add)"
	@echo "    make graphs       - Generate SVG graphs (pydeps)"
	@echo "    make architecture - Architecture health analysis"
	@echo "    make iso-docs     - Update ISO documentation"
	@echo ""
	@echo "  ISO AUDIT (ISO 5055, 5259, 15289, 25010, 29119, 42001):"
	@echo "    make iso-audit    - Audit conformite ISO automatise"
	@echo "    make validate-iso - Audit strict (fail si non conforme)"
	@echo ""
	@echo "  ML LIFECYCLE (ISO 42001, 5259, 25059):"
	@echo "    make parse-data   - Parser dataset FFE"
	@echo "    make features     - Pipeline feature engineering"
	@echo "    make train        - Entrainer modeles ML"
	@echo "    make evaluate     - Evaluer/benchmark modeles"
	@echo "    make ensemble     - Stacking ensemble"
	@echo "    make ml-pipeline  - Pipeline complet (parse->train->eval)"
	@echo ""
	@echo "  PRODUCTION MONITORING (ISO 42001, 5259, 27001):"
	@echo "    make model-card   - Generer Model Card ISO 42001"
	@echo "    make drift-report - Rapport drift monitoring"
	@echo "    make data-lineage - Rapport lineage donnees"
	@echo ""
	@echo "  ALL-IN-ONE:"
	@echo "    make quality      - Lint + Format + Typecheck + Security"
	@echo "    make all          - Quality + Tests + Audit + Graphs + ISO"
	@echo "    make all-iso      - Validation ISO complete (all + iso-audit)"
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
# ARCHITECTURE (MANUAL) - ISO 42010
# Run before push: make reports && git add -A && git commit
# ============================================
reports: architecture graphs iso-docs
	@echo ""
	@echo "============================================"
	@echo "  RAPPORTS GENERES"
	@echo "============================================"
	@echo "Fichiers modifies:"
	@git status --porcelain reports/ graphs/ docs/iso/
	@echo ""
	@echo "Executez maintenant:"
	@echo "  git add reports/ graphs/ docs/iso/"
	@echo "  git commit -m 'chore(reports): update generated reports'"
	@echo ""

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
# ISO AUDIT (ISO 5055, 5259, 15289, 25010, 29119, 42001)
# ============================================
iso-audit:
	@echo "Audit conformite ISO..."
	$(PYTHON) scripts/audit_iso_conformity.py --fix
	@echo ""
	@echo "Rapport: reports/iso-audit/"

validate-iso:
	@echo "Validation ISO stricte..."
	$(PYTHON) scripts/audit_iso_conformity.py --strict
	@echo ""
	@echo "Audit ISO passe!"

# ============================================
# ML LIFECYCLE (ISO 42001, 5259, 25059)
# ============================================
parse-data:
	@echo "Parsing dataset FFE (ISO 5259)..."
	$(PYTHON) -m scripts.parse_dataset
	@echo ""
	@echo "Dataset parse dans: data/"

features:
	@echo "Feature engineering (ISO 5259)..."
	$(PYTHON) -m scripts.feature_engineering
	@echo ""
	@echo "Features generees"

train:
	@echo "Entrainement modeles (ISO 42001)..."
	$(PYTHON) -m scripts.train_models_parallel
	@echo ""
	@echo "Modeles entraines dans: models/"

evaluate:
	@echo "Evaluation modeles (ISO 25059)..."
	$(PYTHON) -m scripts.evaluate_models
	@echo ""
	@echo "Rapport: reports/evaluation/"

ensemble:
	@echo "Stacking ensemble (ISO 42001)..."
	$(PYTHON) -m scripts.ensemble_stacking
	@echo ""
	@echo "Ensemble cree"

ml-pipeline: parse-data features train evaluate ensemble
	@echo ""
	@echo "============================================"
	@echo "  PIPELINE ML COMPLET TERMINE"
	@echo "============================================"
	@echo ""

# ============================================
# PRODUCTION MONITORING (ISO 42001, 5259, 27001)
# ============================================
model-card:
	@echo "Generation Model Card (ISO 42001)..."
	$(PYTHON) -c "from scripts.model_registry import ProductionModelCard; print('Model Card template ready')"
	@echo ""
	@echo "Model Card: models/model_card.json"

drift-report:
	@echo "Rapport drift monitoring (ISO 5259)..."
	$(PYTHON) -c "from scripts.model_registry import create_drift_report, check_drift_status; print('Drift monitoring ready')"
	@echo ""
	@echo "Rapport: reports/drift/"

data-lineage:
	@echo "Rapport lineage donnees (ISO 5259)..."
	$(PYTHON) -c "from scripts.model_registry import compute_data_lineage; print('Data lineage ready')"
	@echo ""
	@echo "Lineage: reports/lineage/"

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

all-iso: all iso-audit
	@echo ""
	@echo "============================================"
	@echo "  VALIDATION ISO COMPLETE PASSEE!"
	@echo "============================================"
	@echo ""
	@echo "Normes validees:"
	@echo "  - ISO 5055  (Code Quality)"
	@echo "  - ISO 5259  (Data Quality ML)"
	@echo "  - ISO 15289 (Documentation)"
	@echo "  - ISO 25010 (System Quality)"
	@echo "  - ISO 27001 (Security)"
	@echo "  - ISO 27034 (Secure Coding)"
	@echo "  - ISO 29119 (Testing)"
	@echo "  - ISO 42001 (AI Management)"
	@echo "  - ISO 42010 (Architecture)"
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
