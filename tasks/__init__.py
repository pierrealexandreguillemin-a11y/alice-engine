"""Tasks - Cross-platform task runner (invoke).

Equivalent du Makefile, compatible Windows/Linux/Mac.
Version: 3.0 - FULL ISO COVERAGE + ML LIFECYCLE + MONITORING

Usage:
    inv --list          # Liste toutes les taches
    inv quality         # Lint + Format + Typecheck + Security
    inv all-iso         # Validation ISO complete

ISO Compliance:
- ISO/IEC 5055 - Code Quality (SRP: modules < 300 lignes)
- ISO/IEC 42001 - AI Management System
- ISO/IEC 5259 - Data Quality for ML

Author: ALICE Engine Team
Last Updated: 2026-01-09
"""

from invoke import Collection

from tasks.architecture import architecture, graphs, iso_docs, reports
from tasks.dev import clean, dev, hooks, install
from tasks.iso_audit import iso_audit, iso_audit_quick, validate_iso
from tasks.ml_lifecycle import (
    ensemble,
    evaluate,
    features,
    ml_pipeline,
    parse_data,
    train,
)
from tasks.monitoring import data_lineage, drift_report, model_card
from tasks.quality import format, lint, quality, security, typecheck
from tasks.testing import all_checks, all_iso, audit, complexity, test, test_cov, validate

# Build namespace
ns = Collection()

# Setup
ns.add_task(install)
ns.add_task(hooks)

# Quality
ns.add_task(lint)
ns.add_task(format)
ns.add_task(typecheck)
ns.add_task(security)
ns.add_task(quality)

# Testing
ns.add_task(test)
ns.add_task(test_cov, name="test-cov")
ns.add_task(audit)
ns.add_task(complexity)
ns.add_task(validate)

# Architecture
ns.add_task(graphs)
ns.add_task(architecture)
ns.add_task(iso_docs, name="iso-docs")
ns.add_task(reports)

# ISO Audit
ns.add_task(iso_audit, name="iso-audit")
ns.add_task(iso_audit_quick, name="iso-audit-quick")
ns.add_task(validate_iso, name="validate-iso")

# ML Lifecycle
ns.add_task(parse_data, name="parse-data")
ns.add_task(features)
ns.add_task(train)
ns.add_task(evaluate)
ns.add_task(ensemble)
ns.add_task(ml_pipeline, name="ml-pipeline")

# Monitoring
ns.add_task(model_card, name="model-card")
ns.add_task(drift_report, name="drift-report")
ns.add_task(data_lineage, name="data-lineage")

# All-in-one
ns.add_task(all_checks, name="all")
ns.add_task(all_iso, name="all-iso")

# Dev
ns.add_task(dev)
ns.add_task(clean)
