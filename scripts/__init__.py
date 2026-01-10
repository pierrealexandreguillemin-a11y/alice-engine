"""Module: scripts/__init__.py - Scripts Package ALICE.

Document ID: ALICE-MOD-SCRIPTS-001
Version: 0.5.0

Scripts utilitaires et pipelines ML pour le moteur ALICE.

Packages:
- training/: Pipeline d'entraînement ML (CatBoost, XGBoost, LightGBM)
- features/: Feature engineering (ALI, CE)
- fairness/: Détection de biais (ISO 24027)
- robustness/: Tests de robustesse (ISO 24029)
- model_registry/: Gestion modèles production
- evaluation/: Métriques et benchmarks

Scripts principaux:
- train_models_parallel.py: Entraînement parallèle multi-modèles
- feature_engineering.py: Extraction features ML
- ensemble_stacking.py: Stacking ensemble
- evaluate_models.py: Benchmarks et évaluation
- parse_dataset/: Parsing données FFE

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management System (pipelines ML)
- ISO/IEC 5259:2024 - Data Quality for ML (lineage)
- ISO/IEC 5055:2021 - Code Quality (modules <300 lignes)
- ISO/IEC TR 24027:2021 - Bias in AI (fairness/)
- ISO/IEC 24029:2021 - Neural Network Robustness (robustness/)
- ISO/IEC 25059:2023 - AI Quality Model

Author: ALICE Engine Team
Last Updated: 2026-01-10
"""
