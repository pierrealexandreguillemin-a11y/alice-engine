"""Package Agents - Architecture Multi-Agent inspirée AG-A/MLZero.

Ce package implémente une architecture multi-agent pour l'automatisation
du pipeline ML ALICE, inspirée d'AutoGluon-Assistant (MLZero).

Modules:
- semantic_memory.py: Base de connaissance ISO et ML
- episodic_memory.py: Historique des exécutions MLflow
- iterative_refinement.py: Corrections automatiques

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Automation)
- ISO/IEC TR 24027:2021 - Bias in AI (Fairness rules)
- ISO/IEC 24029:2021 - Robustness (Validation rules)

Architecture Reference:
- MLZero: A Multi-Agent System for End-to-end ML Automation (NeurIPS 2025)
- AutoGluon-Assistant: https://github.com/autogluon/autogluon-assistant

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.agents.iterative_refinement import IterativeRefinement
from scripts.agents.semantic_memory import ISOSemanticMemory

__all__ = ["ISOSemanticMemory", "IterativeRefinement"]
