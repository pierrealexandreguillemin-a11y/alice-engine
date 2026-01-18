"""Package Serving - MLflow PyFunc pour déploiement Render.

Ce package contient les wrappers MLflow PyFunc pour servir les modèles
ALICE via MLflow Model Serving ou Render.

ISO Compliance:
- ISO/IEC 42001:2023 - AI Management (Model Serving)
- ISO/IEC 5259:2024 - Data Quality (Input Validation)
- ISO/IEC 27034 - Secure Coding

Modules:
- pyfunc_wrapper.py: Wrapper PyFunc générique
- autogluon_pyfunc.py: Wrapper spécifique AutoGluon
- baseline_pyfunc.py: Wrapper pour modèles baseline

Author: ALICE Engine Team
Version: 1.0.0
"""

from scripts.serving.pyfunc_wrapper import AliceModelWrapper

__all__ = ["AliceModelWrapper"]
