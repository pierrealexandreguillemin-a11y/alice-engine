"""Package: scripts/reports - ISO Report Generation.

Document ID: ALICE-MOD-REPORTS-PKG-001
Version: 1.0.0

Package pour la génération de rapports ISO.

Modules:
- generate_iso25059: Générateur rapport ISO 25059

ISO Compliance:
- ISO/IEC 25059:2023 - AI Quality Model
- ISO/IEC 5055:2021 - Code Quality

Author: ALICE Engine Team
"""

from scripts.reports.generate_iso25059 import main as generate_iso25059_report

__all__ = ["generate_iso25059_report"]
