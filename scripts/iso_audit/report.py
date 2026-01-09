"""Génération de rapports d'audit ISO - ISO 5055.

Fonctions pour générer les rapports JSON et Markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.iso_audit.types import AuditResult, ISOViolation


def generate_report(result: AuditResult, output_path: Path) -> None:
    """Génère rapport JSON et Markdown."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Sous-dossier iso-audit
    iso_output = output_path / "iso-audit"
    iso_output.mkdir(parents=True, exist_ok=True)

    _generate_json_report(result, iso_output)
    _generate_markdown_report(result, iso_output)

    print("\n[REPORTS] Rapports generés:")
    print(f"  - {iso_output / 'iso-audit-report.json'}")
    print(f"  - {iso_output / 'ISO_AUDIT_REPORT.md'}")


def _generate_json_report(result: AuditResult, output_path: Path) -> None:
    """Génère le rapport JSON."""
    json_path = output_path / "iso-audit-report.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def _generate_markdown_report(result: AuditResult, output_path: Path) -> None:
    """Génère le rapport Markdown."""
    md_path = output_path / "ISO_AUDIT_REPORT.md"
    md_content = _build_markdown_content(result)
    md_path.write_text(md_content, encoding="utf-8")


def _build_markdown_content(result: AuditResult) -> str:
    """Construit le contenu Markdown du rapport."""
    status = "[OK] CONFORME" if result.compliant else "[FAIL] NON CONFORME"

    md = f"""# Rapport Audit ISO - ALICE Engine

> Généré le {result.timestamp}
> Statut: {status}

## Résumé

| Métrique | Valeur |
|----------|--------|
| Violations totales | {len(result.violations)} |
| Critiques | {sum(1 for v in result.violations if v.severity == "CRITICAL")} |
| Hautes | {sum(1 for v in result.violations if v.severity == "HIGH")} |
| Moyennes | {sum(1 for v in result.violations if v.severity == "MEDIUM")} |
| Basses | {sum(1 for v in result.violations if v.severity == "LOW")} |

## Détail par Norme

"""
    md += _build_violations_by_norm(result.violations)
    return md


def _build_violations_by_norm(violations: list[ISOViolation]) -> str:
    """Construit la section des violations par norme."""
    by_norm: dict[str, list[ISOViolation]] = {}
    for v in violations:
        by_norm.setdefault(v.norm, []).append(v)

    content = ""
    for norm, norm_violations in sorted(by_norm.items()):
        content += f"### {norm}\n\n"
        content += "| Sévérité | Fichier | Message | Correction |\n"
        content += "|----------|---------|---------|------------|\n"
        for v in sorted(norm_violations, key=lambda x: x.severity):
            content += f"| {v.severity} | `{v.file}` | {v.message} | {v.fix_suggestion} |\n"
        content += "\n"

    return content
