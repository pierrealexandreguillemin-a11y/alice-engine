"""Tasks audit ISO - ISO 5055/5259/15289/25010/29119/42001."""

from invoke import Context, task


@task(name="iso-audit")
def iso_audit(c: Context, quick: bool = False) -> None:
    """Run automated ISO conformity audit."""
    print("Audit conformite ISO...")
    cmd = "python -m scripts.audit_iso_conformity --fix"
    if quick:
        cmd += " --quick"
    c.run(cmd)
    print("\nRapport: reports/iso-audit/")


@task(name="iso-audit-quick")
def iso_audit_quick(c: Context) -> None:
    """Run quick ISO audit (skip pytest)."""
    print("Audit conformite ISO (mode rapide)...")
    c.run("python -m scripts.audit_iso_conformity --fix --quick")
    print("\nRapport: reports/iso-audit/")


@task(name="validate-iso")
def validate_iso(c: Context, quick: bool = False) -> None:
    """Run strict ISO audit (fail if non-compliant)."""
    print("Validation ISO stricte...")
    cmd = "python -m scripts.audit_iso_conformity --strict"
    if quick:
        cmd += " --quick"
    c.run(cmd)
    print("\nAudit ISO passe!")
