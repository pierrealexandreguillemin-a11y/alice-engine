"""Module definitions for ISO conformity checks.

Each module defines:
- name: Tool name
- category: Functional category
- iso: Applicable ISO standards
- check: Command to verify installation
"""

# Modules et leur conformite ISO
# Note: Use "python -m" prefix for cross-platform compatibility (Windows/Linux)
MODULES = [
    # === QUALITE CODE ===
    {
        "name": "ruff",
        "category": "Qualite Code",
        "iso": "ISO 25010, ISO 5055",
        "check": "python -m ruff --version",
    },
    {
        "name": "mypy",
        "category": "Type Safety",
        "iso": "ISO 25010, ISO 5055",
        "check": "python -m mypy --version",
    },
    {
        "name": "black",
        "category": "Formatage",
        "iso": "ISO 25010",
        "check": "python -m black --version",
    },
    {
        "name": "isort",
        "category": "Formatage",
        "iso": "ISO 25010",
        "check": "python -m isort --version",
    },
    # === SECURITE ===
    {
        "name": "bandit",
        "category": "Securite",
        "iso": "ISO 27001, ISO 27034, OWASP",
        "check": "python -m bandit --version",
    },
    {
        "name": "safety",
        "category": "Securite",
        "iso": "ISO 27001",
        "check": "python -m safety --version",
    },
    {
        "name": "pip-audit",
        "category": "Securite",
        "iso": "ISO 27001",
        "check": "python -m pip_audit --version",
    },
    {
        "name": "gitleaks",
        "category": "Secrets",
        "iso": "ISO 27001 (P0-CRITICAL)",
        "check": "gitleaks version",
    },
    # === TESTS ===
    {
        "name": "pytest",
        "category": "Tests",
        "iso": "ISO 29119",
        "check": "python -m pytest --version",
    },
    {
        "name": "pytest-cov",
        "category": "Coverage",
        "iso": "ISO 29119",
        "check": 'python -c "import pytest_cov"',
    },
    {
        "name": "pytest-asyncio",
        "category": "Tests Async",
        "iso": "ISO 29119",
        "check": 'python -c "import pytest_asyncio"',
    },
    # === COMPLEXITE ===
    {
        "name": "radon",
        "category": "Complexite",
        "iso": "ISO 25010, ISO 5055",
        "check": "python -m radon --version",
    },
    {
        "name": "xenon",
        "category": "Complexite",
        "iso": "ISO 25010",
        "check": 'python -c "import xenon"',
    },
    # === ARCHITECTURE ===
    {
        "name": "pydeps",
        "category": "Architecture",
        "iso": "ISO 42010",
        "check": 'python -c "import pydeps"',
    },
    {
        "name": "import-linter",
        "category": "Architecture",
        "iso": "ISO 42010",
        "check": 'python -c "import importlinter"',
    },
    # === HOOKS ===
    {
        "name": "pre-commit",
        "category": "Git Hooks",
        "iso": "ISO 12207",
        "check": "python -m pre_commit --version",
    },
    {
        "name": "commitizen",
        "category": "Commits",
        "iso": "ISO 12207",
        "check": "python -m commitizen version",
    },
    # === DOCUMENTATION ===
    {
        "name": "mkdocs",
        "category": "Documentation",
        "iso": "ISO 26514",
        "check": "python -m mkdocs --version",
    },
    {
        "name": "pdoc",
        "category": "API Docs",
        "iso": "ISO 26514",
        "check": 'python -c "import pdoc"',
    },
]
