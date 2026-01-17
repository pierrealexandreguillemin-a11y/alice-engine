#!/usr/bin/env python3
"""PreToolUse hook - BLOQUE si conditions non remplies."""
import json
import sys
from pathlib import Path


def check_data_exists(cwd: Path) -> list[str]:
    """Vérifie données ISO 5259."""
    errors = []
    for name, expected in [('train', 1139819), ('valid', 70647), ('test', 197843)]:
        p = cwd / 'data' / 'features' / f'{name}.parquet'
        if not p.exists():
            errors.append(f'MISSING: {p}')
    return errors


def check_ram(min_gb: float = 6.0) -> list[str]:
    """Vérifie RAM disponible."""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        if avail < min_gb:
            return [f'RAM insuffisante: {avail:.1f}GB < {min_gb}GB requis']
    except ImportError:
        pass  # psutil non installé, skip
    return []


def check_dependencies() -> list[str]:
    """Vérifie dépendances AutoGluon."""
    errors = []
    try:
        import fastai
    except ImportError:
        errors.append('MISSING: fastai (pip install "autogluon.tabular[all]")')
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        errors.append('MISSING: autogluon.tabular')
    return errors


def check_autogluon_model(cwd: Path) -> list[str]:
    """Vérifie modèle AutoGluon existe."""
    model_path = cwd / 'models' / 'autogluon' / 'autogluon_extreme_v2'
    if not model_path.exists():
        return [f'MISSING: {model_path} - Exécuter Phase 3 d\'abord']
    return []


def check_baseline_models(cwd: Path) -> list[str]:
    """Vérifie modèles baseline existent."""
    meta = sorted((cwd / 'models').glob('v*/metadata.json'))
    if not meta:
        return ['MISSING: baseline models - Exécuter Phase 2 d\'abord']
    return []


def check_reports_exist(cwd: Path, reports: list[str]) -> list[str]:
    """Vérifie rapports requis existent."""
    errors = []
    for report in reports:
        p = cwd / 'reports' / report
        if not p.exists():
            errors.append(f'MISSING: {p}')
    return errors


def main():
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    tool_input = data.get('tool_input', {})

    if tool_name != 'Bash':
        sys.exit(0)  # Pas un Bash, autoriser

    command = tool_input.get('command', '')
    cwd = Path(data.get('cwd', '.'))
    errors = []

    # === PHASE 2: Avant baseline training ===
    if 'train_models_parallel' in command:
        print('[PRE-CHECK] Phase 2: Baseline training', file=sys.stderr)
        errors.extend(check_data_exists(cwd))
        errors.extend(check_ram(6.0))

    # === PHASE 3: Avant AutoGluon ===
    elif 'TabularPredictor' in command and 'fit' in command:
        print('[PRE-CHECK] Phase 3: AutoGluon training', file=sys.stderr)
        errors.extend(check_dependencies())
        errors.extend(check_data_exists(cwd))
        errors.extend(check_ram(6.0))

    # === PHASE 4.1: Avant ISO 24029 ===
    elif 'iso_robustness' in command or 'validate_robustness' in command:
        print('[PRE-CHECK] Phase 4.1: ISO 24029', file=sys.stderr)
        errors.extend(check_autogluon_model(cwd))
        errors.extend(check_data_exists(cwd))

    # === PHASE 4.2: Avant ISO 24027 ===
    elif 'iso_fairness' in command or 'validate_fairness' in command:
        print('[PRE-CHECK] Phase 4.2: ISO 24027', file=sys.stderr)
        errors.extend(check_autogluon_model(cwd))

    # === PHASE 4.4: Avant McNemar ===
    elif 'mcnemar' in command.lower() or 'chi2' in command:
        print('[PRE-CHECK] Phase 4.4: McNemar', file=sys.stderr)
        errors.extend(check_autogluon_model(cwd))
        errors.extend(check_baseline_models(cwd))

    # === PHASE 5: Avant rapport final ===
    elif 'ISO_25059_TRAINING_REPORT' in command:
        print('[PRE-CHECK] Phase 5: Final report', file=sys.stderr)
        errors.extend(check_reports_exist(cwd, [
            'autogluon_results.json',
            'robustness_report.json',
            'fairness_report.json',
            'mcnemar_comparison.json'
        ]))

    if errors:
        print('BLOQUE - Conditions non remplies:', file=sys.stderr)
        for e in errors:
            print(f'  * {e}', file=sys.stderr)
        sys.exit(2)  # BLOQUE L'EXÉCUTION

    sys.exit(0)  # Autorise


if __name__ == '__main__':
    main()
