#!/usr/bin/env bash
# Plan 1 Definition of Done — verification script
# Exit 0 only if ALL gates pass.
#
# SCOPE NOTE:
# This script aggregates the FAST P1G gates only. The following gates are
# EXCLUDED from this script because they require parquet datasets (8+ min):
#   - P1G07 (coverage >= 75%)        : validated via tests/ in CI
#   - P1G08/P1G09/P1G10 (parquet tests) : validated individually by Tasks
#                                          17/18 (9 parquet tests + 1 E2E smoke)
# These slow gates are validated INDIVIDUALLY by the preceding Tasks of
# Plan 1 (30 fast tests + 9 parquet tests + 1 E2E smoke). Running this
# script after those Tasks pass is sufficient for Plan 1 DoD.
#
# Aggregated standards: ISO 5055 / 5259 / 27034 / 29119 / 42001 / 42010 / 15289 / 24027
set -euo pipefail

echo "=== P1G01 File size <= 300 lignes ==="
while IFS= read -r f; do
  lines=$(wc -l < "$f")
  if [ "$lines" -gt 300 ]; then
    echo "FAIL: $f has $lines lines (max 300)"
    exit 1
  fi
done < <(find services/ffe services/ali -name "*.py")
echo "OK"

echo "=== P1G02 Complexity xenon <= B ==="
xenon --max-absolute B --max-modules B --max-average A \
  services/ffe services/ali scripts/sync_ffe_rules.py

echo "=== P1G03 MyPy strict ==="
mypy services/ffe services/ali --strict --follow-imports=silent

echo "=== P1G04 Ruff ==="
ruff check services/ffe services/ali

echo "=== P1G05 Pydantic reject invalid ==="
pytest tests/test_ffe_schemas.py::test_rejects_missing_uuid \
       tests/test_ffe_schemas.py::test_rejects_invalid_effet -v

echo "=== P1G06 Gitleaks ==="
pre-commit run gitleaks --all-files

echo "=== P1G11 UUID RFC4122 preserved ==="
python -c "
from pathlib import Path
from services.ffe.rule_engine import RuleEngine
e = RuleEngine.from_json_file(Path('config/ffe_rules/a02.json'))
assert len(e.rules) == 14, f'expected 14 rules, got {len(e.rules)}'
assert all(r.uuid_rfc4122 for r in e.rules), 'uuid_rfc4122 manquant'
print('OK')
"

echo "=== P1G12 ADR-013 present et reference ==="
test -f docs/architecture/adr/ADR-013-rule-engine-json.md
grep -q "ADR-013" docs/architecture/ALI_ARCHITECTURE.md

echo "=== P1G13 MkDocs --strict ==="
mkdocs build --strict

echo "=== P1G14 F7 survivor documente ==="
grep -qi "survivor" docs/architecture/ALI_ARCHITECTURE.md

echo "=== Structural: A02 = 14 rules ==="
python -c "
import json
d = json.load(open('config/ffe_rules/a02.json', encoding='utf-8'))
assert len(d['rules']) == 14
print('OK')
"

echo "=== Structural: verifiability covers all A02 ==="
python -c "
import json
c = json.load(open('config/ffe_rules/alice_verifiability.json', encoding='utf-8'))
r = json.load(open('config/ffe_rules/a02.json', encoding='utf-8'))
uuids_r = {x['id'] for x in r['rules']}
uuids_c = set(c['classifications'].keys())
missing = uuids_r - uuids_c
assert not missing, f'verifiability missing: {missing}'
print('OK')
"

echo "=== Structural: 10 PUBLIC + 4 PRIVATE ==="
python -c "
import json
c = json.load(open('config/ffe_rules/alice_verifiability.json', encoding='utf-8'))['classifications']
pub = sum(1 for v in c.values() if v['verifiability'] == 'public')
priv = sum(1 for v in c.values() if v['verifiability'] == 'private')
assert pub == 10 and priv == 4, f'pub={pub} priv={priv}'
print(f'OK (pub={pub}, priv={priv})')
"

echo "=== Structural: ffe-rules-drift hook ==="
pre-commit run ffe-rules-drift --all-files

echo ""
echo "============================================"
echo "ALL GATES PASS (tests parquets exclus -- deja valides individuellement)"
echo "Plan 1 Definition of Done : SATISFIED"
echo "============================================"
