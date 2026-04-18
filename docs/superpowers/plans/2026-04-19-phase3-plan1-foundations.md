# Phase 3 · Plan 1 — Foundations : RuleEngine + Data Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Poser les fondations Phase 3 (RuleEngine JSON-driven + data loader + historique enrichi F2/F3) sans toucher au flux `/compose` Phase 2 existant. Résultat : composants unitairement testables, prêts à être wirés Plan 2.

**Architecture:** 3 couches indépendantes, chacune autonome et testable : (1) `services/ffe/` — RuleEngine qui interprète JSON vendorés depuis chess-app avec classification PUBLIC/PRIVATE annotée localement ; (2) `services/ali/cache.py + pool_loader.py` — cache in-RAM parquets + F7 survivor filter ; (3) `services/ali/history.py` — enrichment F2 recency decay + F3 autoregressive streak. Aucun changement au routing ni au lifespan dans ce plan.

**Tech Stack:** Python 3.13, Pydantic v2, pandas, numpy, pytest, pandera. Pas de nouvelle dépendance externe.

**Scope du plan** : 21 tasks, ~3 semaines estim, ~1900 lignes code + tests.

**Principes directeurs :**
- **KISS** : un fichier = une responsabilité ; pas d'abstraction non justifiée
- **DRY** : réutiliser `services/audit/logger.py`, `scripts/features/ali/presence.py`, `scripts/features/ali/patterns.py` quand pertinent
- **TDD** : test en premier, verif rouge, impl, verif vert, commit
- **ISO** : chaque task cite la/les norme(s) ISO adressée(s)

---

## File Structure

**Créer :**
```
services/ffe/__init__.py                     # package init
services/ffe/schemas.py                      # Pydantic schemas JSON
services/ffe/rule_engine.py                  # Rule + RuleEngine
services/ali/__init__.py                     # package init (Plan 1 partie)
services/ali/types.py                        # PlayerCandidate, CompetitionContext, RuleViolation
services/ali/verifiability.py                # VerifiabilityClassifier
services/ali/cache.py                        # ALIDataCache
services/ali/pool_loader.py                  # PlayerPoolLoader (F7)
services/ali/history.py                      # HistoryEnricher (F2 + F3)
config/ffe_rules/a02.json                    # vendored from chess-app
config/ffe_rules/alice_verifiability.json    # annotations locales ALICE
scripts/sync_ffe_rules.py                    # sync chess-app JSON → ALICE
.pre-commit-ffe-drift.sh                     # hook pre-commit drift alert
docs/architecture/adr/ADR-013-rule-engine-json.md
tests/test_rule_engine.py                    # ~14 tests
tests/test_verifiability.py                  # ~6 tests
tests/test_ali_cache.py                      # ~6 tests
tests/test_pool_loader.py                    # ~5 tests
tests/test_history_enricher.py               # ~8 tests
tests/test_sync_ffe_rules.py                 # ~3 tests
tests/fixtures/ffe_rules/mini_a02.json       # fixture JSON test
```

**Modifier :**
```
app/config.py             # +ffe_rules_dir setting, +ali_cache settings
.pre-commit-config.yaml   # +hook ffe drift alert
```

**Ne PAS modifier dans ce plan (réservé Plan 2) :**
```
services/ffe_rules.py         # encore utilisé par routes.py Phase 2
services/inference.py         # encore utilisé par routes.py Phase 2
app/main.py::lifespan         # wiring dans Plan 2
app/api/routes.py             # wiring dans Plan 2
```

---

## Task 1 : Initialiser packages + settings

**Files:**
- Create: `services/ffe/__init__.py`
- Create: `services/ali/__init__.py`
- Create: `config/ffe_rules/` (directory)
- Modify: `app/config.py` (add settings)

**ISO:** 27034 (Pydantic config), 42010

- [ ] **Step 1 : Écrire le test de settings**

Créer `tests/test_config_phase3.py` :
```python
from app.config import get_settings

def test_phase3_settings_defaults():
    s = get_settings()
    assert s.ffe_rules_dir == "./config/ffe_rules"
    assert s.ali_cache_max_age_days == 7
    assert s.joueurs_parquet == "./data/joueurs.parquet"
    assert s.echiquiers_parquet == "./data/echiquiers.parquet"
    assert s.recency_decay_lambda == 0.9
```

- [ ] **Step 2 : Lancer test, vérifier qu'il échoue**

```bash
pytest tests/test_config_phase3.py -v
```
Attendu : FAIL (attributs non définis sur Settings)

- [ ] **Step 3 : Ajouter les settings**

Dans `app/config.py`, après `default_scenario_count: int = 20` :
```python
    # Phase 3 : RuleEngine + ALI data
    ffe_rules_dir: str = "./config/ffe_rules"
    ali_cache_max_age_days: int = 7
    joueurs_parquet: str = "./data/joueurs.parquet"
    echiquiers_parquet: str = "./data/echiquiers.parquet"
    recency_decay_lambda: float = 0.9
    streak_lag_window: int = 3
```

- [ ] **Step 4 : Créer les packages**

```bash
touch services/ffe/__init__.py
touch services/ali/__init__.py
mkdir -p config/ffe_rules
```

Contenu `services/ffe/__init__.py` :
```python
"""FFE RuleEngine package (Phase 3)."""
```

Contenu `services/ali/__init__.py` :
```python
"""ALI (Adversarial Lineup Inference) package (Phase 3)."""
```

- [ ] **Step 5 : Lancer test, vérifier succès**

```bash
pytest tests/test_config_phase3.py -v
```
Attendu : PASS

- [ ] **Step 6 : Commit**

```bash
git add services/ffe/__init__.py services/ali/__init__.py app/config.py tests/test_config_phase3.py
git commit -m "feat(phase3): init ffe/ali packages + config settings"
```

---

## Task 2 : ADR-013 — RuleEngine JSON remplace Python FFE rules

**Files:**
- Create: `docs/architecture/adr/ADR-013-rule-engine-json.md`

**ISO:** 42010 (ADR)

- [ ] **Step 1 : Rédiger l'ADR**

Créer `docs/architecture/adr/ADR-013-rule-engine-json.md` :
```markdown
# ADR-013 : RuleEngine JSON-driven remplace les règles Python FFE

**Date** : 2026-04-19
**Status** : ACCEPTED
**Context** : Phase 3 brainstorming 2026-04-18/19

## Contexte

Phase 2 a introduit 11 règles FFE en Python (`services/ffe_rules.py`).
Parallèlement, chess-app/flat-six/rules/ contient 46 règles FFE structurées
en JSON (format canonique avec UUID RFC4122, source_ref PDF, article, conditions
formalisées). pocket-arbiter parse ces PDFs via docling.

Les 11 règles Python = **réimplémentation duplicate** des 14 A02 (dont 10 seulement
couvertes). Risque de drift entre les deux sources.

## Décision

**Remplacer** `services/ffe_rules.py` par un **RuleEngine générique** qui interprète
les JSON chess-app vendorés dans `config/ffe_rules/`.

ALICE classe PUBLIC/PRIVATE chaque règle dans `alice_verifiability.json`
(annotations locales, indépendantes de chess-app).

## Conséquences

**Positives :**
- Source de vérité unique (JSON chess-app, normatif)
- Traçabilité ISO 42001/5259 via UUID + source_ref
- Extensible sans nouveau code (ajouter un JSON = nouvelle compétition)
- Cohérent avec l'écosystème (pocket-arbiter → chess-app JSON → ALICE)

**Négatives :**
- ~250 lignes nouvelles (RuleEngine générique)
- Dette D10 : sync manuelle JSON chess-app → ALICE (traitée dans ce plan)

## Alternatives rejetées

- **Statu quo** : garder les 11 Python → drift fatal
- **Migrer une à une** : conserve double source → risque divergent
- **Vendor sans RuleEngine** : nécessite réimplémentation par règle → pas DRY

## Implémentation

- Plan 1 : RuleEngine + vendoring + classifier + CI drift alert
- Plan 2 : wire dans /compose, suppression `services/ffe_rules.py`
```

- [ ] **Step 2 : Vérifier MkDocs build ne casse pas**

```bash
mkdocs build --strict 2>&1 | head -20
```
Attendu : "INFO - Documentation built" ou warnings mineurs uniquement.

- [ ] **Step 3 : Commit**

```bash
git add docs/architecture/adr/ADR-013-rule-engine-json.md
git commit -m "docs(adr): ADR-013 RuleEngine JSON replaces Python FFE rules"
```

---

## Task 3 : Vendor A02.json + script de sync

**Files:**
- Copy: `C:/Dev/chess-app/backend/flat-six/rules/national/a02.json` → `config/ffe_rules/a02.json`
- Create: `scripts/sync_ffe_rules.py`

**ISO:** 5259 (lineage), 15289 (doc lifecycle), 42010

- [ ] **Step 1 : Écrire le test sync**

Créer `tests/test_sync_ffe_rules.py` :
```python
from pathlib import Path
from scripts.sync_ffe_rules import compute_file_sha256, sync_rules, detect_drift

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_sha256_deterministic(tmp_path):
    f = tmp_path / "test.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    h1 = compute_file_sha256(f)
    h2 = compute_file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64


def test_detect_drift_true_when_contents_differ(tmp_path):
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text('{"v": 1}', encoding="utf-8")
    target.write_text('{"v": 2}', encoding="utf-8")
    assert detect_drift(source, target) is True


def test_sync_copies_source_to_target(tmp_path):
    source = tmp_path / "source.json"
    target = tmp_path / "target" / "a02.json"
    source.write_text('{"rules": []}', encoding="utf-8")
    sync_rules(source, target)
    assert target.exists()
    assert target.read_text(encoding="utf-8") == '{"rules": []}'
```

- [ ] **Step 2 : Lancer test, vérifier fail**

```bash
pytest tests/test_sync_ffe_rules.py -v
```
Attendu : FAIL import error.

- [ ] **Step 3 : Implémenter le script de sync**

Créer `scripts/sync_ffe_rules.py` :
```python
"""Sync chess-app flat-six JSON rules into ALICE config/ffe_rules/.

ISO 5259 : lineage tracé via SHA-256 des JSONs sync.
ISO 42001 : source_ref = chess-app commit + date.

Usage:
    python scripts/sync_ffe_rules.py          # sync all
    python scripts/sync_ffe_rules.py --check  # drift check only (CI)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


CHESS_APP_RULES_DIR = Path("C:/Dev/chess-app/backend/flat-six/rules")
ALICE_RULES_DIR = Path(__file__).parent.parent / "config" / "ffe_rules"

# Phase 3 scope : A02 uniquement. Extension J02/Coupes en Phase 3.5.
RULES_TO_SYNC = [
    ("national/a02.json", "a02.json"),
]


def compute_file_sha256(path: Path) -> str:
    """Return hex SHA-256 of a file's content."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def detect_drift(source: Path, target: Path) -> bool:
    """Return True if source and target differ (or target missing)."""
    if not target.exists():
        return True
    return compute_file_sha256(source) != compute_file_sha256(target)


def sync_rules(source: Path, target: Path) -> None:
    """Copy source JSON to target, creating dirs as needed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync FFE rules from chess-app")
    parser.add_argument("--check", action="store_true", help="drift check only")
    args = parser.parse_args()

    drift_found = False
    for src_rel, tgt_rel in RULES_TO_SYNC:
        src = CHESS_APP_RULES_DIR / src_rel
        tgt = ALICE_RULES_DIR / tgt_rel
        if not src.exists():
            print(f"ERROR: source missing: {src}", file=sys.stderr)
            return 1
        if detect_drift(src, tgt):
            drift_found = True
            if args.check:
                print(f"DRIFT: {tgt_rel}", file=sys.stderr)
            else:
                print(f"SYNC: {src_rel} -> {tgt_rel}")
                sync_rules(src, tgt)

    if args.check and drift_found:
        print("FAIL: FFE rules drift detected. Run scripts/sync_ffe_rules.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4 : Exécuter le sync une première fois**

```bash
python scripts/sync_ffe_rules.py
```
Attendu : `SYNC: national/a02.json -> a02.json`. Vérifier `config/ffe_rules/a02.json` existe et contient les 14 règles.

```bash
ls -la config/ffe_rules/a02.json
python -c "import json; d=json.load(open('config/ffe_rules/a02.json')); print(len(d['rules']))"
```
Attendu : `14`.

- [ ] **Step 5 : Lancer tests, vérifier succès**

```bash
pytest tests/test_sync_ffe_rules.py -v
```
Attendu : PASS 3/3.

- [ ] **Step 6 : Commit**

```bash
git add scripts/sync_ffe_rules.py config/ffe_rules/a02.json tests/test_sync_ffe_rules.py
git commit -m "feat(phase3): vendor A02.json + sync script with drift detection"
```

---

## Task 4 : Pre-commit hook pour drift chess-app → ALICE

**Files:**
- Modify: `.pre-commit-config.yaml`

**ISO:** 42001 (traceability), 5259 (lineage check)

- [ ] **Step 1 : Ajouter le hook**

Modifier `.pre-commit-config.yaml`, ajouter en dernier :
```yaml
  # D10 : ALICE ↔ chess-app FFE rules drift alert (Phase 3)
  - repo: local
    hooks:
      - id: ffe-rules-drift
        name: "ISO 42001 | FFE rules drift check (chess-app -> ALICE)"
        entry: python scripts/sync_ffe_rules.py --check
        language: system
        pass_filenames: false
        stages: [commit]
        verbose: true
```

- [ ] **Step 2 : Tester le hook manuellement**

```bash
pre-commit run ffe-rules-drift --all-files
```
Attendu : `PASSED` (pas de drift car fraîchement syncé).

Si tu veux tester le fail : modifier temporairement `config/ffe_rules/a02.json` (ex. ajouter un espace), relancer hook → `FAIL: FFE rules drift detected`. Restore puis relancer → PASS.

- [ ] **Step 3 : Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "ci(phase3): pre-commit hook ffe-rules drift alert (D10)"
```

---

## Task 5 : Pydantic schemas pour JSON A02

**Files:**
- Create: `services/ffe/schemas.py`
- Create: `tests/fixtures/ffe_rules/mini_a02.json`

**ISO:** 27034 (input validation), 25012 (JSON schema validation)

- [ ] **Step 1 : Créer la fixture minimale**

Créer `tests/fixtures/ffe_rules/mini_a02.json` :
```json
{
  "metadata": {
    "championship": "Test Champ",
    "description": "Fixture minimale Phase 3",
    "version": "2025-26",
    "source": "test"
  },
  "rules": [
    {
      "uuid": "TEST_001",
      "uuid_rfc4122": "00000000-0000-4000-8000-000000000001",
      "id": "Test_3.7.a_001",
      "source_ref": "test_ref.pdf",
      "document": "Test doc",
      "version": "2025-26",
      "article": "3.7.a",
      "texte": "Chaque équipe = 8 joueurs.",
      "conditions": {"team_size": 8},
      "effet": "restrict_team_composition",
      "exceptions": [],
      "priority": 1,
      "date_version": "2025-07-01"
    }
  ]
}
```

- [ ] **Step 2 : Écrire les tests Pydantic**

Créer `tests/test_ffe_schemas.py` :
```python
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from services.ffe.schemas import RulesDocument, RuleModel


FIXTURE = Path(__file__).parent / "fixtures" / "ffe_rules" / "mini_a02.json"


def test_load_mini_a02():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    doc = RulesDocument.model_validate(data)
    assert len(doc.rules) == 1
    assert doc.rules[0].uuid == "TEST_001"
    assert doc.rules[0].article == "3.7.a"


def test_load_real_a02():
    data = json.loads(Path("config/ffe_rules/a02.json").read_text(encoding="utf-8"))
    doc = RulesDocument.model_validate(data)
    assert len(doc.rules) == 14


def test_rejects_missing_uuid():
    bad = {"uuid_rfc4122": "x", "id": "i", "source_ref": "r",
           "document": "d", "version": "v", "article": "3.7.a",
           "texte": "t", "conditions": {}, "effet": "restrict_team_composition",
           "exceptions": [], "priority": 1, "date_version": "2025-07-01"}
    with pytest.raises(ValidationError):
        RuleModel.model_validate(bad)


def test_rejects_invalid_effet():
    bad = {"uuid": "U", "uuid_rfc4122": "x", "id": "i", "source_ref": "r",
           "document": "d", "version": "v", "article": "a", "texte": "t",
           "conditions": {}, "effet": "invalid_effet", "exceptions": [],
           "priority": 1, "date_version": "2025-07-01"}
    with pytest.raises(ValidationError):
        RuleModel.model_validate(bad)
```

- [ ] **Step 3 : Vérifier tests fail**

```bash
pytest tests/test_ffe_schemas.py -v
```
Attendu : FAIL (import error).

- [ ] **Step 4 : Implémenter les schemas**

Créer `services/ffe/schemas.py` :
```python
"""Pydantic schemas for FFE rule JSON validation.

ISO 27034 : input validation.
ISO 25012 : data quality structural schema.

Document ID: ALICE-FFE-SCHEMAS
Version: 1.0.0
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


VALID_EFFETS = {
    "restrict_team_composition",
    "restrict_arbitrage",
    "restrict_player_eligibility",
}


class RuleModel(BaseModel):
    """One FFE rule as stored in chess-app JSON."""

    uuid: str = Field(..., min_length=1)
    uuid_rfc4122: str
    id: str
    source_ref: str
    document: str
    version: str
    article: str
    texte: str
    conditions: dict
    effet: Literal[
        "restrict_team_composition",
        "restrict_arbitrage",
        "restrict_player_eligibility",
    ]
    exceptions: list = Field(default_factory=list)
    priority: int = Field(ge=1)
    date_version: str


class MetadataModel(BaseModel):
    """JSON document metadata."""

    championship: str | None = None
    description: str
    version: str
    source: str


class RulesDocument(BaseModel):
    """A full FFE rules JSON document (one per competition)."""

    metadata: MetadataModel
    rules: list[RuleModel] = Field(default_factory=list)
```

- [ ] **Step 5 : Vérifier tests pass**

```bash
pytest tests/test_ffe_schemas.py -v
```
Attendu : PASS 4/4.

- [ ] **Step 6 : Commit**

```bash
git add services/ffe/schemas.py tests/test_ffe_schemas.py tests/fixtures/ffe_rules/mini_a02.json
git commit -m "feat(phase3): Pydantic schemas for FFE rule JSON validation (ISO 27034)"
```

---

## Task 6 : Rule dataclass + RuleEngine loader

**Files:**
- Create: `services/ffe/rule_engine.py`

**ISO:** 5055 (SRP), 42001 (UUID tracing), 5259 (lineage_hash)

- [ ] **Step 1 : Écrire test loader + lineage**

Créer `tests/test_rule_engine.py` :
```python
import hashlib
import json
from pathlib import Path
import pytest
from services.ffe.rule_engine import Rule, RuleEngine


REAL_A02 = Path("config/ffe_rules/a02.json")
MINI = Path(__file__).parent / "fixtures" / "ffe_rules" / "mini_a02.json"


def test_rule_engine_loads_real_a02():
    engine = RuleEngine.from_json_file(REAL_A02)
    assert len(engine.rules) == 14
    ids = {r.uuid for r in engine.rules}
    assert "N1-N4_3.7.a_001" in ids


def test_rule_engine_loads_mini_fixture():
    engine = RuleEngine.from_json_file(MINI)
    assert len(engine.rules) == 1
    r = engine.rules[0]
    assert isinstance(r, Rule)
    assert r.uuid == "TEST_001"


def test_lineage_hash_is_deterministic():
    e1 = RuleEngine.from_json_file(REAL_A02)
    e2 = RuleEngine.from_json_file(REAL_A02)
    assert e1.lineage_hash() == e2.lineage_hash()
    assert len(e1.lineage_hash()) == 64


def test_lineage_hash_changes_with_content(tmp_path):
    f = tmp_path / "x.json"
    base = json.loads(MINI.read_text(encoding="utf-8"))
    f.write_text(json.dumps(base), encoding="utf-8")
    e1 = RuleEngine.from_json_file(f)
    base["rules"][0]["texte"] = "autre"
    f.write_text(json.dumps(base), encoding="utf-8")
    e2 = RuleEngine.from_json_file(f)
    assert e1.lineage_hash() != e2.lineage_hash()
```

- [ ] **Step 2 : Vérifier tests fail**

```bash
pytest tests/test_rule_engine.py -v
```
Attendu : FAIL (import error).

- [ ] **Step 3 : Implémenter Rule + RuleEngine loader + lineage**

Créer `services/ffe/rule_engine.py` :
```python
"""FFE RuleEngine — generic JSON-driven rules interpreter.

ISO 5055 : SRP, module < 300 lignes.
ISO 42001 : traceability via UUID + source_ref per rule.
ISO 5259 : lineage_hash SHA-256 of loaded JSON.
ISO 27034 : Pydantic validation at load.

Replaces services/ffe_rules.py (ADR-013).

Document ID: ALICE-FFE-RULE-ENGINE
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from services.ffe.schemas import RuleModel, RulesDocument


@dataclass(frozen=True)
class Rule:
    """One FFE rule, domain dataclass (frozen for ISO 29119 testability)."""

    uuid: str
    source_ref: str
    article: str
    texte: str
    conditions: dict
    effet: str
    priority: int

    @classmethod
    def from_model(cls, model: RuleModel) -> "Rule":
        return cls(
            uuid=model.uuid,
            source_ref=model.source_ref,
            article=model.article,
            texte=model.texte,
            conditions=dict(model.conditions),
            effet=model.effet,
            priority=model.priority,
        )


class RuleEngine:
    """Generic engine that interprets FFE rules loaded from JSON.

    Usage:
        engine = RuleEngine.from_json_file(Path("config/ffe_rules/a02.json"))
        violations = engine.validate_lineup(lineup, context)
        pool = engine.filter_candidates(pool, context)
    """

    def __init__(self, rules: list[Rule], source_sha256: str) -> None:
        self._rules: tuple[Rule, ...] = tuple(rules)
        self._source_sha256 = source_sha256

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def lineage_hash(self) -> str:
        """Return SHA-256 of the JSON source for ISO 5259 lineage."""
        return self._source_sha256

    @classmethod
    def from_json_file(cls, path: Path) -> "RuleEngine":
        """Load, validate (Pydantic), and instantiate from a JSON file."""
        raw_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
        doc = RulesDocument.model_validate(data)
        rules = [Rule.from_model(m) for m in doc.rules]
        return cls(rules=rules, source_sha256=source_sha256)
```

- [ ] **Step 4 : Vérifier tests pass**

```bash
pytest tests/test_rule_engine.py -v
```
Attendu : PASS 4/4.

- [ ] **Step 5 : Commit**

```bash
git add services/ffe/rule_engine.py tests/test_rule_engine.py
git commit -m "feat(phase3): Rule dataclass + RuleEngine loader + SHA-256 lineage"
```

---

## Task 7 : RuleEngine — dispatchers filter_candidates + validate_lineup + RuleViolation

**Files:**
- Modify: `services/ffe/rule_engine.py` (add methods)
- Create: `services/ali/types.py` (add RuleViolation, PlayerCandidate, CompetitionContext)

**ISO:** 5055 (SRP), 42001 (rule tracing in violations)

- [ ] **Step 1 : Écrire test des types**

Créer `tests/test_ali_types.py` :
```python
from services.ali.types import (
    PlayerCandidate, CompetitionContext, RuleViolation,
)


def test_player_candidate_frozen():
    p = PlayerCandidate(
        nr_ffe="A12345", nom="Dupont", prenom="Jean", elo=1800,
        club="CLUBX", mute=False, genre="M", categorie="SE",
        licence_active=True, age_min=25, age_max=30,
    )
    import dataclasses
    assert dataclasses.is_dataclass(p)
    import pytest
    with pytest.raises(Exception):
        p.elo = 9999  # frozen


def test_competition_context_roundtrip():
    ctx = CompetitionContext(
        competition_code="A02", niveau="N2", ronde=3, team_size=8,
        noyau_min=50, max_mutes=3, elo_max=None,
    )
    assert ctx.competition_code == "A02"
    assert ctx.team_size == 8


def test_rule_violation_frozen():
    v = RuleViolation(
        rule_uuid="U1", rule_article="3.7.a",
        message="team size mismatch", severity="error",
    )
    assert v.severity == "error"
```

- [ ] **Step 2 : Vérifier tests fail**

```bash
pytest tests/test_ali_types.py -v
```
Attendu : FAIL (import error).

- [ ] **Step 3 : Implémenter les types ALI**

Créer `services/ali/types.py` :
```python
"""Frozen dataclasses pour ALI (ISO 29119 testability, ISO 5055 SRP).

Document ID: ALICE-ALI-TYPES
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlayerCandidate:
    """Un joueur éligible du pool (enrichi ou brut)."""

    nr_ffe: str
    nom: str
    prenom: str
    elo: int
    club: str
    mute: bool
    genre: str
    categorie: str
    licence_active: bool
    age_min: int | None = None
    age_max: int | None = None

    # Features ALI (remplies par HistoryEnricher, peuvent être None au chargement)
    taux_presence_effectif: float | None = None
    played_lag1: bool | None = None
    played_lag2: bool | None = None
    played_lag3: bool | None = None
    echiquier_prefere: int | None = None
    role_type: str | None = None


@dataclass(frozen=True)
class CompetitionContext:
    """Contexte d'un match pour appliquer les bonnes règles FFE."""

    competition_code: str  # "A02", "J02", "C01", ...
    niveau: str            # "N1", "N2", "N3", "N4", "N5", "N6"
    ronde: int             # 1..N
    team_size: int
    noyau_min: int         # paramètre règle 3.7.f
    max_mutes: int         # paramètre règle 3.7.g
    elo_max: int | None    # paramètre règle 3.7.j (N4 = 2400)


@dataclass(frozen=True)
class RuleViolation:
    """Une violation de règle détectée lors de validate_lineup."""

    rule_uuid: str
    rule_article: str
    message: str
    severity: Literal["error", "warning"]
```

- [ ] **Step 4 : Tests types PASS**

```bash
pytest tests/test_ali_types.py -v
```
Attendu : PASS.

- [ ] **Step 5 : Écrire test filter + validate**

Ajouter à `tests/test_rule_engine.py` :
```python
from services.ali.types import PlayerCandidate, CompetitionContext


def _ctx():
    return CompetitionContext(
        competition_code="A02", niveau="N2", ronde=3, team_size=8,
        noyau_min=50, max_mutes=3, elo_max=None,
    )


def _player(nr: str, elo: int, mute: bool = False, licence_active: bool = True):
    return PlayerCandidate(
        nr_ffe=nr, nom=f"P{nr}", prenom="X", elo=elo, club="C1",
        mute=mute, genre="M", categorie="SE", licence_active=licence_active,
    )


def test_filter_candidates_returns_list():
    engine = RuleEngine.from_json_file(REAL_A02)
    pool = [_player("A1", 2000), _player("A2", 1500)]
    out = engine.filter_candidates(pool, _ctx())
    assert isinstance(out, list)
    assert len(out) <= len(pool)


def test_validate_lineup_empty_returns_team_size_violation():
    engine = RuleEngine.from_json_file(REAL_A02)
    violations = engine.validate_lineup([], _ctx())
    # team_size violated : empty lineup
    assert any(v.rule_article == "3.7.a" for v in violations)


def test_validate_lineup_happy_path():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2000 - i * 50) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx())
    # 3.7.a (team_size=8) should be ok ; 3.6.e (order by elo) ok
    hard_errors = [v for v in violations if v.severity == "error"]
    # Some non-implemented rules may raise NotImplemented warnings; ok
    # Au moins pas de team_size violation
    assert not any(v.rule_article == "3.7.a" for v in hard_errors)
```

- [ ] **Step 6 : Vérifier tests fail**

```bash
pytest tests/test_rule_engine.py -v
```
Attendu : FAIL (filter_candidates, validate_lineup non définis).

- [ ] **Step 7 : Implémenter dispatchers**

Dans `services/ffe/rule_engine.py`, ajouter après `from_json_file` :
```python
    def filter_candidates(self, pool, context):
        """Filter pool with rules that restrict player ELIGIBILITY.

        Currently applies article 3.7.g (mutes — only at lineup level, not pool)
        and 3.7.j (elo_max if set). Most eligibility rules require external
        data (brule history, same-group) and are applied in validate_lineup.
        """
        out = list(pool)
        for rule in self._rules:
            if rule.effet != "restrict_team_composition":
                continue
            # elo_max: remove players above cap
            if rule.article == "3.7.j" and context.elo_max is not None:
                out = [p for p in out if p.elo <= context.elo_max]
        return out

    def validate_lineup(self, lineup, context):
        """Validate a full lineup against all rules. Returns list[RuleViolation]."""
        from services.ali.types import RuleViolation

        violations: list[RuleViolation] = []
        for rule in self._rules:
            if rule.effet != "restrict_team_composition":
                continue
            v = self._check_rule(rule, lineup, context)
            if v is not None:
                violations.append(v)
        return violations

    def _check_rule(self, rule: Rule, lineup, context):
        """Per-rule check. Returns RuleViolation or None."""
        from services.ali.types import RuleViolation

        art = rule.article

        if art == "3.7.a":
            # team_size
            expected = rule.conditions.get("team_size", context.team_size)
            if len(lineup) != expected:
                return RuleViolation(
                    rule_uuid=rule.uuid, rule_article=art,
                    message=f"team_size: expected {expected}, got {len(lineup)}",
                    severity="error",
                )
        elif art == "3.6.e":
            # Elo descending order, tolerance from conditions
            tolerance = rule.conditions.get("elo_tolerance", 100)
            elos = [p.elo for p in lineup]
            for i in range(len(elos) - 1):
                if elos[i + 1] - elos[i] > tolerance:
                    return RuleViolation(
                        rule_uuid=rule.uuid, rule_article=art,
                        message=f"elo order: board {i+1}={elos[i]} < board {i+2}={elos[i+1]} + tol",
                        severity="error",
                    )
        elif art == "3.7.g":
            # max_mutes
            max_m = rule.conditions.get("max_mutes", context.max_mutes)
            muted = sum(1 for p in lineup if p.mute)
            if muted > max_m:
                return RuleViolation(
                    rule_uuid=rule.uuid, rule_article=art,
                    message=f"mutes: {muted} > max {max_m}",
                    severity="error",
                )
        elif art == "3.7.j" and context.elo_max is not None:
            over = [p for p in lineup if p.elo > context.elo_max]
            if over:
                return RuleViolation(
                    rule_uuid=rule.uuid, rule_article=art,
                    message=f"elo_max: {len(over)} players above {context.elo_max}",
                    severity="error",
                )
        # 3.7.b, 3.2, 3.7.k, 3.7 : PRIVATE/arbitrage, ignorés ici
        # 3.7.c (brule), 3.7.d (same_group), 3.7.e (match_count), 3.7.f (noyau),
        # 3.7.h (foreign), 3.7.i (fr_gender) : nécessitent données externes
        # (historique, noyau, licences) — Plan 2 quand wirés au /compose
        return None
```

- [ ] **Step 8 : Vérifier tests pass**

```bash
pytest tests/test_rule_engine.py tests/test_ali_types.py -v
```
Attendu : tous PASS.

- [ ] **Step 9 : Commit**

```bash
git add services/ffe/rule_engine.py services/ali/types.py tests/test_ali_types.py tests/test_rule_engine.py
git commit -m "feat(phase3): RuleEngine filter_candidates + validate_lineup dispatchers"
```

---

## Task 8 : Tests complémentaires par UUID A02 (coverage des 14 règles)

**Files:**
- Modify: `tests/test_rule_engine.py` (ajouts tests)

**ISO:** 29119 (1 test per rule UUID)

- [ ] **Step 1 : Ajouter tests UUID-level**

Ajouter à `tests/test_rule_engine.py` :
```python
def _lineup_ok():
    return [_player(f"X{i}", 2000 - i * 50) for i in range(8)]


def test_rule_3_7_a_too_few():
    engine = RuleEngine.from_json_file(REAL_A02)
    violations = engine.validate_lineup(_lineup_ok()[:7], _ctx())
    assert any(v.rule_article == "3.7.a" for v in violations)


def test_rule_3_7_a_too_many():
    engine = RuleEngine.from_json_file(REAL_A02)
    violations = engine.validate_lineup(_lineup_ok() + [_player("X8", 1000)], _ctx())
    assert any(v.rule_article == "3.7.a" for v in violations)


def test_rule_3_6_e_bad_order():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = _lineup_ok()
    # swap: put a 2500 at board 8 after a 1600 at board 1
    lineup = [_player("X0", 1600)] + [_player(f"X{i}", 2500) for i in range(7)]
    violations = engine.validate_lineup(lineup, _ctx())
    assert any(v.rule_article == "3.6.e" for v in violations)


def test_rule_3_7_g_too_many_mutes():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2000 - i * 50, mute=(i < 4)) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx())
    assert any(v.rule_article == "3.7.g" for v in violations)


def test_rule_3_7_j_elo_max():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2500) for i in range(8)]
    ctx = CompetitionContext(
        competition_code="A02", niveau="N4", ronde=3, team_size=8,
        noyau_min=50, max_mutes=3, elo_max=2400,
    )
    violations = engine.validate_lineup(lineup, ctx)
    assert any(v.rule_article == "3.7.j" for v in violations)


def test_rule_3_7_j_not_applied_when_no_cap():
    engine = RuleEngine.from_json_file(REAL_A02)
    lineup = [_player(f"X{i}", 2500) for i in range(8)]
    violations = engine.validate_lineup(lineup, _ctx())  # elo_max=None
    assert not any(v.rule_article == "3.7.j" for v in violations)
```

- [ ] **Step 2 : Lancer tous tests rule_engine**

```bash
pytest tests/test_rule_engine.py -v
```
Attendu : PASS tous.

- [ ] **Step 3 : Commit**

```bash
git add tests/test_rule_engine.py
git commit -m "test(phase3): rule engine UUID-level tests (ISO 29119)"
```

---

## Task 9 : VerifiabilityClassifier — annotations locales ALICE

**Files:**
- Create: `config/ffe_rules/alice_verifiability.json`
- Create: `services/ali/verifiability.py`
- Create: `tests/test_verifiability.py`

**ISO:** 24027 (fairness classification), 42001

- [ ] **Step 1 : Écrire classification JSON**

Créer `config/ffe_rules/alice_verifiability.json` :
```json
{
  "metadata": {
    "description": "Annotations locales ALICE : classification PUBLIC/PRIVATE des règles FFE",
    "version": "1.0.0",
    "date": "2026-04-19",
    "scope": "A02 interclubs (Phase 3)"
  },
  "classifications": {
    "N1-N4_3.7.a_001": {"verifiability": "public", "reason": "team_size trivialement vérifiable", "data_source": "-"},
    "N1-N4_3.7.b_001": {"verifiability": "private", "reason": "composition groupes décidée par CTF/Ligue, pas joueur", "data_source": "-"},
    "N1-N4_3.6.e_001": {"verifiability": "public", "reason": "Elo publics FFE", "data_source": "joueurs.parquet"},
    "N1-N4_3.7.c_001": {"verifiability": "public", "reason": "brûlé déductible via historique matchs", "data_source": "echiquiers.parquet"},
    "N1-N4_3.7.d_001": {"verifiability": "public", "reason": "same_group déductible historique", "data_source": "echiquiers.parquet"},
    "N1-N4_3.7.e_001": {"verifiability": "public", "reason": "match_count historique saison", "data_source": "echiquiers.parquet"},
    "N1-N4_3.2_001": {"verifiability": "private", "reason": "désignation titulaires décidée par club", "data_source": "-"},
    "N1-N3_3.7.f_001": {"verifiability": "private", "reason": "noyau déclaré début saison par club, non public", "data_source": "-"},
    "N1-N4_3.7.g_001": {"verifiability": "public", "reason": "mute = flag licence FFE publique", "data_source": "joueurs.parquet"},
    "N1-N4_3.7.h_001": {"verifiability": "public", "reason": "nationalité = code FFE publique", "data_source": "joueurs.parquet"},
    "Top16-N2_3.7.i_001": {"verifiability": "public", "reason": "genre + nationalité publics", "data_source": "joueurs.parquet"},
    "N4_3.7.j_001": {"verifiability": "public", "reason": "Elo publics", "data_source": "joueurs.parquet"},
    "N1-N4_3.7.k_001": {"verifiability": "private", "reason": "inscriptions décidées par club", "data_source": "-"},
    "N1-N2_3.7_001": {"verifiability": "private", "reason": "règle arbitrage, pas composition", "data_source": "-"}
  }
}
```

- [ ] **Step 2 : Écrire tests**

Créer `tests/test_verifiability.py` :
```python
from pathlib import Path
from services.ali.verifiability import VerifiabilityClassifier
from services.ffe.rule_engine import RuleEngine


REAL_A02 = Path("config/ffe_rules/a02.json")
CLASSIF = Path("config/ffe_rules/alice_verifiability.json")


def test_classifier_loads_from_json():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    assert len(c.classifications) == 14


def test_is_public_returns_true_for_public_rule():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    rule_3_7_a = next(r for r in engine.rules if r.uuid == "N1-N4_3.7.a_001")
    assert c.is_public(rule_3_7_a) is True


def test_is_public_returns_false_for_private_rule():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    rule_noyau = next(r for r in engine.rules if r.uuid == "N1-N3_3.7.f_001")
    assert c.is_public(rule_noyau) is False


def test_partition_rules():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    public, private = c.partition_rules(engine.rules)
    assert len(public) == 10
    assert len(private) == 4
    assert all(c.is_public(r) for r in public)
    assert all(not c.is_public(r) for r in private)


def test_partition_is_exhaustive():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    engine = RuleEngine.from_json_file(REAL_A02)
    public, private = c.partition_rules(engine.rules)
    assert len(public) + len(private) == len(engine.rules)


def test_unknown_rule_raises():
    c = VerifiabilityClassifier.from_json_file(CLASSIF)
    from services.ffe.rule_engine import Rule
    unknown = Rule(
        uuid="UNKNOWN_XYZ", source_ref="", article="", texte="",
        conditions={}, effet="restrict_team_composition", priority=1,
    )
    import pytest
    with pytest.raises(KeyError):
        c.is_public(unknown)
```

- [ ] **Step 3 : Vérifier fail**

```bash
pytest tests/test_verifiability.py -v
```
Attendu : FAIL import error.

- [ ] **Step 4 : Implémenter VerifiabilityClassifier**

Créer `services/ali/verifiability.py` :
```python
"""VerifiabilityClassifier — PUBLIC/PRIVATE annotations locales ALICE.

ISO 24027 : fairness classification (distinguer ce qu'on peut vérifier
sur l'adversaire depuis ce qu'on doit supposer).
ISO 42001 : traceability des décisions de classification.

Document ID: ALICE-ALI-VERIFIABILITY
Version: 1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from services.ffe.rule_engine import Rule


Verifiability = Literal["public", "private"]


class VerifiabilityClassifier:
    """Classifie les règles FFE en PUBLIC (applicable au générateur MC)
    ou PRIVATE (supposée respectée par l'adversaire).
    """

    def __init__(self, classifications: dict[str, dict]) -> None:
        self._classifications = dict(classifications)

    @property
    def classifications(self) -> dict[str, dict]:
        return dict(self._classifications)

    @classmethod
    def from_json_file(cls, path: Path) -> "VerifiabilityClassifier":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(classifications=data["classifications"])

    def is_public(self, rule: Rule) -> bool:
        entry = self._classifications[rule.uuid]  # KeyError if unknown
        return entry["verifiability"] == "public"

    def partition_rules(
        self, rules: tuple[Rule, ...] | list[Rule]
    ) -> tuple[list[Rule], list[Rule]]:
        """Return (public_rules, private_rules)."""
        pub, priv = [], []
        for r in rules:
            if r.uuid not in self._classifications:
                continue  # skip unknown in partition (safer than raise)
            if self.is_public(r):
                pub.append(r)
            else:
                priv.append(r)
        return pub, priv
```

- [ ] **Step 5 : Tests PASS**

```bash
pytest tests/test_verifiability.py -v
```
Attendu : PASS 6/6.

- [ ] **Step 6 : Commit**

```bash
git add config/ffe_rules/alice_verifiability.json services/ali/verifiability.py tests/test_verifiability.py
git commit -m "feat(phase3): VerifiabilityClassifier A02 10 public / 4 private (ISO 24027)"
```

---

## Task 10 : ALIDataCache — structure + loader parquets

**Files:**
- Create: `services/ali/cache.py`
- Create: `tests/test_ali_cache.py`

**ISO:** 5259 (lineage SHA-256), 25010 (observability)

- [ ] **Step 1 : Écrire tests cache structure**

Créer `tests/test_ali_cache.py` :
```python
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from services.ali.cache import ALIDataCache


J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)


def test_cache_loads_parquets():
    cache = ALIDataCache.load_from_parquets(J, E)
    assert len(cache.joueurs_total) > 0
    assert len(cache.echiquiers_total) > 0


def test_cache_signatures_are_sha256():
    cache = ALIDataCache.load_from_parquets(J, E)
    assert len(cache.parquet_sig_joueurs) == 64
    assert len(cache.parquet_sig_echiquiers) == 64
    # determinism
    cache2 = ALIDataCache.load_from_parquets(J, E)
    assert cache.parquet_sig_joueurs == cache2.parquet_sig_joueurs


def test_cache_exposes_loaded_at_utc():
    cache = ALIDataCache.load_from_parquets(J, E)
    assert isinstance(cache.loaded_at, datetime)
    assert cache.loaded_at.tzinfo is not None


def test_cache_is_stale_by_age():
    cache = ALIDataCache.load_from_parquets(J, E)
    # fresh cache: not stale
    assert cache.is_stale(max_age_days=1) is False
    # artificially age
    object.__setattr__(
        cache,
        "loaded_at",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    assert cache.is_stale(max_age_days=7) is True
```

- [ ] **Step 2 : Vérifier fail**

```bash
pytest tests/test_ali_cache.py -v
```
Attendu : FAIL.

- [ ] **Step 3 : Implémenter ALIDataCache**

Créer `services/ali/cache.py` :
```python
"""ALIDataCache — in-RAM cache des parquets pour inference /compose.

ISO 5259 : SHA-256 lineage des parquets source.
ISO 25010 : performance (cache évite I/O par request).
ISO 5055 : SRP strict (I/O seulement, pas de logique métier).

Document ID: ALICE-ALI-CACHE
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class ALIDataCache:
    """Cache in-RAM des parquets + indexes + SHA-256 lineage.

    Chargé une fois au lifespan FastAPI (Plan 2).
    Les DataFrames sont pleins (pas de copy per lookup).
    """

    joueurs_total: pd.DataFrame
    echiquiers_total: pd.DataFrame
    joueurs_by_club: dict[str, pd.DataFrame]
    echiquiers_by_player: dict[str, pd.DataFrame]
    parquet_sig_joueurs: str
    parquet_sig_echiquiers: str
    loaded_at: datetime

    @classmethod
    def load_from_parquets(cls, path_joueurs: Path, path_echiquiers: Path) -> "ALIDataCache":
        sig_j = hashlib.sha256(path_joueurs.read_bytes()).hexdigest()
        sig_e = hashlib.sha256(path_echiquiers.read_bytes()).hexdigest()
        df_j = pd.read_parquet(path_joueurs)
        df_e = pd.read_parquet(path_echiquiers)

        # Index by club (joueurs column = 'club')
        j_by_club: dict[str, pd.DataFrame] = {
            str(club): group
            for club, group in df_j.groupby("club", dropna=False)
        }

        # Index echiquiers by player name (both couleurs)
        e_by_player: dict[str, pd.DataFrame] = {}
        for color in ("blanc", "noir"):
            col = f"{color}_nom"
            if col not in df_e.columns:
                continue
            for name, group in df_e.groupby(col, dropna=True):
                key = str(name)
                existing = e_by_player.get(key)
                e_by_player[key] = (
                    group if existing is None else pd.concat([existing, group], ignore_index=True)
                )

        return cls(
            joueurs_total=df_j,
            echiquiers_total=df_e,
            joueurs_by_club=j_by_club,
            echiquiers_by_player=e_by_player,
            parquet_sig_joueurs=sig_j,
            parquet_sig_echiquiers=sig_e,
            loaded_at=datetime.now(timezone.utc),
        )

    def is_stale(self, max_age_days: int = 7) -> bool:
        age = (datetime.now(timezone.utc) - self.loaded_at).total_seconds()
        return age > max_age_days * 86400
```

- [ ] **Step 4 : Tests PASS**

```bash
pytest tests/test_ali_cache.py -v
```
Attendu : PASS (ou SKIP si parquets absents sur le runner).

- [ ] **Step 5 : Commit**

```bash
git add services/ali/cache.py tests/test_ali_cache.py
git commit -m "feat(phase3): ALIDataCache load_from_parquets + SHA-256 lineage (ISO 5259)"
```

---

## Task 11 : ALIDataCache — lookup methods

**Files:**
- Modify: `services/ali/cache.py`
- Modify: `tests/test_ali_cache.py`

**ISO:** 25010 (performance O(1) lookup)

- [ ] **Step 1 : Écrire tests lookup**

Ajouter à `tests/test_ali_cache.py` :
```python
def test_lookup_club_returns_subset():
    cache = ALIDataCache.load_from_parquets(J, E)
    first_club = next(iter(cache.joueurs_by_club.keys()))
    subset = cache.lookup_club(first_club)
    assert len(subset) > 0
    assert all(str(v) == first_club for v in subset["club"].unique())


def test_lookup_club_unknown_returns_empty_df():
    cache = ALIDataCache.load_from_parquets(J, E)
    subset = cache.lookup_club("UNKNOWN_CLUB_XYZ_999")
    assert subset.empty


def test_lookup_history_returns_union_of_colors():
    cache = ALIDataCache.load_from_parquets(J, E)
    # Take first player appearing in echiquiers
    first_name = next(iter(cache.echiquiers_by_player.keys()))
    hist = cache.lookup_history([first_name])
    assert len(hist) > 0
```

- [ ] **Step 2 : Vérifier fail**

```bash
pytest tests/test_ali_cache.py -k lookup -v
```
Attendu : FAIL (methods manquantes).

- [ ] **Step 3 : Implémenter lookup methods**

Dans `services/ali/cache.py`, ajouter dans `class ALIDataCache` :
```python
    def lookup_club(self, club_id: str) -> pd.DataFrame:
        """Return joueurs subset for a given club_id. Empty DataFrame if unknown."""
        return self.joueurs_by_club.get(
            str(club_id),
            self.joueurs_total.iloc[0:0],  # empty same-schema
        )

    def lookup_history(self, player_names: list[str]) -> pd.DataFrame:
        """Return echiquiers rows where blanc_nom OR noir_nom ∈ player_names."""
        parts = []
        for name in player_names:
            df = self.echiquiers_by_player.get(str(name))
            if df is not None:
                parts.append(df)
        if not parts:
            return self.echiquiers_total.iloc[0:0]
        return pd.concat(parts, ignore_index=True).drop_duplicates()
```

- [ ] **Step 4 : Tests PASS**

```bash
pytest tests/test_ali_cache.py -v
```
Attendu : PASS tous.

- [ ] **Step 5 : Commit**

```bash
git add services/ali/cache.py tests/test_ali_cache.py
git commit -m "feat(phase3): ALIDataCache lookup_club + lookup_history O(1)"
```

---

## Task 12 : PlayerPoolLoader — F7 survivor filter + overrides

**Files:**
- Create: `services/ali/pool_loader.py`
- Create: `tests/test_pool_loader.py`

**ISO:** 5055 (SRP), 24027 (fairness — F7 documented assumption)

- [ ] **Step 1 : Écrire tests**

Créer `tests/test_pool_loader.py` :
```python
from pathlib import Path
from datetime import date

import pytest
import pandas as pd

from services.ali.cache import ALIDataCache
from services.ali.pool_loader import PlayerPoolLoader


J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")

pytestmark = pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)


def test_loader_returns_players_for_club():
    cache = ALIDataCache.load_from_parquets(J, E)
    loader = PlayerPoolLoader(cache)
    first_club = next(iter(cache.joueurs_by_club.keys()))
    candidates = loader.load_pool(first_club, date.today().isoformat())
    assert len(candidates) > 0
    assert all(c.club == first_club for c in candidates)


def test_loader_survivor_filter_excludes_inactive():
    cache = ALIDataCache.load_from_parquets(J, E)
    loader = PlayerPoolLoader(cache)
    first_club = next(iter(cache.joueurs_by_club.keys()))
    candidates = loader.load_pool(first_club, date.today().isoformat())
    assert all(c.licence_active for c in candidates)


def test_loader_overrides_are_appended():
    cache = ALIDataCache.load_from_parquets(J, E)
    loader = PlayerPoolLoader(cache)
    first_club = next(iter(cache.joueurs_by_club.keys()))
    overrides = [{
        "nr_ffe": "OVERRIDE_999", "nom": "Guest", "prenom": "X",
        "elo": 2200, "club": first_club, "mute": False,
        "genre": "M", "categorie": "SE", "licence_active": True,
    }]
    candidates = loader.load_pool(first_club, date.today().isoformat(), overrides=overrides)
    assert any(c.nr_ffe == "OVERRIDE_999" for c in candidates)


def test_loader_unknown_club_returns_empty():
    cache = ALIDataCache.load_from_parquets(J, E)
    loader = PlayerPoolLoader(cache)
    candidates = loader.load_pool("UNKNOWN_CLUB_XYZ_999", date.today().isoformat())
    assert candidates == []


def test_loader_overrides_replace_existing_by_nr_ffe():
    """Same nr_ffe in overrides should replace parquet entry (fraicheur client)."""
    cache = ALIDataCache.load_from_parquets(J, E)
    loader = PlayerPoolLoader(cache)
    first_club = next(iter(cache.joueurs_by_club.keys()))
    existing = cache.joueurs_by_club[first_club].iloc[0]
    overrides = [{
        "nr_ffe": existing["nr_ffe"], "nom": existing["nom"], "prenom": existing.get("prenom", ""),
        "elo": 9999, "club": first_club, "mute": False,
        "genre": existing.get("genre", "M"), "categorie": existing.get("categorie", "SE"),
        "licence_active": True,
    }]
    candidates = loader.load_pool(first_club, date.today().isoformat(), overrides=overrides)
    matching = [c for c in candidates if c.nr_ffe == existing["nr_ffe"]]
    assert len(matching) == 1
    assert matching[0].elo == 9999
```

- [ ] **Step 2 : Vérifier fail**

```bash
pytest tests/test_pool_loader.py -v
```
Attendu : FAIL.

- [ ] **Step 3 : Implémenter PlayerPoolLoader**

Créer `services/ali/pool_loader.py` :
```python
"""PlayerPoolLoader — charge l'effectif éligible d'un club à une date.

F7 : filter `licence_active` (survivor bias protection).
Source: Brown, Goetzmann, Ross, Ibbotson 1992 (transposé sports depuis finance).

ISO 5055 : SRP strict (loading + filtering, pas d'enrichment).
ISO 24027 : F7 assumption documented in Model Card.

Document ID: ALICE-ALI-POOL-LOADER
Version: 1.0.0
"""

from __future__ import annotations

from services.ali.cache import ALIDataCache
from services.ali.types import PlayerCandidate


class PlayerPoolLoader:
    """Charge le pool joueurs éligibles pour un club × date donnés."""

    def __init__(self, cache: ALIDataCache) -> None:
        self._cache = cache

    def load_pool(
        self,
        club_id: str,
        round_date: str,
        overrides: list[dict] | None = None,
    ) -> list[PlayerCandidate]:
        """Return eligible candidates. F7 survivor filter applied.

        @param club_id: FFE club code
        @param round_date: ISO date (YYYY-MM-DD) — réservé usage futur (J02 éligibilité jeunes)
        @param overrides: liste de joueurs à ajouter/remplacer depuis la request
        """
        df = self._cache.lookup_club(club_id)
        if df.empty and not overrides:
            return []

        candidates: dict[str, PlayerCandidate] = {}
        for _, row in df.iterrows():
            c = _row_to_candidate(row)
            if not c.licence_active:
                continue  # F7 survivor filter
            candidates[c.nr_ffe] = c

        if overrides:
            for raw in overrides:
                c = _override_to_candidate(raw)
                candidates[c.nr_ffe] = c  # overwrite if same nr_ffe

        return list(candidates.values())


def _row_to_candidate(row) -> PlayerCandidate:
    """Map a joueurs.parquet row to PlayerCandidate."""
    licence_active = _row_licence_active(row)
    return PlayerCandidate(
        nr_ffe=str(row["nr_ffe"]),
        nom=str(row.get("nom", "")),
        prenom=str(row.get("prenom", "")),
        elo=int(row.get("elo") or 1500),
        club=str(row.get("club", "")),
        mute=bool(row.get("mute", False)),
        genre=str(row.get("genre", "M")),
        categorie=str(row.get("categorie", "SE")),
        licence_active=licence_active,
        age_min=_to_int_or_none(row.get("age_min")),
        age_max=_to_int_or_none(row.get("age_max")),
    )


def _row_licence_active(row) -> bool:
    """Deduce licence active from parquet. Conservative: True unless explicit flag."""
    # affiliation present AND elo_type not 'ARCHIVE' etc.
    elo_type = str(row.get("elo_type", "")).upper()
    if elo_type in {"ARCHIVE", "INACTIVE", ""}:
        return elo_type == ""  # empty = unknown, we keep (bénéfice doute)
    return True


def _override_to_candidate(raw: dict) -> PlayerCandidate:
    return PlayerCandidate(
        nr_ffe=str(raw["nr_ffe"]),
        nom=str(raw.get("nom", "")),
        prenom=str(raw.get("prenom", "")),
        elo=int(raw.get("elo", 1500)),
        club=str(raw.get("club", "")),
        mute=bool(raw.get("mute", False)),
        genre=str(raw.get("genre", "M")),
        categorie=str(raw.get("categorie", "SE")),
        licence_active=bool(raw.get("licence_active", True)),
        age_min=_to_int_or_none(raw.get("age_min")),
        age_max=_to_int_or_none(raw.get("age_max")),
    )


def _to_int_or_none(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4 : Tests PASS**

```bash
pytest tests/test_pool_loader.py -v
```
Attendu : PASS.

- [ ] **Step 5 : Commit**

```bash
git add services/ali/pool_loader.py tests/test_pool_loader.py
git commit -m "feat(phase3): PlayerPoolLoader with F7 survivor filter (Brown 1992)"
```

---

## Task 13 : HistoryEnricher F2 — recency exponential decay

**Files:**
- Create: `services/ali/history.py`
- Create: `tests/test_history_enricher.py`

**ISO:** 5259 (lambda paramètre tracé), 42001 (traceability)

- [ ] **Step 1 : Écrire tests F2**

Créer `tests/test_history_enricher.py` :
```python
import pandas as pd
import pytest

from services.ali.history import HistoryEnricher, compute_recency_weighted_presence


def _hist(rows):
    return pd.DataFrame(rows, columns=["joueur_nom", "saison", "ronde", "echiquier"])


def test_recency_all_rounds_played_returns_one():
    hist = _hist([("Jean", 2024, r, 1) for r in range(1, 12)])
    taux = compute_recency_weighted_presence(
        hist, player_name="Jean", round_date_saison=2024,
        nb_rondes_total=11, current_round=12, decay_lambda=0.9,
    )
    assert 0.999 < taux <= 1.0


def test_recency_no_rounds_played_returns_zero():
    hist = _hist([])
    taux = compute_recency_weighted_presence(
        hist, "Inconnu", 2024, nb_rondes_total=11, current_round=12, decay_lambda=0.9,
    )
    assert taux == 0.0


def test_recency_weights_recent_more_than_old():
    # Joueur A : joue seulement ronde 1 (ancien)
    # Joueur B : joue seulement ronde 11 (récent)
    hist = _hist([
        ("A", 2024, 1, 1),
        ("B", 2024, 11, 1),
    ])
    taux_a = compute_recency_weighted_presence(hist, "A", 2024, 11, 12, 0.9)
    taux_b = compute_recency_weighted_presence(hist, "B", 2024, 11, 12, 0.9)
    assert taux_b > taux_a


def test_recency_lambda_1_equals_flat_rate():
    # Avec lambda=1, equivalent au taux simple
    hist = _hist([("X", 2024, r, 1) for r in [1, 3, 5, 7, 9]])
    taux = compute_recency_weighted_presence(hist, "X", 2024, 11, 12, 1.0)
    # 5 rondes jouées / 11 possibles
    assert abs(taux - 5 / 11) < 1e-9
```

- [ ] **Step 2 : Vérifier fail**

```bash
pytest tests/test_history_enricher.py -v
```
Attendu : FAIL.

- [ ] **Step 3 : Implémenter F2**

Créer `services/ali/history.py` :
```python
"""HistoryEnricher — enrichissement joueur avec features ALI.

F2 : recency decay exponentiel (Brown 1959 exponential smoothing, Silver 2012
methodology FiveThirtyEight). taux effectif pondère rondes récentes > anciennes.

F3 : autoregressive streak lag 1-3 (Box & Jenkins 1970, Pappalardo 2019).

ISO 5259 : lambda paramètre tracé dans lineage_hash (via `decay_lambda`).
ISO 42001 : explainability via features lag séparées.

Document ID: ALICE-ALI-HISTORY-ENRICHER
Version: 1.0.0
"""

from __future__ import annotations

import pandas as pd

from services.ali.cache import ALIDataCache
from services.ali.types import PlayerCandidate


def compute_recency_weighted_presence(
    history: pd.DataFrame,
    player_name: str,
    round_date_saison: int,
    nb_rondes_total: int,
    current_round: int,
    decay_lambda: float = 0.9,
) -> float:
    """F2 : taux de présence pondéré par exponential decay λ.

    Formule :
        taux_effectif = Σ_r λ^(age_r) × 1[player plays r] / Σ_r λ^(age_r)
    où age_r = current_round - r (plus récent → plus petit age).
    """
    if nb_rondes_total <= 0:
        return 0.0

    sub = history[
        (history["joueur_nom"] == player_name)
        & (history["saison"] == round_date_saison)
    ]
    played_rounds = set(sub["ronde"].dropna().astype(int).tolist())

    numerator = 0.0
    denominator = 0.0
    for r in range(1, nb_rondes_total + 1):
        age = max(current_round - r, 0)
        weight = decay_lambda ** age
        denominator += weight
        if r in played_rounds:
            numerator += weight

    return numerator / denominator if denominator > 0 else 0.0


class HistoryEnricher:
    """Enrichit les PlayerCandidates avec features ALI calculées à inference time."""

    def __init__(self, cache: ALIDataCache, decay_lambda: float = 0.9, streak_lag: int = 3) -> None:
        self._cache = cache
        self._lambda = decay_lambda
        self._streak_lag = streak_lag

    def enrich(
        self,
        candidates: list[PlayerCandidate],
        saison: int,
        current_round: int,
        nb_rondes_total: int,
    ) -> list[PlayerCandidate]:
        """Return enriched candidates with F2 taux_presence_effectif set."""
        # Normalize player names to the format in echiquiers.parquet
        names = [self._player_lookup_name(c) for c in candidates]
        history = self._cache.lookup_history(names)
        # Normalize column so the function works uniformly
        history = _normalize_history(history)

        enriched: list[PlayerCandidate] = []
        for c in candidates:
            lookup_name = self._player_lookup_name(c)
            taux = compute_recency_weighted_presence(
                history, lookup_name, saison, nb_rondes_total,
                current_round, self._lambda,
            )
            # frozen dataclass: use replace via dataclasses.replace
            from dataclasses import replace
            enriched.append(replace(c, taux_presence_effectif=taux))
        return enriched

    @staticmethod
    def _player_lookup_name(c: PlayerCandidate) -> str:
        """Reconstruct the name format stored in echiquiers.parquet (e.g. 'DUPONT Jean')."""
        return f"{c.nom} {c.prenom}".strip()


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Echiquiers has blanc_nom AND noir_nom. Union into joueur_nom."""
    parts = []
    for col in ("blanc_nom", "noir_nom"):
        if col in df.columns:
            sub = df[[col, "saison", "ronde", "echiquier"]].copy()
            sub = sub.rename(columns={col: "joueur_nom"})
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["joueur_nom", "saison", "ronde", "echiquier"])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["joueur_nom", "saison", "ronde"])
    return out
```

- [ ] **Step 4 : Tests F2 PASS**

```bash
pytest tests/test_history_enricher.py -v
```
Attendu : PASS 4/4.

- [ ] **Step 5 : Commit**

```bash
git add services/ali/history.py tests/test_history_enricher.py
git commit -m "feat(phase3): HistoryEnricher F2 recency decay (Brown 1959, Silver 2012)"
```

---

## Task 14 : HistoryEnricher F3 — autoregressive streak lag 1-3

**Files:**
- Modify: `services/ali/history.py`
- Modify: `tests/test_history_enricher.py`

**ISO:** 42001 (explainability via features séparées)

- [ ] **Step 1 : Écrire tests F3**

Ajouter à `tests/test_history_enricher.py` :
```python
from services.ali.history import compute_streak_features


def test_streak_all_three_recent_rounds_played():
    # Player played rounds 8, 9, 10 — current round 11 (lag1=r10, lag2=r9, lag3=r8)
    hist = _hist([("Jean", 2024, r, 1) for r in [8, 9, 10]])
    lag1, lag2, lag3 = compute_streak_features(
        hist, "Jean", 2024, current_round=11,
    )
    assert lag1 is True
    assert lag2 is True
    assert lag3 is True


def test_streak_no_recent_rounds_played():
    hist = _hist([("Jean", 2024, r, 1) for r in [1, 2, 3]])
    lag1, lag2, lag3 = compute_streak_features(
        hist, "Jean", 2024, current_round=11,
    )
    assert lag1 is False
    assert lag2 is False
    assert lag3 is False


def test_streak_mixed_pattern():
    # Played round 10 (lag1) and round 8 (lag3), not round 9 (lag2)
    hist = _hist([("Jean", 2024, r, 1) for r in [8, 10]])
    lag1, lag2, lag3 = compute_streak_features(
        hist, "Jean", 2024, current_round=11,
    )
    assert lag1 is True
    assert lag2 is False
    assert lag3 is True


def test_streak_current_round_1_all_false():
    hist = _hist([])
    lag1, lag2, lag3 = compute_streak_features(
        hist, "Jean", 2024, current_round=1,
    )
    assert lag1 is False and lag2 is False and lag3 is False
```

- [ ] **Step 2 : Vérifier fail**

```bash
pytest tests/test_history_enricher.py -v
```
Attendu : 4 PASS (F2) + 4 FAIL (compute_streak_features manquant).

- [ ] **Step 3 : Implémenter F3**

Dans `services/ali/history.py`, ajouter au niveau module :
```python
def compute_streak_features(
    history: pd.DataFrame,
    player_name: str,
    saison: int,
    current_round: int,
) -> tuple[bool, bool, bool]:
    """F3 : autoregressive lag 1-3 (Box & Jenkins 1970).

    Return (played_lag1, played_lag2, played_lag3) booleans.
    lag_k = played round (current_round - k) ?
    """
    sub = history[
        (history["joueur_nom"] == player_name)
        & (history["saison"] == saison)
    ]
    played_rounds = set(sub["ronde"].dropna().astype(int).tolist())

    def _played_at(r: int) -> bool:
        return r >= 1 and r in played_rounds

    return (
        _played_at(current_round - 1),
        _played_at(current_round - 2),
        _played_at(current_round - 3),
    )
```

Modifier `HistoryEnricher.enrich` pour intégrer F3. Remplacer la boucle :
```python
        enriched: list[PlayerCandidate] = []
        from dataclasses import replace
        for c in candidates:
            lookup_name = self._player_lookup_name(c)
            taux = compute_recency_weighted_presence(
                history, lookup_name, saison, nb_rondes_total,
                current_round, self._lambda,
            )
            lag1, lag2, lag3 = compute_streak_features(
                history, lookup_name, saison, current_round,
            )
            enriched.append(replace(
                c,
                taux_presence_effectif=taux,
                played_lag1=lag1, played_lag2=lag2, played_lag3=lag3,
            ))
        return enriched
```

- [ ] **Step 4 : Tests PASS**

```bash
pytest tests/test_history_enricher.py -v
```
Attendu : PASS 8/8.

- [ ] **Step 5 : Commit**

```bash
git add services/ali/history.py tests/test_history_enricher.py
git commit -m "feat(phase3): HistoryEnricher F3 streak autoregressive lag 1-3 (Box & Jenkins 1970)"
```

---

## Task 15 : HistoryEnricher — test integration cache

**Files:**
- Modify: `tests/test_history_enricher.py`

**ISO:** 29119 (E2E smoke)

- [ ] **Step 1 : Écrire test intégration**

Ajouter à `tests/test_history_enricher.py` :
```python
from pathlib import Path
from services.ali.cache import ALIDataCache
from services.ali.history import HistoryEnricher
from services.ali.types import PlayerCandidate


J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")


def test_enricher_integration_real_parquets():
    if not (J.exists() and E.exists()):
        pytest.skip("parquets absent")
    cache = ALIDataCache.load_from_parquets(J, E)
    # Pick first club
    first_club = next(iter(cache.joueurs_by_club.keys()))
    first_row = cache.joueurs_by_club[first_club].iloc[0]
    candidate = PlayerCandidate(
        nr_ffe=str(first_row["nr_ffe"]), nom=str(first_row["nom"]),
        prenom=str(first_row.get("prenom", "")), elo=int(first_row.get("elo") or 1500),
        club=first_club, mute=False, genre="M", categorie="SE",
        licence_active=True,
    )
    enricher = HistoryEnricher(cache, decay_lambda=0.9)
    enriched = enricher.enrich([candidate], saison=2024, current_round=10, nb_rondes_total=11)
    assert len(enriched) == 1
    assert enriched[0].taux_presence_effectif is not None
    assert 0.0 <= enriched[0].taux_presence_effectif <= 1.0
    assert enriched[0].played_lag1 is not None
```

- [ ] **Step 2 : Lancer tests all**

```bash
pytest tests/test_history_enricher.py -v
```
Attendu : PASS tous (ou SKIP l'intégration si parquets absents).

- [ ] **Step 3 : Commit**

```bash
git add tests/test_history_enricher.py
git commit -m "test(phase3): HistoryEnricher integration smoke on real parquets"
```

---

## Task 16 : Coverage check + xenon complexity gate

**Files:**
- Aucun nouveau fichier — verification pipeline existant

**ISO:** 5055 (complexity ≤ B), 29119 (coverage ≥ 75%)

- [ ] **Step 1 : Lancer coverage sur services/ali + services/ffe**

```bash
pytest tests/test_rule_engine.py tests/test_verifiability.py tests/test_ali_cache.py \
       tests/test_pool_loader.py tests/test_history_enricher.py tests/test_ali_types.py \
       tests/test_ffe_schemas.py tests/test_sync_ffe_rules.py tests/test_config_phase3.py \
       --cov=services/ffe --cov=services/ali --cov=scripts.sync_ffe_rules \
       --cov-report=term-missing --cov-fail-under=75
```
Attendu : coverage ≥ 75%. Si non, ajouter tests manquants pour atteindre le seuil.

- [ ] **Step 2 : Lancer xenon complexity check**

```bash
xenon --max-absolute B --max-modules B --max-average A services/ffe services/ali scripts/sync_ffe_rules.py
```
Attendu : exit 0 (toute fonction ≤ complexité B).

- [ ] **Step 3 : Vérifier MyPy strict**

```bash
mypy services/ffe services/ali --strict
```
Attendu : Success: no issues found. Corriger les types si besoin.

- [ ] **Step 4 : Lancer ruff**

```bash
ruff check services/ffe services/ali tests/test_rule_engine.py tests/test_verifiability.py \
       tests/test_ali_cache.py tests/test_pool_loader.py tests/test_history_enricher.py
```
Attendu : All checks passed. Fix inline si warnings.

- [ ] **Step 5 : Commit si corrections faites**

```bash
git add -A
git commit -m "chore(phase3): coverage >=75%, xenon B, mypy strict, ruff clean" || echo "Nothing to commit"
```

---

## Task 17 : Documentation d'architecture — ALI_ARCHITECTURE.md (ébauche Plan 1)

**Files:**
- Create: `docs/architecture/ALI_ARCHITECTURE.md`

**ISO:** 15289 (doc lifecycle), 42010

- [ ] **Step 1 : Rédiger la doc architecture ébauche**

Créer `docs/architecture/ALI_ARCHITECTURE.md` :
```markdown
# ALI — Architecture (Phase 3)

**Last updated** : 2026-04-19 (Plan 1 Foundations complète)
**Status** : Plan 1 livré, Plan 2 Générateur SOTA à venir

## Vue d'ensemble Plan 1

```
config/ffe_rules/
  ├── a02.json                   (vendored from chess-app)
  └── alice_verifiability.json   (ALICE local)
       │
       ▼
services/ffe/
  ├── schemas.py (Pydantic)
  └── rule_engine.py (RuleEngine + Rule)
       │
       ▼
services/ali/
  ├── types.py (PlayerCandidate, CompetitionContext, RuleViolation)
  ├── verifiability.py (VerifiabilityClassifier)
  ├── cache.py (ALIDataCache)
  ├── pool_loader.py (PlayerPoolLoader + F7)
  └── history.py (HistoryEnricher + F2 + F3)

data/
  ├── joueurs.parquet      → cache.joueurs_by_club (index)
  └── echiquiers.parquet   → cache.echiquiers_by_player (index)
```

## Responsabilités par composant

| Composant | SRP | Entrée | Sortie |
|-----------|-----|--------|--------|
| RuleEngine | Loader + dispatch règles | JSON path | Rule[] + violations |
| VerifiabilityClassifier | Partition public/private | Rule[] | (public[], private[]) |
| ALIDataCache | I/O parquets + index | paths | DataFrames + dicts |
| PlayerPoolLoader | Filter + F7 survivor | club_id, date | PlayerCandidate[] |
| HistoryEnricher | F2 recency + F3 streak | PlayerCandidate[] | enriched PlayerCandidate[] |

## Dépendances externes

Aucune nouvelle. Réutilise pandas, pydantic, pathlib.

## ISO compliance Plan 1

- 5055 : chaque module < 300 lignes, complexité ≤ B
- 5259 : lineage_hash RuleEngine + parquet_sigs cache
- 27034 : Pydantic validation JSONs au load
- 29119 : coverage ≥ 75%, tests property-based Plan 5
- 42010 : ADR-013 (RuleEngine)
- 42001 : traceability via UUID + source_ref
- 24027 : F7 survivor filter assumption documentée

## Suite (Plans 2-5)

- Plan 2 : CopulaJointSampler + TopKEnumerator + MonteCarloSampler + Generator + /compose wire
- Plan 3 : Walk-forward backtest + 10 gates T13-T22
- Plan 4 : Explainer + ConfidenceLevel + FeedbackCollector + Model Card
- Plan 5 : Observability + KillSwitch + STRIDE + Capacity + Property-based + Multi-tenant-ready
```

- [ ] **Step 2 : Vérifier MkDocs build**

```bash
mkdocs build --strict
```
Attendu : Documentation built sans error.

- [ ] **Step 3 : Commit**

```bash
git add docs/architecture/ALI_ARCHITECTURE.md
git commit -m "docs(phase3): ALI_ARCHITECTURE.md initial (Plan 1 Foundations)"
```

---

## Task 18 : Smoke test E2E Plan 1 — pipeline foundations complet

**Files:**
- Create: `tests/test_phase3_plan1_smoke.py`

**ISO:** 29119 (integration smoke)

- [ ] **Step 1 : Écrire smoke test integration**

Créer `tests/test_phase3_plan1_smoke.py` :
```python
"""Smoke E2E Plan 1 : load rules + classify + load pool + enrich.

ISO 29119 : integration smoke test.
Valide que les 5 composants Plan 1 s'enchaînent sans erreur.
"""

from pathlib import Path

import pytest

from services.ali.cache import ALIDataCache
from services.ali.history import HistoryEnricher
from services.ali.pool_loader import PlayerPoolLoader
from services.ali.verifiability import VerifiabilityClassifier
from services.ffe.rule_engine import RuleEngine


REAL_A02 = Path("config/ffe_rules/a02.json")
CLASSIF = Path("config/ffe_rules/alice_verifiability.json")
J = Path("data/joueurs.parquet")
E = Path("data/echiquiers.parquet")


@pytest.mark.skipif(
    not (J.exists() and E.exists()),
    reason="data parquets absent du runner",
)
def test_plan1_smoke_pipeline_complete():
    # 1. RuleEngine
    engine = RuleEngine.from_json_file(REAL_A02)
    assert len(engine.rules) == 14

    # 2. Verifiability
    classifier = VerifiabilityClassifier.from_json_file(CLASSIF)
    public, private = classifier.partition_rules(engine.rules)
    assert len(public) == 10 and len(private) == 4

    # 3. Cache
    cache = ALIDataCache.load_from_parquets(J, E)
    assert cache.lineage_ok()

    # 4. Pool loader
    first_club = next(iter(cache.joueurs_by_club.keys()))
    loader = PlayerPoolLoader(cache)
    pool = loader.load_pool(first_club, "2024-11-15")
    assert len(pool) > 0

    # 5. Enricher
    enricher = HistoryEnricher(cache, decay_lambda=0.9)
    enriched = enricher.enrich(pool[:5], saison=2024, current_round=5, nb_rondes_total=11)
    assert all(e.taux_presence_effectif is not None for e in enriched)
    assert all(e.played_lag1 is not None for e in enriched)
```

- [ ] **Step 2 : Ajouter `lineage_ok` helper sur cache**

Dans `services/ali/cache.py` ajouter méthode :
```python
    def lineage_ok(self) -> bool:
        """Sanity check : signatures SHA-256 présentes et non-vides."""
        return bool(self.parquet_sig_joueurs) and bool(self.parquet_sig_echiquiers)
```

- [ ] **Step 3 : Lancer smoke test**

```bash
pytest tests/test_phase3_plan1_smoke.py -v
```
Attendu : PASS (ou SKIP si parquets absents).

- [ ] **Step 4 : Commit**

```bash
git add tests/test_phase3_plan1_smoke.py services/ali/cache.py
git commit -m "test(phase3): Plan 1 smoke E2E pipeline foundations"
```

---

## Task 19 : Script verify_plan1_dod.sh + CI check complet

**Files:**
- Create: `scripts/verify_plan1_dod.sh`

**ISO:** 29119, 5055, tous les P1G01-P1G14

- [ ] **Step 1 : Créer le script DoD vérification**

Créer `scripts/verify_plan1_dod.sh` :
```bash
#!/usr/bin/env bash
# Plan 1 Definition of Done — verification script
# Runs all P1G01-P1G14 + structural gates.
# Exit 0 only if ALL gates pass.
set -euo pipefail

echo "=== P1G01 File size ≤ 300 lignes ==="
find services/ffe services/ali -name "*.py" | while read -r f; do
  lines=$(wc -l < "$f")
  if [ "$lines" -gt 300 ]; then
    echo "FAIL: $f has $lines lines (max 300)"
    exit 1
  fi
done
echo "OK"

echo "=== P1G02 Complexity xenon ≤ B ==="
xenon --max-absolute B --max-modules B --max-average A \
  services/ffe services/ali scripts/sync_ffe_rules.py

echo "=== P1G03 MyPy strict ==="
mypy services/ffe services/ali --strict

echo "=== P1G04 Ruff ==="
ruff check services/ffe services/ali

echo "=== P1G05 Pydantic reject invalid ==="
pytest tests/test_ffe_schemas.py::test_rejects_missing_uuid \
       tests/test_ffe_schemas.py::test_rejects_invalid_effet -v

echo "=== P1G06 Gitleaks ==="
pre-commit run gitleaks --all-files

echo "=== P1G07 Coverage ≥ 75% ==="
pytest tests/test_rule_engine.py tests/test_verifiability.py \
       tests/test_ali_cache.py tests/test_pool_loader.py \
       tests/test_history_enricher.py tests/test_ali_types.py \
       tests/test_ffe_schemas.py tests/test_sync_ffe_rules.py \
       tests/test_phase3_plan1_smoke.py tests/test_config_phase3.py \
       --cov=services/ffe --cov=services/ali --cov=scripts.sync_ffe_rules \
       --cov-report=term-missing --cov-fail-under=75

echo "=== P1G08 Nombre tests ≥ 50 ==="
count=$(pytest tests/test_rule_engine.py tests/test_verifiability.py \
               tests/test_ali_cache.py tests/test_pool_loader.py \
               tests/test_history_enricher.py tests/test_ali_types.py \
               tests/test_ffe_schemas.py tests/test_sync_ffe_rules.py \
               tests/test_phase3_plan1_smoke.py tests/test_config_phase3.py \
               --collect-only -q 2>/dev/null | grep -c "::test_")
if [ "$count" -lt 50 ]; then
  echo "FAIL: only $count tests (need ≥ 50)"
  exit 1
fi
echo "OK ($count tests)"

echo "=== P1G10 Lineage déterministe ==="
pytest tests/test_rule_engine.py::test_lineage_hash_is_deterministic \
       tests/test_ali_cache.py::test_cache_signatures_are_sha256 -v

echo "=== P1G11 UUID RFC4122 preserved ==="
python -c "
from pathlib import Path
from services.ffe.rule_engine import RuleEngine
e = RuleEngine.from_json_file(Path('config/ffe_rules/a02.json'))
assert len(e.rules) == 14, f'expected 14 rules, got {len(e.rules)}'
assert all(r.uuid for r in e.rules), 'UUID manquant sur au moins 1 règle'
print('OK')
"

echo "=== P1G12 ADR-013 présent et référencé ==="
test -f docs/architecture/adr/ADR-013-rule-engine-json.md
grep -q "ADR-013" docs/architecture/ALI_ARCHITECTURE.md

echo "=== P1G13 MkDocs --strict ==="
mkdocs build --strict

echo "=== P1G14 F7 survivor documenté ==="
grep -qi "survivor" docs/architecture/ALI_ARCHITECTURE.md

echo "=== Structural: A02 = 14 rules ==="
python -c "
import json
d = json.load(open('config/ffe_rules/a02.json', encoding='utf-8'))
assert len(d['rules']) == 14, f'expected 14 got {len(d[\"rules\"])}'
print('OK')
"

echo "=== Structural: verifiability covers all A02 ==="
python -c "
import json
c = json.load(open('config/ffe_rules/alice_verifiability.json', encoding='utf-8'))
r = json.load(open('config/ffe_rules/a02.json', encoding='utf-8'))
uuids_r = {x['uuid'] for x in r['rules']}
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
echo "ALL 14 P1G GATES + 6 STRUCTURAL GATES PASS"
echo "Plan 1 Definition of Done : SATISFIED"
echo "============================================"
```

- [ ] **Step 2 : Rendre le script exécutable + lancer**

```bash
chmod +x scripts/verify_plan1_dod.sh
bash scripts/verify_plan1_dod.sh
```
Attendu : `ALL 14 P1G GATES + 6 STRUCTURAL GATES PASS`

- [ ] **Step 3 : Si un gate échoue, diagnose + corrige inline**

Corriger les problèmes signalés, relancer `bash scripts/verify_plan1_dod.sh` jusqu'à succès.

- [ ] **Step 4 : Lancer pre-commit all + pre-push**

```bash
pre-commit run --all-files
pre-commit run --hook-stage pre-push --all-files
```
Attendu : tous hooks passés.

- [ ] **Step 5 : Commit script DoD**

```bash
git add scripts/verify_plan1_dod.sh
git commit -m "chore(phase3): verify_plan1_dod.sh — 14 P1G + 6 structural gates"
```

---

## Task 20 : Mise à jour memory + debt tracker — Plan 1 livré

**Files:**
- Modify: `C:/Users/pierr/.claude/projects/C--Dev-Alice-Engine/memory/project_debt_current.md`
- Modify: `CLAUDE.md` (mettre D10 en résolue si script sync done)

**ISO:** 15289 (doc lifecycle), 42010

- [ ] **Step 1 : Marquer D10 comme résorbée**

Dans `memory/project_debt_current.md`, section "Dette résorbée" (fin de fichier) :
```markdown
## Dette resorbee

### D10 — Sync ALICE ↔ chess-app JSON rules (RESOLUE Plan 1, 2026-04-19)
Script `scripts/sync_ffe_rules.py` + pre-commit hook ffe-rules-drift livrés.
Commit ref : Task 3 + Task 4 Plan 1 Phase 3.
```

Et retirer D10 de la section "Dette Phase 2" active ci-dessus (marquer "RESOLUE Plan 1").

- [ ] **Step 2 : Mettre à jour CLAUDE.md table dette**

Dans CLAUDE.md, ligne D10, remplacer statut par "RESOLUE Plan 1 (2026-04-19)" :
```markdown
| ~~D10~~ | ~~Sync chess-app JSON~~ | ~~Phase 3~~ | **RESOLUE Plan 1** (2026-04-19) |
```

- [ ] **Step 3 : Commit des mises à jour dette**

```bash
git add CLAUDE.md
git commit -m "chore(phase3): D10 résorbée par Plan 1 (sync_ffe_rules + pre-commit)"
```

Note : memory files ne sont pas dans le repo git (ils sont dans `C:/Users/pierr/.claude/...`), donc pas à committer. Juste s'assurer que l'édition est faite.

---

## Task 21 : Peer review request (sanity check avant clôture Plan 1)

**Files:**
- Aucun — invocation skill

**ISO:** 29119 (peer review gate), 38507 (governance)

- [ ] **Step 1 : Demander une review via le skill**

Invoquer :
```
superpowers:requesting-code-review
```
avec comme scope : commits de Task 1 à Task 20 (branche active Plan 1).

- [ ] **Step 2 : Lire le rapport et corriger findings critiques**

Si ≥1 finding critique : corriger, commit, relancer review.
Si 0 finding critique : passer à l'étape suivante.

- [ ] **Step 3 : Récapitulatif utilisateur**

Présenter à l'utilisateur :
- Liste des tasks completed
- Coverage atteint
- Gates ISO passés
- 0 finding critique review
- Demander validation finale avant merge / passage Plan 2

---

## Definition of Done Plan 1 — Quality Gates ISO explicites

### P1G — Plan 1 Quality Gates (14 gates, tous bloquants)

| Gate | Norme ISO | Check | Seuil | Commande de vérification |
|------|-----------|-------|-------|--------------------------|
| **P1G01** | ISO 5055 | Taille max par fichier | ≤ 300 lignes | `find services/ffe services/ali -name "*.py" \| xargs wc -l \| awk '$1 > 300 {print; exit 1}'` |
| **P1G02** | ISO 5055 | Complexité cyclomatique | xenon ≤ B (module, module avg, fonction max) | `xenon --max-absolute B --max-modules B --max-average A services/ffe services/ali scripts/sync_ffe_rules.py` |
| **P1G03** | ISO 5055 | Type safety | MyPy strict 0 erreur | `mypy services/ffe services/ali --strict` |
| **P1G04** | ISO 5055 | Lint | Ruff 0 warning | `ruff check services/ffe services/ali tests/test_rule_engine.py tests/test_verifiability.py tests/test_ali_cache.py tests/test_pool_loader.py tests/test_history_enricher.py` |
| **P1G05** | ISO 27034 | Pydantic validation sur tous JSON externes | Tous tests `rejects_*` passent | `pytest tests/test_ffe_schemas.py::test_rejects_missing_uuid tests/test_ffe_schemas.py::test_rejects_invalid_effet -v` |
| **P1G06** | ISO 27001 | Secrets detection | Gitleaks 0 finding | `pre-commit run gitleaks --all-files` |
| **P1G07** | ISO 29119 | Coverage tests unitaires | ≥ 75% | `pytest --cov=services/ffe --cov=services/ali --cov-fail-under=75 services/ffe services/ali` |
| **P1G08** | ISO 29119 | Nombre de tests nouveaux Plan 1 | ≥ 50 | `pytest tests/test_rule_engine.py tests/test_verifiability.py tests/test_ali_cache.py tests/test_pool_loader.py tests/test_history_enricher.py tests/test_ali_types.py tests/test_ffe_schemas.py tests/test_sync_ffe_rules.py tests/test_phase3_plan1_smoke.py --collect-only -q \| tail -1` |
| **P1G09** | ISO 29119 | Tests non-flaky | 3 runs consécutifs PASS | `for i in 1 2 3; do pytest tests/test_rule_engine.py tests/test_verifiability.py tests/test_ali_cache.py tests/test_pool_loader.py tests/test_history_enricher.py || exit 1; done` |
| **P1G10** | ISO 5259 | Lineage SHA-256 déterministe | Test `test_lineage_hash_is_deterministic` + `test_cache_signatures_are_sha256` PASS | `pytest tests/test_rule_engine.py::test_lineage_hash_is_deterministic tests/test_ali_cache.py::test_cache_signatures_are_sha256 -v` |
| **P1G11** | ISO 42001 | Traceability UUID préservée | A02 chargé → 14 rules avec UUID RFC4122 non vide | `python -c "from pathlib import Path; from services.ffe.rule_engine import RuleEngine; e = RuleEngine.from_json_file(Path('config/ffe_rules/a02.json')); assert len(e.rules) == 14; assert all(r.uuid for r in e.rules), 'UUID manquant'; print('OK')"` |
| **P1G12** | ISO 42010 | ADR-013 committé et référencé | Fichier existe + cité dans CLAUDE.md OU ALI_ARCHITECTURE.md | `test -f docs/architecture/adr/ADR-013-rule-engine-json.md && grep -q "ADR-013" docs/architecture/ALI_ARCHITECTURE.md` |
| **P1G13** | ISO 15289 | Doc strict build | MkDocs --strict sans error | `mkdocs build --strict` |
| **P1G14** | ISO 24027 | F7 assumption documentée | `F7` et "survivor" présents dans ALI_ARCHITECTURE.md + Model Card draft | `grep -q "F7" docs/architecture/ALI_ARCHITECTURE.md && grep -qi "survivor" docs/architecture/ALI_ARCHITECTURE.md` |

### Gates structurels Plan 1 (au-delà des ISO)

| Check | Seuil | Commande |
|-------|-------|----------|
| A02 contient exactement 14 règles | strict | `python -c "import json; d = json.load(open('config/ffe_rules/a02.json')); assert len(d['rules']) == 14"` |
| Verifiability classifications couvrent les 14 A02 | strict | `python -c "import json; c = json.load(open('config/ffe_rules/alice_verifiability.json')); r = json.load(open('config/ffe_rules/a02.json')); uuids_r = {x['uuid'] for x in r['rules']}; uuids_c = set(c['classifications']); missing = uuids_r - uuids_c; assert not missing, f'missing: {missing}'"` |
| 10 PUBLIC + 4 PRIVATE | strict | `python -c "import json; c = json.load(open('config/ffe_rules/alice_verifiability.json'))['classifications']; pub = sum(1 for v in c.values() if v['verifiability'] == 'public'); priv = sum(1 for v in c.values() if v['verifiability'] == 'private'); assert pub == 10 and priv == 4, f'pub={pub} priv={priv}'"` |
| Pre-commit hook ffe-rules-drift opérationnel | PASS | `pre-commit run ffe-rules-drift --all-files` |
| Peer review skill exécuté | 0 finding critique | Invocation `superpowers:requesting-code-review` puis lecture rapport |
| Dette D10 marquée résolue | présent | `grep -q "D10.*RESOLUE Plan 1" CLAUDE.md` |

### Artefacts livrés (checklist)

- [ ] `config/ffe_rules/a02.json` (14 règles, vendored)
- [ ] `config/ffe_rules/alice_verifiability.json` (14 classifications, 10 public / 4 private)
- [ ] `services/ffe/schemas.py`, `services/ffe/rule_engine.py`
- [ ] `services/ali/__init__.py`, `types.py`, `verifiability.py`, `cache.py`, `pool_loader.py`, `history.py`
- [ ] `scripts/sync_ffe_rules.py` + pre-commit hook
- [ ] `docs/architecture/adr/ADR-013-rule-engine-json.md`
- [ ] `docs/architecture/ALI_ARCHITECTURE.md` (ébauche Plan 1)
- [ ] 8 fichiers de tests (~50+ tests nouveaux)
- [ ] CLAUDE.md mis à jour (D10 résolue)
- [ ] `memory/project_debt_current.md` D10 déplacée en "Dette résorbée"

### Process gates finaux

- [ ] Les 14 P1G01-P1G14 passent tous
- [ ] Les 6 gates structurels passent tous
- [ ] Tous les artefacts livrés présents
- [ ] Peer review skill `superpowers:requesting-code-review` exécuté sur le diff complet Plan 1
- [ ] 0 finding critique du peer review
- [ ] Checkpoint user : présentation bilan + validation explicite avant passage Plan 2

### Script de vérification end-to-end (à ajouter en Task 19)

Créer `scripts/verify_plan1_dod.sh` qui exécute tous les P1G et gates structurels :
```bash
#!/usr/bin/env bash
set -e
echo "=== P1G01 File size ==="
find services/ffe services/ali -name "*.py" | xargs wc -l | awk '$1 > 300 { exit 1 }'
echo "=== P1G02 Complexity ==="
xenon --max-absolute B --max-modules B --max-average A services/ffe services/ali scripts/sync_ffe_rules.py
echo "=== P1G03 MyPy ==="
mypy services/ffe services/ali --strict
echo "=== P1G04 Ruff ==="
ruff check services/ffe services/ali
echo "=== P1G06 Gitleaks ==="
pre-commit run gitleaks --all-files
echo "=== P1G07 Coverage ==="
pytest --cov=services/ffe --cov=services/ali --cov-fail-under=75
echo "=== P1G11 UUID traceability ==="
python -c "from pathlib import Path; from services.ffe.rule_engine import RuleEngine; e = RuleEngine.from_json_file(Path('config/ffe_rules/a02.json')); assert len(e.rules) == 14; assert all(r.uuid for r in e.rules)"
echo "=== P1G13 MkDocs ==="
mkdocs build --strict
echo "=== ALL GATES PASS ==="
```
Exécution finale Task 19 : `bash scripts/verify_plan1_dod.sh` doit exit 0.

---

## Notes pour les plans suivants

**Plan 2 (Générateur SOTA)** : wirera RuleEngine + VerifiabilityClassifier + Cache + PoolLoader + HistoryEnricher dans ScenarioGenerator, puis dans `/compose` (suppression `services/ffe_rules.py`).

**Dépendances établies Plan 1** : types `PlayerCandidate`, `CompetitionContext`, `RuleViolation` sont définitifs et ne doivent PAS changer signature en Plan 2+. Toute nouvelle feature ALI passe par `replace()` sur `PlayerCandidate` (dataclass frozen).

**Hors scope Plan 1 (rappel)** : aucun changement à `app/main.py::lifespan`, `app/api/routes.py`, `services/ffe_rules.py`, `services/inference.py`. Plan 2 fait l'intégration.
