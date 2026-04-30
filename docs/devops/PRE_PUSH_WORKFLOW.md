# Pre-Push Workflow — Alice Engine

**Owner** : DevOps / R-PRE-PUSH-01
**Status** : ✅ RESOLUE 2026-04-30 (master HEAD origin = `8919ca0`, CI run 25178421194 ALL GREEN)
**Contrat** : `git push` total < 90 secondes wallclock. Tests lourds en CI async.

## Pourquoi ce contrat

`feedback_push_max_2min.md` — workflow developer-friendly. Push qui prend
> 2 min = développeur perd le flow + frustration. SOTA dev practice 2025 :
pre-commit/pre-push light en local, full suite + coverage en CI async.

## Hooks pre-push actuels (`.pre-commit-config.yaml`)

| Hook | Cible | Mesure 2026-04-30 |
|------|-------|---------------------|
| `check-added-large-files` | <2s | ~1s ✅ |
| `pytest-fast` | < 60s | **60.24s** ✅ (1660 passed, 32 deselected) |
| `xenon` complexity | < 5s | ~3s ✅ |
| `pip-audit` security | < 15s | ~10s ✅ |
| `mkdocs-build` | **VIRÉ pre-push** → `stages: [manual]` | ~5-6 min (était bloquant) |

**Total cible** : ≤ 90s.
**Mesure 2026-04-30 (après fixes)** : 78–87s wallclock stabilisé.

## Architecture finale

### Pre-push (local, <90s)

- ISO 27001 — large file detection
- ISO 29119 — pytest-fast (`-m "not slow"` + 19 ignores)
- ISO 25010 — xenon complexity
- ISO 27001 — pip-audit (CVE local)

### Manual stage (run on-demand)

```bash
pre-commit run --hook-stage manual mkdocs-build       # ISO 15289
pre-commit run --hook-stage manual architecture-health # ISO 42010
pre-commit run --hook-stage manual generate-graphs     # ISO 42010
pre-commit run --hook-stage manual update-iso-docs     # ISO 15289
make graphs | make iso-docs                            # raccourcis
```

### CI GitHub Actions (`.github/workflows/ci.yml`)

| Job | Trigger | Durée |
|-----|---------|-------|
| Quality Checks | PR + push master | ~30s |
| Tests & Coverage | PR + push master | ~3 min |
| Security Audit | PR + push master | ~30s |
| Complexity Analysis | push master | ~1 min |
| Generate Artifacts | push master | ~3 min |
| Update ISO Documentation | release only | ~3 min |

CI utilise `uv pip install --system --prerelease=allow` (résolveur PubGrub
de Astral) — pip standard saturait le résolveur sur deps ML lourds
(autogluon umbrella + lightning).

## R-PRE-PUSH-01 — fixes appliqués (2026-04-30)

| # | Commit | Fix | Effet |
|---|--------|-----|-------|
| 1 | `6c92ba4` | `mkdocs-build` `stages: [pre-push]` → `[manual]` | -5 à -6 min |
| 2 | `0116e93` | `@pytest.mark.slow` sur `tests/test_ali_cache.py` (10 tests, 6 reloads parquets ~684s) | -11 min |
| 3 | `96bfeff` | `tests/conftest.py` `cleanup_after_test` autouse function-scoped → module-scoped (gc.collect() ~5.8s × 60+ tests) | -6 min |
| 3 | `96bfeff` | `pytestmark = pytest.mark.slow` sur `test_generator.py`, `test_pool_loader.py`, `test_history_enricher.py::test_enricher_integration_real_parquets`, `test_ground_truth.py` (tous chargent vrais parquets via `ali_data_cache` fixture session ~92s setup) | élimine setup heavy |
| 3 | `96bfeff` | déclaration marker `slow` dans `pyproject.toml [tool.pytest.ini_options]` | hook `-m "not slow"` filtre auto |
| 4 | `4104c4c` | ruff format 3 fichiers (split implicit string concat sur 2 lignes) + CI security audit `pip-audit -r requirements.txt -r requirements-dev.txt` (sans installer env complet) | CI Quality + Security ✅ |
| 5 | `00df4f5` | CI Tests + Artifacts + ISO-docs : `pip install` → `uv pip install --system` (résolveur PubGrub) | CI install ✅ |
| 6 | `bf9330c` | `uv pip install --prerelease=allow` (autogluon stable 1.5.0 contraintes lightning insolvables) | autogluon dev wheels OK |
| 7 | `afd1642` | `requirements.txt` : `autogluon>=1.5.0` → `autogluon-tabular>=1.5.0` (ADR-011 : umbrella inutile, multimodal lightning unsolvable) | uv résolution propre |
| 8 | `8919ca0` | `requirements.txt` : add `statsmodels>=0.14.0` (drift local vs CI, mcnemar tests) | CI Tests ✅ |

## Override `--no-verify` (NE PAS UTILISER)

`git push --no-verify` interdit sauf cas exceptionnels documentés
(`feedback_no_verify` + section CLAUDE.md). Usage 2026-04-29 day 1
documenté en CLAUDE.md ligne 93 — désormais résolu, plus de
`--no-verify` toléré.

Si nécessaire :

1. Pré-valider tests localement en chunks (`pytest tests/X --tb=line`)
2. Pré-valider lint+types+security manuellement
3. Logger dans CLAUDE.md le pourquoi avec date+SHA
4. Ouvrir dette R-PRE-PUSH-XX pour fix hook

## Pattern pour développeurs

```bash
# Workflow normal
git add <files>
git commit -m "feat(...)"     # pre-commit hooks rapides (~10s)
git push                       # pre-push hooks < 90s

# Si push > 90s
time pre-commit run --hook-stage pre-push --all-files
# Identifier le hook coupable, soit
# - le marquer slow
# - le déplacer en manual
# - l'optimiser

# Profiler pytest-fast slow tests
python -m pytest tests/ -m "not slow" --durations=15 [+ ignores] -x
```

## Profiling pytest-fast (cible <60s)

Si pytest-fast dérive au-dessus de 60s :

1. Run avec `--durations=20` pour identifier les > 5s
2. Vérifier si test charge les vrais parquets (`data/joueurs.parquet`,
   `data/echiquiers.parquet`, fixture `ali_data_cache` session-scoped)
3. Si oui → `pytestmark = pytest.mark.slow` (file-level) ou
   `@pytest.mark.slow` (test-level)
4. Documenter dans le commit message + ce doc
5. CI doit toujours valider via `pytest -m slow tests/` en async

## Lecons retenues

- **`gc.collect()` autouse function-scoped + fixture session-scoped GB**
  = poison cumulatif. Toujours scope=module au minimum.
- **`autogluon` umbrella** = trap deps multimodal+lightning. Utiliser
  `autogluon-tabular` quand seul `TabularPredictor` est utilisé.
- **uv > pip** sur deps ML lourds. Ajouter `--prerelease=allow` quand
  stable contraintes insolvables.
- **Drift requirements.txt vs env local** : valider en CI fresh install
  ou via `pip-compile` (à terme — futur Phase 5).

## Historique R-PRE-PUSH-01

- **2026-04-29 jour** : push pre-push hook stale 2h, push final
  `b81a328` via `--no-verify` exceptionnel.
- **2026-04-29 22h** : tentative push `2b9e247` sans `--no-verify` →
  mesure réelle 9m22s wallclock (FAIL). Cause `mkdocs-build` pré-push
  identifiée. Fix `mkdocs-build → manual` appliqué (uncommitted).
- **2026-04-30** : 8 commits de fixes successifs (`6c92ba4` →
  `8919ca0`), trajectoire 9m22s → 78–87s wallclock. CI ALL GREEN.
  R-PRE-PUSH-01 RESOLUE.
