# Pre-Push Workflow — Alice Engine

**Owner** : DevOps / R-PRE-PUSH-01
**Status** : 🟡 PARTIAL (mkdocs-build viré pre-push 2026-04-29 22h, pytest-fast 145s reste à descendre < 90s)
**Contrat** : `git push` total < 90 secondes wallclock. Tests lourds en CI async.

## Pourquoi ce contrat

`feedback_push_max_2min.md` — workflow developer-friendly. Push qui prend
> 2 min = développeur perd le flow + frustration. SOTA dev practice 2025 :
pre-commit/pre-push light en local, full suite + coverage en CI async.

## Hooks pre-push actuels (`.pre-commit-config.yaml`)

| Hook | Cible | Mesure 2026-04-29 |
|------|-------|---------------------|
| `pytest-fast` | < 60s | **145s** (FAIL — à descendre via `@pytest.mark.slow`) |
| `xenon` complexity | < 5s | ~3s |
| `pip-audit` security | < 15s | ~10s |
| `mkdocs-build` | **VIRÉ pre-push 22h** → `stages: [manual]` | ~5-6 min (était bloquant) |

**Total cible** : ≤ 90s.
**Mesure 22h09 push 2b9e247 (avant fix)** : 9m22s. Cause : mkdocs.

## Fix en cours R-PRE-PUSH-01

### 1. mkdocs-build viré pre-push (FAIT, uncommitted)

`.pre-commit-config.yaml` ligne ~257-264 : `stages: [pre-push]` →
`stages: [manual]`. Run via `pre-commit run --hook-stage manual mkdocs-build`
ou en CI sur PR.

### 2. Marker @pytest.mark.slow (TODO)

Profiler pytest-fast `--durations=15` pour identifier les tests > 5s.
Ajouter `@pytest.mark.slow` sur :

- Tests qui chargent parquets entiers (`data/joueurs.parquet` 95K joueurs)
- Tests qui font full backtest run (>5s)
- Property-based hypothesis tests (>5s)
- Smoke pipeline E2E (>5s)
- Coverage measurement runs

### 3. Déclarer marker dans `pyproject.toml`

```toml
[tool.pytest.ini_options]
markers = [
  "slow: tests > 5s, exclus par défaut en pre-push (run via pytest -m slow)",
]
```

### 4. Validation finale

```bash
# Mesure isolée hook fast
time pre-commit run --hook-stage pre-push pytest-fast
# Cible : < 60s

# Mesure push complet
time git push origin master
# Cible : < 90s wallclock (sans --no-verify)
```

## Hooks manual (run on-demand ou CI)

Ces hooks NE BLOQUENT PAS le push :

| Hook | Run command |
|------|-------------|
| `mkdocs-build` | `pre-commit run --hook-stage manual mkdocs-build` |
| `architecture-health` | `pre-commit run --hook-stage manual architecture-health` |
| `generate-graphs` | `make graphs` |
| `update-iso-docs` | `make iso-docs` |

## CI GitHub Actions (TODO Phase 4 entry gate)

À implémenter dans `.github/workflows/ci.yml` :

- **Pull request** : full pytest + coverage 70% + mkdocs build strict + ruff +
  mypy + xenon + pip-audit + bandit + gitleaks
- **Push to master** : idem + DVC pipeline reproduce stages
- **Nightly** : robustness/fairness multi-fold backtest + smoke E2E réel

Cible : 100% des slow tests couverts en CI, jamais en pre-push.

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
```

## Override `--no-verify` (NE PAS UTILISER)

`git push --no-verify` interdit sauf cas exceptionnels documentés
(`feedback_no_verify` + section CLAUDE.md). Usage 2026-04-29 day 1
documenté en CLAUDE.md ligne 93. Si nécessaire :

1. Pré-valider tests localement en chunks (`pytest tests/X --tb=line`)
2. Pré-valider lint+types+security manuellement
3. Logger dans CLAUDE.md le pourquoi avec date+SHA
4. Ouvrir dette R-PRE-PUSH-XX pour fix hook

## Historique R-PRE-PUSH-01

- **2026-04-29 jour** : push pre-push hook stale 2h, push final
  `b81a328` via `--no-verify` exceptionnel.
- **2026-04-29 22h** : tentative push `2b9e247` sans `--no-verify` →
  mesure réelle 9m22s wallclock (FAIL). Cause `mkdocs-build` pré-push
  identifiée. Fix `mkdocs-build → manual` appliqué (uncommitted).
- **TODO** : commit fix + marker tests slow + push validé < 90s + CI async.
