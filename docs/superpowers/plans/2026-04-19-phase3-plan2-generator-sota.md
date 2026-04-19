# Phase 3 · Plan 2 — Générateur SOTA : Copula + LHS + ScenarioGenerator

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Tasks ordonnées avec checkbox, TDD strict.

**Goal:** Livrer le générateur ALI complet SOTA (copule gaussienne F6 + LHS/antithetic F5 + TopK enumeration), wirer dans `/compose`, supprimer `services/ffe_rules.py` legacy. Plan 1 Foundations déjà mergé.

**Architecture:** ScenarioGenerator orchestrateur qui consomme les modules Plan 1 (RuleEngine, VerifiabilityClassifier, ALIDataCache, PlayerPoolLoader, HistoryEnricher) + 3 nouveaux samplers (CopulaJointSampler, TopKEnumerator, MonteCarloSampler) → ScenarioSet 20 scénarios pondérés → boucle inférence Phase 2 × 20 → CE moyenne pondérée.

**Tech Stack:** Python 3.13, pandas, numpy, scipy (pour copule + LHS), pytest. Pas de nouvelle dep majeure.

**Scope:** 16 tasks, ~3 semaines estim. Coverage cible ≥75%, 14 P2G ISO gates.

**Principes:** KISS, DRY (réutiliser Plan 1), TDD, ISO documenté à chaque étape.

---

## File Structure

**Créer :**
```
services/ali/joint_sampler.py          # CopulaJointSampler (F6 SOTA)
services/ali/topk.py                   # TopKEnumerator
services/ali/monte_carlo.py            # MonteCarloSampler + LHS + antithetic
services/ali/scenario.py               # Scenario, ScenarioSet, BoardAssignment, Lineup types
services/ali/generator.py              # ScenarioGenerator orchestrateur
docs/architecture/adr/ADR-014-ali-mc-hybride-sota.md
scripts/verify_plan2_dod.sh
tests/test_joint_sampler.py
tests/test_topk.py
tests/test_monte_carlo.py
tests/test_scenario.py
tests/test_generator.py
tests/test_phase3_plan2_smoke.py       # E2E compose flow
```

**Modifier :**
```
services/ali/pool_loader.py            # Task 1 D-P3-04 fix (licence_active correctement)
services/ali/history.py                # Si besoin pour F1 fit data
app/api/routes.py                      # Wire ScenarioGenerator dans /compose
app/main.py                            # lifespan : load ScenarioGenerator + samplers
docs/architecture/ALI_ARCHITECTURE.md  # Update Plan 2 components
```

**Supprimer (à fin Plan 2 quand routes wirées) :**
```
services/ffe_rules.py                  # remplacé par RuleEngine
```

---

## Task 1 : D-P3-04 fix — F7 réinterprété + validation schema joueurs.parquet

**Files:**
- Modify: `services/ali/pool_loader.py` (`_row_licence_active`)
- Modify: `tests/test_pool_loader.py` (vérifier comportement)
- Modify: `docs/architecture/ALI_ARCHITECTURE.md` section F7

**ISO:** 24027 (fairness), 5259 (lineage), 42001 (assumption explicit)

**Découverte** : `joueurs.parquet` n'a PAS de flag explicite `licence_active`. Valeurs `elo_type` ∈ {E, F, N, ''} — pas de ARCHIVE/INACTIVE. Tous les joueurs présents sont actifs par construction (database FFE active license at scrape time).

**Fix** : `_row_licence_active` retourne `True` pour toute ligne du parquet. Documenter que F7 est enforcé via "membership in joueurs.parquet[club==X]" et non via flag — la survivor bias est protégée naturellement car les joueurs ayant quitté le club n'apparaissent plus dans `joueurs.parquet[club==X]`.

- [ ] **Step 1:** Modifier `services/ali/pool_loader.py::_row_licence_active`:
```python
def _row_licence_active(row) -> bool:
    """Return True : joueurs.parquet ne contient que des licences actives FFE.

    F7 (survivor bias) est enforce via la composition meme du parquet :
    joueurs.parquet est mis a jour regulierement par scraping FFE actif.
    Joueurs ayant quitte le club n'apparaissent plus dans
    joueurs_by_club[club_id] -> filtre implicite par membership.

    Source : analyse schema reel 2026-04-19, valeurs elo_type = {E, F, N, ''}.
    Aucune valeur ARCHIVE/INACTIVE observee sur 83K lignes.
    """
    return True
```

- [ ] **Step 2:** Mettre à jour `services/ali/pool_loader.py::_override_to_candidate` : conserve `licence_active` du dict si fourni (capitaine peut signaler licence en cours de renouvellement).

- [ ] **Step 3:** Lancer `pytest tests/test_pool_loader.py -v` → 5/5 PASS (le test `test_loader_survivor_filter_excludes_inactive` reste valide via override).

- [ ] **Step 4:** Mettre à jour `docs/architecture/ALI_ARCHITECTURE.md` section F7 :

Remplacer la note F7 actuelle par :
```markdown
### F7 — Survivor bias filter (interprétation FFE)

`joueurs.parquet` est la base FFE des licences actives au moment du scraping.
Les joueurs ayant quitté un club n'apparaissent plus dans `joueurs_by_club[club_id]`,
donc F7 est enforcé **implicitement** via la composition du parquet.

`_row_licence_active` retourne True pour toutes les lignes — pas de flag
ARCHIVE/INACTIVE observé sur 83K joueurs (valeurs `elo_type` ∈ {E, F, N, ''}).

**Override** : un capitaine peut passer `licence_active: false` dans `overrides`
pour signaler une licence en cours de renouvellement (cas exceptionnel hors
fenêtre de scraping).

Source : audit schema réel 2026-04-19 (Plan 2 Task 1, finding D-P3-04).
```

- [ ] **Step 5:** Commit:
```bash
git add services/ali/pool_loader.py tests/test_pool_loader.py docs/architecture/ALI_ARCHITECTURE.md
git commit -m "fix(phase3): D-P3-04 F7 reinterpreted — implicit via joueurs.parquet membership"
```

---

## Task 2 : ADR-014 ALI Monte Carlo Hybride SOTA

**Files:** Create `docs/architecture/adr/ADR-014-ali-mc-hybride-sota.md`

**ISO:** 42010

- [ ] **Step 1:** Créer ADR avec sections : Contexte (Plan 1 livre Foundations, Plan 2 livre générateur), Décision (10 TopK déterministe + 10 MC stochastique LHS+antithetic, copule gaussienne pour joint sampling), Conséquences (SOTA full ISO 42001, complexité +250 lignes), Alternatives rejetées (PtO vs DFL, Gibbs vs copule, IID vs LHS, point estimate vs conformal).

- [ ] **Step 2:** mkdocs build --strict pass.

- [ ] **Step 3:** Commit `docs(adr): ADR-014 ALI MC hybride SOTA`.

---

## Task 3 : Scenario types

**Files:**
- Create `services/ali/scenario.py`: `BoardAssignment`, `Lineup`, `Scenario`, `ScenarioSet` (frozen dataclasses)
- Create `tests/test_scenario.py`: 4-5 tests

**ISO:** 5055, 29119, 5259 (lineage_hash dans ScenarioSet)

- [ ] Tests d'abord (frozen, hash, validation), puis impl, puis commit.

```python
@dataclass(frozen=True)
class BoardAssignment:
    board: int
    player: PlayerCandidate
    p_assignment: float

@dataclass(frozen=True)
class Lineup:
    team_size: int
    assignments: tuple[BoardAssignment, ...]

@dataclass(frozen=True)
class Scenario:
    lineup: Lineup
    joint_prob: float       # unnormalized
    weight: float           # normalized
    source: Literal["topk", "monte_carlo"]

@dataclass(frozen=True)
class ScenarioSet:
    scenarios: tuple[Scenario, ...]
    opponent_club_id: str
    round_date: str
    generated_at: str
    lineage_hash: str       # SHA-256

    def validate(self) -> None:
        # sum weights = 1.0 ± 1e-4
        # len = 20
        # scenarios distincts (hash sur (player nr_ffe, board) tuple)
```

---

## Task 4 : CopulaJointSampler (F6 SOTA)

**Files:**
- Create `services/ali/joint_sampler.py` (~200 lignes)
- Create `tests/test_joint_sampler.py` (~10 tests)

**ISO:** 42001 (SOTA documented), 5259 (lineage)

**Algorithme :**
1. `fit(history, club_id)` : co-presence matrix N×N depuis echiquiers.parquet, Spearman rank correlation
2. Transform marginales → N(0,1) via empirical CDF rangs
3. `sample(rng)` : draw N(0, Σ) via Cholesky → inverse-CDF binaire selon `taux_presence_effectif`

**Sources** : Sklar 1959, Genest & Favre 2007, Nelsen 2006.

- [ ] Test : matrice fit positif semi-définie, samples ∈ {0,1}^N, marginales empiriques ≈ taux_presence input, reproductibilité avec seed.

---

## Task 5 : TopKEnumerator (déterministe)

**Files:**
- Create `services/ali/topk.py` (~150 lignes)
- Create `tests/test_topk.py` (~6 tests)

**ISO:** 5055 (SRP), 29119 (déterministe testable)

**Algorithme :** branch-and-bound priorisé, énumération greedy avec backtracking sur les K positions, contraintes RuleEngine PUBLIC, score = produit `P(present) × P(board=k|present)`. Top-10 lineups distincts par joint_prob desc.

- [ ] Tests : déterminisme (même input → même output), respect ordre Elo, K lineups distincts garantis.

---

## Task 6 : MonteCarloSampler + LHS + antithetic (F5 SOTA)

**Files:**
- Create `services/ali/monte_carlo.py` (~180 lignes)
- Create `tests/test_monte_carlo.py` (~8 tests)

**ISO:** 42001 (SOTA), 25010 (rejection_rate observable)

**Algorithme :**
1. Generate LHS sample `u ∈ [0,1]^N × 10` (scipy.stats.qmc.LatinHypercube)
2. Antithetic pairs : pour chaque tirage `u`, ajouter `1-u` → 5 pairs = 10 scénarios neg corrélés
3. Inverse-transform via CopulaJointSampler → presence vector
4. Conditionner par F3 streak features
5. Select team_size + assign boards Elo desc
6. RuleEngine.validate_lineup → reject/resample max 50 retries
7. Compute `joint_prob` per scenario

**Sources** : McKay 1979, Hammersley & Morton 1956, Owen 2013.

- [ ] Tests : seed reproductibilité, rejection rate < 30%, LHS coverage stratifié, antithetic pairs négativement corrélés.

---

## Task 7 : ScenarioGenerator orchestrateur

**Files:**
- Create `services/ali/generator.py` (~120 lignes)
- Create `tests/test_generator.py` (~6 tests + smoke)

**ISO:** 5259 (lineage_hash propagé), 5055 (SRP), 29119

```python
class ScenarioGenerator:
    def __init__(self, engine, classifier, cache):
        ...

    def generate(self, opponent_club_id, round_date, context, overrides=None,
                 n_topk=10, n_mc=10) -> ScenarioSet:
        # 1. PlayerPoolLoader.load_pool (F7)
        # 2. HistoryEnricher.enrich (F2 + F3)
        # 3. CopulaJointSampler.fit
        # 4. RuleEngine.filter_candidates → eligible
        # 5. classifier.partition_rules → public, private
        # 6. TopKEnumerator.enumerate(eligible, n_topk, engine, public)
        # 7. MonteCarloSampler.sample(eligible, n_mc, copula, engine, public)
        # 8. merge + normalize weights
        # 9. validate_scenario_set
        # 10. return ScenarioSet(lineage_hash=SHA-256(parquet+rules+inputs+λ))
```

---

## Task 8 : Wire `/compose` route + lifespan

**Files:**
- Modify `app/main.py::lifespan`
- Modify `app/api/routes.py::compose_route`

**ISO:** 25010 (latence <2s), 27001 (audit log)

- [ ] Lifespan : `app.state.scenario_generator = ScenarioGenerator(...)` + log lineage hash startup
- [ ] /compose : appel scenario_generator.generate(...) → boucle Phase 2 inference × 20 → CE moyenne pondérée → response avec metadata
- [ ] Latence cible <2s mesurée via benchmark
- [ ] Audit log inclut `lineage_hash` + `rule_uuids_applied`

---

## Task 9 : Suppression `services/ffe_rules.py` legacy

**Files:**
- Delete `services/ffe_rules.py`
- Verify `app/api/routes.py` n'importe plus de l'ancien fichier

- [ ] Recherche `from services.ffe_rules` partout, remplacer par RuleEngine
- [ ] Tests Phase 2 existants doivent toujours passer (refactor compatible)

---

## Task 10 : Smoke E2E Plan 2 — flow `/compose` complet

**Files:** Create `tests/test_phase3_plan2_smoke.py`

- [ ] Test E2E : POST /compose avec données réelles (TestClient), assertion réponse contient lineup + scenarios_summary + metadata.lineage_hash, latence < 2s.

---

## Task 11 : Quality gates T18-T21 (structurels)

T18 sum(weights)=1.0, T19 len=20, T20 distincts, T21 rejection_rate≤30%.

- [ ] Tests dédiés dans test_generator.py.

---

## Task 12 : ALI_ARCHITECTURE.md update Plan 2

- [ ] Ajout section Plan 2 (samplers, generator, wire), maj diagramme.

---

## Task 13 : verify_plan2_dod.sh

Script analogue Plan 1, gates :
- P2G01-P2G14 (line counts, xenon, mypy, ruff, tests, coverage 75%, gates structurels T18-T21)

---

## Task 14 : Coverage check + lint complet

- [ ] pytest --cov sur tous tests Plan 1 + Plan 2 → ≥75%

---

## Task 15 : Update memory + CLAUDE.md

- [ ] D-P3-04 résorbée (Task 1)
- [ ] Findings Plan 2 tracés s'il y a lieu
- [ ] CLAUDE.md table couches mise à jour : ALI prédiction adverse passe de FALLBACK à DONE-SOTA

---

## Task 16 : Peer review skill + checkpoint user final

- [ ] Invoquer `superpowers:requesting-code-review` sur diff complet Plan 2
- [ ] Récap user : tests, gates, dette tracée, demande validation merge
- [ ] Si APPROVED : merge fast-forward sur master

---

## Definition of Done Plan 2

### Quality gates
- [ ] 14 P2G gates verts
- [ ] Coverage `services/ali/` ≥ 75% (Plan 1 + Plan 2 combinés)
- [ ] Latence `/compose` < 2s (benchmark mesuré)
- [ ] xenon ≤ B, mypy strict, ruff clean

### Résorption dette
- [ ] D2 (ALI = tri Elo 1 scénario) → résolue
- [ ] D5 (services/composer.py legacy) → décision finale (supprimé ou documenté)
- [ ] D-P3-04 (F7 schema) → résolue Task 1
- [ ] D-P3-08 (filter_candidates couvre que 3.7.j) → résolue ou tracée Plan 3

### Artefacts
- [ ] ADR-014 ALI MC hybride SOTA
- [ ] ALI_ARCHITECTURE.md mis à jour
- [ ] Smoke E2E /compose passe avec données réelles
- [ ] Peer review APPROVED

### Process
- [ ] Checkpoint user final avant merge
