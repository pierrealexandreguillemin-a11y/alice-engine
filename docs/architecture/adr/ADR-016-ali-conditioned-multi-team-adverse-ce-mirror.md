# ADR-016 : ALI conditionné multi-équipes adverse via CE-adverse miroir SOTA (Approche A)

- **Status** : Proposed (stub Phase 4a — implementation pending)
- **Date** : 2026-04-28 (creation), Phase 4a kick-off attendu T+1
- **Context** : T22 review post-mortem 2026-04-28 — finding structurel
  D-P3-19 / R-ALI-06 (CRITICAL OPEN, score 20)
- **Décideurs** : user (ML lead) + Claude assistant (validation user
  2026-04-28 "optimal sota ou go fuck yourself")
- **Supersedes** : aucun (NEW design Phase 4a upstream Phase 4b CE user)
- **Cross-références** :
  - `docs/iso/ALI_QUALITY_GATES_REPORT.md` §6.2 §7.5
  - `docs/iso/AI_RISK_ASSESSMENT.md` §R-ALI-06
  - `docs/iso/AI_RISK_REGISTER.md` §2.7 R-ALI-06
  - `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 4a + 4b
  - `docs/project/DEBT_LEDGER.md` D-P3-19
  - `memory/project_debt_current.md` D-P3-19

---

## Contexte

ALI Phase 3 prédit la composition d'**une équipe adverse spécifique** (ex
Mulhouse Philidor 1 N3) mais reçoit en entrée le **pool club total** via
`services/ali/pool_loader.py::load_pool(club_id, round_date)` sans
information sur l'allocation simultanée des **autres équipes du même club
adverse** qui jouent le même weekend (ex Mulhouse Philidor 2 N4, Mulhouse
Philidor 3 R1).

**Évidence empirique** (T22 backtest hold-out 2024 stratifié champion
mode N=70 N3 saison 2024) :

- 117 clubs alignent 2-4 équipes simultanément en N3 ronde 5 saison 2024.
- Gap recall by_pool_size = 0.28 (small Q1 0.74 vs xlarge Q4 0.46)
  cross-validation directe.
- `data/joueurs.parquet` schema 19 cols sans `equipe` — l'attribution
  est observable seulement via `data/echiquiers.parquet::blanc_equipe`.

**Mécanisme** : observed lineup N3 = joueurs Elo rang 17-24 (rangs 1-8
forcés en N1 par §3.7.b, 9-16 en N2). ALI sample dans pool total ⇒
sur-représentation top Elo en N3 ⇒ recall structurellement faible.

**Impact** : bloquant gates absolus P3G07-P3G11 (recall, jaccard, brier,
ece, mae, mcnemar legacy). Compensé Phase 3 par Wilcoxon paired SOTA
(ADR-017) qui démontre significativité statistique sur recall continu
(p=8.26e-13) malgré gates absolus FAIL.

---

## Décision

**Approche A SOTA (retenue, validée user 2026-04-28)** : ALI conditionné
par **CE-adverse miroir**.

Pour chaque club adverse multi-équipes :

1. **CE-adverse miroir** : un solveur OR-Tools simule l'allocation
   joueurs × équipes adverses sous mêmes contraintes FFE A02 §3.7.b
   (ordre Elo descendant entre équipes), §3.7.c (joueur brûlé), §3.7.d
   (même groupe), §3.7.f (noyau 50 %). Réutilise primitives
   `services/ce/` Phase 4b (à concevoir).

2. **ALI Phase 4a sample conditionné** :
   `ScenarioGenerator.generate(opponent_club_id, target_team,
   simultaneous_teams)` reçoit la liste des autres équipes adverses du
   club + leur allocation simulée. Pool sampling `target_team` = pool
   club adverse total **moins** joueurs alloués aux équipes supérieures.

3. **20 scénarios joints** par équipe cible, sous distribution
   conditionnelle vraie (vs marginal Phase 3).

4. **Re-backtest hold-out 2024** N=70 attendu :
   - recall ≥ 0.65 (vs 0.57 Phase 3)
   - Jaccard ≥ 0.50 (vs 0.39 Phase 3)
   - Brier ≤ 0.22 (vs 0.29 Phase 3)
   - McNemar n_disc ≥ 25 attendu (vs 3) → puissance α=0.05 OK

**Approche B rejetée** (joint sampling sans CE-adverse miroir) :

- moins SOTA car ne réutilise pas les primitives FFE OR-Tools déjà
  prévues Phase 4b
- risque divergence logique entre CE-user et inférence ALI
- duplication code OR-Tools

Approche A garantit cohérence mathématique CE-user ↔ ALI.

---

## Conséquences

### Positives

- ALI sample dans la **vraie distribution conditionnelle** observée.
- Re-utilisation primitives FFE OR-Tools Phase 4b (DRY).
- Gates absolus P3G07-P3G11 atteignables sans hack de seuil.
- Coherence math CE-user ↔ ALI garantie.

### Négatives

- Phase 4a NEW upstream Phase 4b ⇒ +complexité roadmap (split 4a/4b).
- Solveur OR-Tools miroir = compute ~5-10s additionnel par match
  multi-équipes (vs ~2s Phase 3 marginal).
- D8 Phase 3.5 fairness/robustness reste prérequis (entry gate Phase 4a).

### Implementation envisagée (Phase 4a kick-off TBD)

- Refactor `services/ali/pool_loader.py` : signature `load_pool(club_id,
  round_date, exclude_players: list[str] = []) -> list[PlayerCandidate]`.
- NEW `services/ali/adverse_ce.py` : OR-Tools solver miroir, réutilise
  primitives `services/ce/` Phase 4b.
- Refactor `services/ali/generator.py::generate(...)` : ajout
  `simultaneous_teams: list[str]` ; orchestration : appel CE-adverse
  pour équipes supérieures → pool exclude → MC sample équipe cible.
- Update `app/api/routes.py::/compose` : assembler `simultaneous_teams`
  depuis `data/echiquiers.parquet` (colonnes `saison` + `ronde` +
  `groupe` permettent grouping).
- ADR-016 status `Proposed → Accepted` à la création des primitives
  `services/ce/` Phase 4b.

---

## Sources SOTA

- FFE A02 (rules officielles interclubs) §3.7.b/c/d/f.
- OR-Tools constraint programming (Google) — solveur Phase 4b cible.
- ISO/IEC 24029:2021 §6 (robustesse — multi-team consistency).
- ISO/IEC 25059:2023 §6 (quality model AI — fonctionnalité prédictive).

---

**Validation** : ADR-016 a force normative à partir de Phase 4a kick-off.
Tant que status = `Proposed`, tous les forward references dans la doc
ISO/Plan/Roadmap sont à interpréter comme "design retenu, implementation
Phase 4a pending".
