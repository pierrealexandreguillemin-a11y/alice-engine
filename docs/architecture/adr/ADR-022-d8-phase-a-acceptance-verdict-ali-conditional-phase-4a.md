# ADR-022 : D8 Phase A acceptance verdict — Phase 4a ALI conditional retained (D-P3-19 empirical confirmation)

**Date** : 2026-05-16
**Statut** : Accepté (user explicit 2026-05-16, pivot diagnostique guidé ISO + SOTA ML)
**Standards** : ISO/IEC TR 24029-2:2024 (robustness), ISO/IEC TR 24027:2021 (fairness),
ISO/IEC 42001:2023 (lifecycle traceability), ISO/IEC 42005:2025 (impact assessment),
ISO/IEC 23894:2023 (AI risk management), ISO/IEC 42010 (architecture decision record),
ISO/IEC 5055:2021 (SRP single-source-of-truth)

---

## Contexte

Phase A D8 audit (ADR-019 : saison 2024 × 5 divisions) complétée le 2026-05-16 :
- 5/5 Kaggle kernels COMPLETE (Top 16 v4 = 82 matches + N1 v3 = 56 + N2 v3 = 101 + N3 v3 = 168 + N4 v3 = 85)
- Cumul **492 matches** (>> floor spec §1.3 = 200)
- Champion artefact (MLP 18f stacking + temp_scaler) SHA-256 identique cross-divisions (champion invariant)

Aggregator local execution (`scripts/d8/aggregate.py --mode phase-a`) :
- **6/19 PASS, 13 FAIL, 0 INCONCLUSIVE**
- 7 stress/DRO closures réelles wirées (D-2026-05-09 RÉSOLUE), 0 INCONCLUSIVE pending data.

Reference : `reports/d8/phase_a/2026-05-16-acceptance.md` (verdict détaillé case-by-case).

---

## Décision

### Decision principale

**Phase 4a (ALI joint conditionnel CE-adverse miroir, Approche A SOTA validée user 2026-04-28) RETENUE comme entry strategy.** 10/13 FAIL = confirmation empirique quantifiée D-P3-19 / R-ALI-06 (déjà identifiée T22 post-mortem 2026-04-28). Phase 4a résout structurellement par construction ranking-based.

### Decision dérivées (3)

1. **Phase 3.5b cleanup mineur** (tracé, non-bloquant Phase 4a entry) :
   - Fix G_ROB_07 threshold `set_size_max ≤ 3.0` (absolu) → `set_size_mean / support_max ≤ 0.50` (ratio, Angelopoulos 2023 §4.2 efficiency relatif)
   - Fix kernel `scripts/d8/dro.py` agrégation `recall_worst_case = min over per-match` → `percentile(5%) over per-match` (Sinha 2018 §4 prévu distribution shift continu, pas outlier dominant)
   - Refactor `scripts/d8/aggregate.py::_fairness_metrics` : consommer `r["breakdowns"]["by_elo_strata"]` (n statistique) au lieu de réimplémenter proxy `_by_ronde` (statistique faible par ronde)

2. **Phase 3.6 retraining adversarial (Madry 2018 PGD + Goodfellow 2015 augmentation Elo)** : **NON RETENUE comme première intention** (patch symptomatique qui ne corrige pas la cause profonde D-P3-19 manque conditionnement multi-équipes A02 §3.7.b). **PLANIFIÉE comme contingency** si robustness post-Phase 4a insuffisante. Aucun engagement effort tant que Phase 4a entry gate non-franchi.

3. **Aggregator patch ISO 5055 SRP préservé** (cette session, RÉSOLUE) :
   - `scripts/d8/aggregate.py::main()` : argparse `--mode {saison,phase-a}` + branche `load_audit_reports` ADR-019. Backward-compat saison-mode (4 saisons × multi-division) préservée 100%.
   - `scripts/d8/types.py::D8FullReport` : 2 nouveaux champs `audit_mode: str = "saison"` (default) + `divisions: list[str]` (phase-a populated).
   - `tests/d8/test_aggregate_phase_a_mode.py` : 12 tests nouveau mode + 17 tests saison-mode existants restent verts (29 PASS).

---

## Mécanisme structurel D-P3-19 confirmé empiriquement

**Hypothèse posée T22 post-mortem 2026-04-28** :
> "117 clubs alignent 2-4 équipes simultanément en N3 ronde 5 saison 2024. FFE A02 §3.7.b force les top Elo en équipe supérieure : observed équipe 1 = Elo rang 1-8, équipe 2 = rang 9-16, équipe 3 = rang 17-24. ALI Phase 3 ignore cette contrainte → sample top Elo dans équipe N3 alors qu'en réalité ils sont en N1."

**Confirmation empirique quantifiée 2026-05-16** :

| Signature | Mesure | Cohérence mécanisme |
|-----------|-------:|---------------------|
| Cross-niveau gap stress Elo 1% | Top 16 = 0.335 vs N4 = 0.017 (×20) | Top 16 = écarts Elo 5-15 pts entre élite 2500-2700 → 1% noise (±25 pts) flippe rankings adjacents → A02 §3.7.b redistribue alloc équipes → pool effectif Top 16 change. N4 = écarts Elo 80+ pts → 1% noise (±15 pts) ne flippe rien. |
| ECE_ali per-strata uniforme | Q1 (n=39) = 0.468, Q2 (n=38) = 0.467 | Modèle sample pool club total naïf alors qu'empiriquement joueur top Elo va systématiquement en équipe sup (déterministe A02 §3.7.b). Calibration impossible sans conditionnement. |
| Roster turnover 5% Top 16 amplifié 7× | Top 16 = 0.353 vs N4 = 0.054 | Joueurs élite ont identités Elo spécifiques (peu de substituts à même niveau), N4 a "pool de substituts" plus large. |

**SOTA-cohérence** :
- Mehrabi 2021 §4.1 fairness max_gap → 0.226 vs threshold 0.10
- Pleiss 2017 §4 calibration ECE per-group → 0.496 vs 0.05
- Goodfellow 2015 ε-robustness → 0.119 vs 0.020
- Tran 2022 §3.4 roster turnover → 0.144 vs 0.050

---

## Pourquoi Phase 4a résout (logique SOTA)

ALI Phase 4a Approche A SOTA (cf. spec `2026-03-23-alice-prod-roadmap-design.md` §Phase 4a) :

1. **CE-adverse miroir** : OR-Tools solver simule l'allocation adversaire joueurs × équipes simultanées sous mêmes contraintes FFE A02 §3.7.b/c/d/f (ordre Elo descendant + joueur brûlé + même groupe + noyau 50%).

2. **ALI conditionné** : `ScenarioGenerator.generate(opponent_club_id, target_team, simultaneous_teams)` reçoit la liste des autres équipes adverses du club + leur allocation simulée. Pool sampling `target_team` = pool club total **moins** joueurs alloués aux équipes adverses supérieures (résolution mécanisme A02 §3.7.b).

3. **Robustness ranking-invariant by design** : Sample compositions basé sur **rankings** Elo (top-N par équipe), pas sur **valeurs absolues** Elo. ±1% noise Elo ne flippe que les joueurs adjacents (eux-mêmes une dimension d'incertitude légitime à modéliser via Sklar 1959 copule gaussienne + Monte Carlo conditionnel).

4. **Re-backtest hold-out 2024** attendu (cf. D-P3-19) : recall ≥ 0.65 (vs 0.57), Jaccard ≥ 0.50 (vs 0.39), Brier ≤ 0.22 (vs 0.29). McNemar n_disc ≥ 25 attendu (vs 3) → puissance α=0.05 OK.

---

## Pourquoi NON Phase 3.6 retraining adversarial

Madry 2018 PGD adversarial training + Goodfellow 2015 Elo noise augmentation rendrait MLP **plus robuste à small noise local sur Elo features**, mais **ne lui apprend pas que A02 §3.7.b distribue déterministiquement les top Elo par équipe descendante**.

Conséquence : retraining ad-hoc pré-Phase 4a corrigerait partiellement la sensibilité numérique du MLP aux perturbations Elo (composante locale), mais maintiendrait le bug structurel sampling pool naïf (composante structurelle = cause profonde D-P3-19). Vrai gain limité.

Logique SOTA : corriger l'architecture (Phase 4a) **avant** d'investir effort retraining (Phase 3.6). Si après Phase 4a robustness toujours insuffisante (FAIL famille 1 résiduels post-D-P3-19), **alors** Phase 3.6 retraining adversarial justifiée. Pas avant.

Cohérent avec feedback `industry_standards` + `complete_or_nothing` : corriger la cause profonde une fois > patcher symptômes multiples.

---

## Aggregator implementation (ISO 5055 SRP)

### Avant cette session

`scripts/d8/aggregate.py::main()` ne wire que `load_saison_reports` (4 saisons × multi-division). `load_audit_reports` (ADR-019 phase-a 5 divisions) existe depuis 2026-05-10 mais **orphelin** (jamais branché dans `main()`).

### Après cette session

`scripts/d8/aggregate.py::main(argv: list[str] | None = None)` :
```python
args = _parse_args(argv)  # --mode {saison,phase-a} --input-dir --output-dir
if args.mode == "phase-a":
    reports = dict(load_audit_reports(input_dir))
else:
    reports = dict(load_saison_reports(input_dir))
# Cycle pipeline commun : verify_lineage → fuse → metrics → gates
if args.mode == "phase-a":
    full_report = _build_full_report_phase_a(...)
else:
    full_report = _build_full_report_saison(...)
```

`scripts/d8/types.py::D8FullReport` : 2 champs nouveaux backward-compat (`audit_mode = "saison"` default + `divisions = []` default). Tests 17 saison-mode existants restent verts.

`scripts/d8/aggregate.py::load_audit_reports` : 3 candidate layouts supportés (Kaggle auto-mount + local download + flat fallback). Cette session ajout candidate `input_dir / div_slug / f"d8_{saison}_{div_slug}.json"` pour layout local download `outputs/d8/2024/{div}/...`.

Tests : `tests/d8/test_aggregate_phase_a_mode.py` (NEW, 12 tests pure-function + integration). Suite aggregator complète = **29 PASS** (17 saison-mode + 12 phase-a-mode).

---

## Dettes ouvertes traceables (no silent debt)

Référence canonique : `memory/project_debt_current.md` (source of truth) + `docs/project/DEBT_LEDGER.md` v1.0.3 (versioned mirror).

| Dette ID | Description | Phase résolution | Status |
|----------|-------------|------------------|--------|
| **D-P3-19 / R-ALI-06** | ALI multi-équipes joint conditionnel CE-adverse miroir Approche A | **Phase 4a** | OPEN — **confirmation empirique ajoutée 2026-05-16** |
| D-2026-05-14-top16-v4-validation | Top 16 v4 output download + valid n_matches | cette session | **RÉSOLUE** (output downloaded, n=82, schema d8.v1) |
| D-2026-05-16-aggregator-phase-a-mode | `aggregate.py main()` n'avait pas `--mode=phase-a` (`load_audit_reports` orphelin) | cette session | **RÉSOLUE** (ce ADR-022) |
| D-2026-05-16-rob07-threshold-absolute | G_ROB_07 threshold absolu 3.0 sans context support_max=8 Phase A | **Phase 3.5b cleanup** | OPEN |
| D-2026-05-16-dro-aggregation-min-vs-percentile | G_ROB_08/09 kernel `dro.py` agrégation `min over per-match worst` → 1 outlier domine | **Phase 3.5b cleanup** | OPEN |
| D-2026-05-16-aggregator-fairness-uses-by-ronde | `_fairness_metrics` réimplémente proxy `_by_ronde` au lieu de consommer `r["breakdowns"]` (déjà calculés par division) | **Phase 3.5b cleanup** | OPEN (non-bloquant : conclusions D-P3-19 inchangées par strata Elo) |
| D-2026-05-16-lineage-code-sha-disparate-phase-a | N1-N4 v3 = CODE_SHA `11db85f` vs Top 16 v4 = `84d2f6d`. Fonctionnellement équivalents (ADR-021 isolé) mais lineage ISO 5259 §lineage non-cohérent | **Phase 5+ re-deploy uniform** | OPEN |
| D-2026-05-16-phase-3-6-adversarial-contingency | Phase 3.6 retraining (Madry 2018 PGD + augmentation Goodfellow 2015) **planifiée comme contingency uniquement** si robustness post-Phase 4a insuffisante | **Contingency post-Phase 4a re-run D8** | CONTINGENCY (pas pré-emptive, pas d'effort engagé tant que Phase 4a non-franchi) |

---

## Alternatives considérées

| Alternative | Effort | Gain estimé | Verdict | Justification |
|-------------|-------:|-------------|---------|---------------|
| **A. Phase 4a ALI conditionnel (RETENU)** | 3-5 sem | Résout 10/13 FAIL structurellement (D-P3-19). Robustness ranking-invariant by design. | ✓ **RETENU** | Résolution déjà planifiée, logique SOTA validée 2026-04-28, confirmation empirique cette session. |
| B. Phase 3.5b cleanup mineur seul | 1 session | 6/19 → 8-9/19 PASS (ROB_07 marginal + ROB_08/09 si percentile). Aucune amélioration modèle. | DIFFÉRÉ — Phase 3.5b post-Phase 4a (mineur, non-bloquant Phase 4a entry) | Ne résout pas cause profonde D-P3-19. À faire post-Phase 4a pour clean signal validation. |
| C. Phase 3.6 retraining adversarial pré-emptive | 2-3 sem | Amélioration partielle robustness MLP. NE TOUCHE PAS cause profonde D-P3-19. | REJETÉ comme première intention, PLANIFIÉ contingency | Patch symptomatique. Logique SOTA = corriger architecture ALI d'abord. |
| D. Accept-as-is + BLOQUER Phase 4a strict | 0 effort | 0 amélioration. Roadmap bloquée. | REJETÉ | Non-actionnable, ignore D-P3-19 déjà planifié. |
| E. Recalibration "aveugle" seuils SOTA | 0 effort | Faux PASS sans amélioration. | REJETÉ | Feedback `complete_or_nothing` + spec §5.3 "pas de relâchement aveugle". |

---

## Impact assessment ISO 42005

**R-ALI-06 update** (cf. `docs/iso/AI_RISK_REGISTER.md`) :
- **Évidence empirique 2026-05-16** : Top 16 stress Elo 1% noise → recall drop 33.5% (vs threshold 2%). Confirme R-ALI-06 manifestation prod = produit instable Top 16 (stakes max joueurs pros, prix tournois).
- **Mitigation prévue** : Phase 4a ALI joint conditionnel ranking-invariant. Re-mesure D8 Phase A post-Phase 4a attendue ≤ 5% drop @ 1% noise Top 16.
- **Severity** : élevée (blocant prod élite Top 16). Resté élevée jusqu'à Phase 4a delivery.

**R-ALI-01 quantification SOTA cross-niveau** :
- **Évidence empirique** : disparité ×20 Top 16 vs N4 stress Elo = signature cross-niveau D-P3-19. Quantification SOTA validée.
- **Mitigation prévue** : Phase 4a corrige par construction (sampling structurellement adapté par niveau).

---

## Cross-references

- **Spec D8 Phase A** : `docs/superpowers/specs/2026-04-30-d8-fairness-robustness-design.md` §5 (19 gates G-A SOTA)
- **Spec Phase 4a** : `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 4a Approche A
- **ADR-016** : `docs/architecture/adr/ADR-016-ali-conditioned-multi-team-adverse-ce-mirror.md` — Phase 4a Approche A decision
- **ADR-019/020/021** : Phase A multi-divisions + groupe filter + Top 16 rondes
- **D-P3-19 source of truth** : `memory/project_debt_current.md`
- **Acceptance report détaillé** : `reports/d8/phase_a/2026-05-16-acceptance.md`
- **R-ALI-06** : `docs/iso/AI_RISK_REGISTER.md`
- **Champion lineage** : `docs/iso/ALI_MODEL_CARD.md` §11 + `models/cache/mlp_champion_metadata.json`

---

## Décision user formelle

**Decision** : Phase 4a entry strategy retenue (Option A). Phase 3.5b cleanup mineur tracé (non-bloquant). Phase 3.6 retraining adversarial **NON pré-emptive**, **planifiée contingency** post-Phase 4a si robustness insuffisante.

**Validation user** : 2026-05-16 (cette session, pivot diagnostique post-REPRISE ADR-021).

---

**ADR créé** : 2026-05-16 par session post-REPRISE
**Trigger** : Phase A D8 audit aggregator 5/5 outputs complete → 13 FAIL empiriques → diagnostic révisé → user validation Option A.
