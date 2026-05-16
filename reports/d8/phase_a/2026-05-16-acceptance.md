# D8 Phase A — Acceptance Report (verdict + cause attribution + phase de résolution)

**Document ID** : ALICE-D8-PHASE-A-ACCEPTANCE-2026-05-16
**Version** : 1.0.0
**Date** : 2026-05-16
**Scope** : ADR-019 Phase A — saison 2024 × 5 divisions (Top 16 + Nationale 1-4)
**ISO Compliance** : ISO/IEC 24029-2 (robustness), 24027 (fairness), 42001 (lifecycle),
42005 (impact), 25059 (quality), 23894 (risk), 5259 (data lineage), 5055 (architecture).

---

## 1. Verdict empirique

| Indicateur | Valeur |
|------------|-------:|
| Gates G-A G_FAIR_01..10 + G_ROB_01..09 | **6/19 PASS, 13 FAIL, 0 INCONCLUSIVE** |
| N matches cumulé Phase A | **492** (Top 16 82 + N1 56 + N2 101 + N3 168 + N4 85) |
| Floor spec §1.3 | 200 — ✓ dépassé |
| Champion artefact SHA-256 (MLP+temp_scaler) | Identique cross-divisions ✓ (champion invariant Plan 3 T22.0) |
| Lineage CODE_SHA | Delta accepté : N1-N4 = `11db85f` (ADR-020), Top 16 = `84d2f6d` (ADR-020+ADR-021) — ADR-021 isolé Top 16 rondes_default, pas d'impact fonctionnel N1-N4 |
| Aggregator mode | `phase-a` (ADR-019), patch cette session — voir §6 |

**Phase 4a entry gate** : **BLOCKED** par 11/13 FAIL = confirmation empirique quantifiée
D-P3-19 / R-ALI-06 (déjà identifiée T22 post-mortem 2026-04-28, désormais quantifiée).

---

## 2. Gates breakdown (19 G-A)

### 2.1 PASS (6)

| Gate | Mesuré | Threshold | Source |
|------|-------:|----------:|--------|
| G_FAIR_03 demographic_parity_diff | 0.0000 | ≤ 0.10 | Hardt 2016 |
| G_FAIR_04 equalized_odds_diff | 0.0000 | ≤ 0.10 | Hardt 2016 §3.2 |
| G_FAIR_07 TPR_ratio_min | 1.0000 | ≥ 0.80 | EEOC §1607.4D + Feldman 2015 |
| G_FAIR_09 BSS_per_group | 0.3942 | ≥ 0.30 | Pappalardo 2019 §3.4 |
| G_FAIR_10 PSI_per_dim | 0.0000 | ≤ 0.20 | Yurdakul 2020 |
| G_ROB_06 conformal_coverage_90 | 0.9486 | ≥ 0.90 | Vovk 2024 §2.3 |

### 2.2 FAIL (13) — cause attribution + phase de résolution

| Gate | Mesuré | Threshold | Δ | Famille | Cause | Résolution prévue |
|------|-------:|----------:|---:|---------|-------|-------------------|
| G_FAIR_01 max_gap_recall | 0.2258 | ≤ 0.10 | +0.126 | **Signal réel** | Cross-niveau gap : Top 16 recall 0.683 vs N4 0.246 (×2.8). ALI Phase 3 sample sans contrainte FFE A02 §3.7.b multi-équipes → over-confiance top Elo dans divisions secondaires. | **Phase 4a** (D-P3-19) |
| G_FAIR_02 recall_per_group_min | 0.5159 | ≥ 0.85 | -0.334 | **Signal réel** | Recall absolu faible par groupe (e.g. Top 16 Q1 <1500 n=39 recall=0.644). Conditionnement multi-équipes manquant amplifie l'erreur pour pools >30 joueurs. | **Phase 4a** (D-P3-19) |
| G_FAIR_05 calibration_ECE_per_group | 0.4958 | ≤ 0.05 | +0.446 | **Signal réel** | ECE_ali = écart prédit/observé sur P(joueur aligné). Mesuré 0.47 uniformément cross-strata Elo (Q1=0.468 n=39, Q2=0.467 n=38, Q3=0.429 n=5). Calibration ALI mauvaise par construction : sample pool club total ignore A02 §3.7.b → sur-confiance top Elo. **⚠ Distinct de ECE_draw=0.0016 champion (calibration P(D) per-board, très bonne).** | **Phase 4a** (D-P3-19) |
| G_FAIR_06 multicalibration_alpha | 0.4958 | ≤ 0.05 | +0.446 | **Signal réel** | Hébert-Johnson 2018 α = max ECE intra-groupe. Identique FAIR_05 (proxy aggregator). | **Phase 4a** (D-P3-19) |
| G_FAIR_08 brier_per_group | 0.4605 | ≤ 0.30 | +0.161 | **Signal réel** | Brier_ali sur P(joueur aligné) élevé toutes strata (Q1=0.437 n=39, Q2=0.439). Sur-confiance ALI prédictions binaires. | **Phase 4a** (D-P3-19) |
| G_ROB_01 recall_drop_1pct | 0.1186 | ≤ 0.02 | +0.099 | **Signal réel** | Stress Elo 1% bruit → recall drop 11.9% global. Top 16 amplifié à 33.5% (1% noise Elo 2500 = ±25 pts flippe rankings adjacents → A02 §3.7.b redistribue pool effectif → ALI predict change). N4 = 1.7% (écarts Elo 80+ pts insensibles à ±15 pts). Mécanisme structurel D-P3-19. | **Phase 4a** (D-P3-19) |
| G_ROB_02 recall_drop_5pct | 0.1390 | ≤ 0.05 | +0.089 | **Signal réel** | Madry 2018 strict ε=0.05. Idem ROB_01 amplifié. Top 16 = 36.6%, N4 = 3.3%. | **Phase 4a** (D-P3-19) |
| G_ROB_03 recall_drop_10pct | 0.1105 | ≤ 0.10 | +0.011 | **Signal réel** | Marginal mais FAIL. Saturation effect : à 10% bruit l'amplification Top 16 ralentit (déjà fortement perturbé à 5%). | **Phase 4a** (D-P3-19) |
| G_ROB_04 roster_5pct | 0.1443 | ≤ 0.05 | +0.094 | **Signal réel** | Tran 2022 §3.4 turnover 5% roster → recall drop 14.4%. Top 16 = 35.3%, N4 = 5.4%. Joueurs élite ont identités spécifiques (peu remplaçables au top niveau). | **Phase 4a** (D-P3-19) |
| G_ROB_05 roster_20pct | 0.1996 | ≤ 0.15 | +0.050 | **Signal réel** | Recht 2019 §5 turnover 20% → drop 20.0%. Idem ROB_04 amplifié. | **Phase 4a** (D-P3-19) |
| G_ROB_07 conformal_set_size_max | 6.0429 | ≤ 3.0 | +3.043 | **Threshold absolu vs support** | Set size ALI 6.04 sur support_max=8 boards Phase A (ADR-020) = 75% efficiency relative. Threshold absolu 3.0 hérité spec D8 sans context support (Angelopoulos 2023 §4.2 définit efficiency en relatif). Même avec threshold ratio 0.50 (= 4.0 absolu pour support 8), measure 6.04 reste FAIL marginal → corollaire calibration ALI mauvaise (cf. FAIR_05/06). | (a) Cleanup mineur : passer à threshold ratio relatif `set_size_mean/support_max ≤ 0.50` — **Phase 3.5b cleanup** ; (b) Cause sous-jacente résolue **Phase 4a** (D-P3-19). |
| G_ROB_08 DRO_eps_005_min | 0.0000 | ≥ 0.70 | -0.700 | **Design-inadapté** | Kernel `dro.py` calcule `recall_worst_case = min over per-match worst-perturbation`. 1 match catastrophique (recall=0) domine min absolu. Sinha 2018 §4 Wasserstein worst-case prévu pour distribution shift continu, pas pour outlier match dominant à granularité agrégée. | (a) **Phase 3.5b cleanup mineur** : changer agrégation kernel `min` → `percentile(5%)` ou `mean over worst-k`. (b) Cause sous-jacente : ALI sensitive aux outliers → également **Phase 4a** (D-P3-19). |
| G_ROB_09 DRO_eps_010_min | 0.0000 | ≥ 0.55 | -0.550 | **Design-inadapté** | Idem ROB_08 à ε=0.10 (Duchi 2021 §6). | Idem ROB_08. |

### 2.3 Synthèse cause → phase

| Famille | Gates | Cause | Phase résolution |
|---------|-------|-------|------------------|
| **Signal réel modèle (10 gates)** | FAIR_01/02/05/06/08 + ROB_01-05 | **D-P3-19 / R-ALI-06** : ALI Phase 3 ignore conditionnement FFE A02 §3.7.b multi-équipes. Empiriquement confirmé par disparité ×20 Top 16 vs N4 stress + ECE_ali uniforme cross-strata. | **Phase 4a Approche A SOTA** (T22 post-mortem 2026-04-28, validée user). ALI conditionné par CE-adverse miroir OR-Tools simulant l'allocation adverse sous contraintes A02 §3.7.b/c/d/f. Ranking-based donc invariant à perturbations Elo ε% (sauf joueurs adjacents). |
| **Threshold inadapté (1 gate)** | ROB_07 | Threshold absolu 3.0 hérité spec D8 sans context support_max=8 Phase A. | **Phase 3.5b cleanup** : threshold relatif `set_size_mean/support_max ≤ 0.50` (Angelopoulos 2023 §4.2 efficiency relatif). Note : cause sous-jacente (set_size élevé) reste **Phase 4a**. |
| **Design-inadapté agrégation (2 gates)** | ROB_08/09 | Kernel `scripts/d8/dro.py` agrégation worst-case = `min over per-match`. 1 outlier match domine. | **Phase 3.5b cleanup** : changer agrégation `min` → `percentile(5%)`. Cause sous-jacente (ALI outlier sensitivity) reste **Phase 4a**. |

**11/13 FAIL → Phase 4a (D-P3-19)** — résolution déjà planifiée, logique SOTA validée 2026-04-28.
**3/13 FAIL** ont en plus une composante "Phase 3.5b cleanup" mineur (ROB_07 threshold + ROB_08/09 agrégation).

---

## 3. Empirical evidence per division

### 3.1 Stress Elo recall_drop

| Division | Baseline recall | @ 1% noise | @ 5% noise | @ 10% noise | % perte relative @ 1% |
|----------|----------------:|-----------:|-----------:|------------:|----------------------:|
| Top 16   | 0.683 | **0.335** | **0.366** | n/a | **49.0%** |
| N1       | 0.567 | 0.087 | 0.142 | n/a | 15.4% |
| N2       | 0.496 | 0.083 | 0.063 | n/a | 16.8% |
| N3       | 0.460 | 0.071 | 0.092 | n/a | 15.4% |
| N4       | 0.246 | 0.017 | 0.033 | n/a | 6.9% |

### 3.2 Stress roster turnover

| Division | @ 5% turnover | @ 10% turnover | @ 20% turnover |
|----------|--------------:|---------------:|---------------:|
| Top 16   | **0.353** | 0.384 | **0.420** |
| N1       | 0.121 | 0.092 | 0.133 |
| N2       | 0.092 | 0.117 | 0.142 |
| N3       | 0.102 | 0.109 | 0.237 |
| N4       | 0.054 | 0.039 | 0.067 |

### 3.3 Per-strata Elo breakdown (exemple Top 16)

| Strata Elo | n | recall_mean | ECE_mean | Brier_mean | BSS_mean |
|-----------|--:|------------:|---------:|-----------:|---------:|
| Q1 <1500 | 39 | 0.644 | **0.468** | 0.437 | 0.444 |
| Q2 1500-1700 | 38 | 0.622 | **0.467** | 0.439 | 0.438 |
| Q3 1700-1900 | 5 | 0.625 | 0.429 | 0.402 | 0.489 |

**Lecture** : ECE_ali=0.47 UNIFORME sur strata avec n=39 et n=38 (statistiquement solide) → calibration ALI mauvaise par construction, pas un artifact d'échantillonnage. Distinct de ECE_draw=0.0016 champion mesuré sur P(D) per-board (différente métrique, différent étage du pipeline).

---

## 4. Mécanisme structurel D-P3-19 confirmé empiriquement

**Hypothèse posée 2026-04-28 (T22 post-mortem)** :
> 117 clubs alignent 2-4 équipes simultanément en N3 ronde 5 saison 2024. FFE A02 §3.7.b force les top Elo en équipe supérieure. ALI Phase 3 sample dans pool total ⇒ sur-représentation top Elo en N3 ⇒ recall structurellement faible.

**Confirmation empirique 2026-05-16** :
1. **Cross-niveau gap×20 stress Elo** : disparité Top 16 (33.5% drop @ 1%) vs N4 (1.7%). Signature du mécanisme :
   - Top 16 = écarts Elo 5-15 pts entre joueurs élite 2500-2700 → 1% noise (±25 pts) flippe rankings adjacents → A02 §3.7.b redistribue alloc équipes → pool effectif Top 16 change → ALI predict flippe.
   - N4 = écarts Elo 80+ pts entre joueurs 1200-2200 → 1% noise (±15 pts) ne flippe rien → ALI predict stable.
2. **ECE_ali uniforme 0.47** : modèle prédit P(joueur aligné) en sample naïf pool club total, alors qu'empiriquement joueur top Elo va systématiquement en équipe sup (A02 §3.7.b déterministe). Calibration impossible sans conditionnement.
3. **Roster turnover Top 16 amplifié 7×** : Top 16 35.3% @ 5% turnover vs N4 5.4%. Joueurs élite ont identités Elo spécifiques (peu de substituts à même niveau), N4 a "pool de substituts" plus large.

Mécanisme structurel D-P3-19 **quantifié empiriquement et SOTA-cohérent** avec :
- Mehrabi 2021 §4.1 (max_gap recall fairness) → 0.226 vs 0.10
- Pleiss 2017 §4 (calibration ECE per-group) → 0.50 vs 0.05
- Goodfellow 2015 (recall drop @ ε noise) → 0.119 vs 0.020
- Tran 2022 §3.4 (roster turnover robustness) → 0.144 vs 0.050

---

## 5. Implication produit (contexte sportif)

ALICE Engine sert à recommander composition USER × N équipes club sous contraintes FFE A02 §3.7. Avec ALI Phase 3 actuel :
- Entre deux rondes consécutives, Elo joueurs bouge ±5-15 pts par match joué (réalité FFE).
- → ALI predict adverse-composition différente.
- → CE optimise composition USER différemment.
- → User reçoit cette semaine "alignez X" et semaine prochaine "alignez Y" sans changement de fond.
- **Top 16 = stakes max (joueurs pros, prix tournois)** = exactement où produit doit être le plus stable → instabilité actuelle = blocant prod (R-ALI-01 SOTA cross-niveau quantification).

Phase 4a Approche A SOTA résout structurellement :
- Sample compositions adverses sous contraintes A02 §3.7.b/c/d/f via CE-adverse miroir OR-Tools.
- Ranking-based → invariant à small noise Elo (sauf joueurs adjacents, qui sont eux-mêmes une dimension d'incertitude légitime à modéliser).
- Sklar 1959 copule gaussienne + Monte Carlo conditionnel inter-équipes.

---

## 6. Aggregator patch cette session (production-ready)

Le aggregator Phase A nécessitait un patch pour supporter le mode ADR-019 (1 saison × 5 divisions). Implémentation cette session :

- `scripts/d8/aggregate.py` : nouveau `_parse_args()` argparse + flag `--mode {saison,phase-a}` + branche `load_audit_reports` (ADR-019 layout) + `_build_full_report_phase_a()` qui collapse lineage_per_saison à canonical entry (MLP+temp SHA invariant).
- `scripts/d8/aggregate.py::load_audit_reports` : 3 layouts supportés (Kaggle auto-mount + local download + flat fallback).
- `scripts/d8/types.py::D8FullReport` : 2 nouveaux champs `audit_mode: str = "saison"` (default backward-compat) + `divisions: list[str]` (phase-a populated).
- `tests/d8/test_aggregate_phase_a_mode.py` : 12 tests nouveau mode (load + verify_lineage + fuse + build_full_report + parse_args). 17 tests saison-mode existants restent verts.

ISO 5055 SRP préservé (single entry-point avec mode-discriminator), ISO 42010 cohérence architecture (pas de fork code path), ISO 27001 §A.14.2.5 changement contrôlé via tests + ADR-022.

Commande CLI :
```bash
PYTHONPATH=. .venv/Scripts/python scripts/d8/aggregate.py --mode phase-a \
  --input-dir outputs/d8/2024 --output-dir reports/d8/phase_a
```

---

## 7. Decision matrix Phase 4a entry

| Option | Description | Effort | Gain | Verdict |
|--------|-------------|-------:|------|---------|
| A. Avancer Phase 4a (RECO) | Implémenter ALI joint conditionnel CE-adverse miroir Approche A SOTA. Re-run D8 Phase A après Phase 4a pour validation. | 3-5 sem | Résout 10/13 FAIL structurellement (D-P3-19). Robustness ranking-invariant by design. | **RETENU** — résolution déjà planifiée, logique SOTA validée 2026-04-28, confirmation empirique cette session. |
| B. Phase 3.5b cleanup mineur | Fix threshold ROB_07 ratio relatif + DRO ROB_08/09 agrégation `min` → `percentile(5%)`. | 1 session | Passerait 6/19 → 8-9/19 PASS (ROB_07 marginal + ROB_08/09 si percentile). Aucune amélioration modèle. | **DIFFÉRÉ Phase 3.5b** (mineur, non-bloquant Phase 4a entry). |
| C. Phase 3.6 retraining adversarial | Madry 2018 PGD + Goodfellow 2015 Elo augmentation pendant fit champion. | 2-3 sem | Amélioration partielle robustness MLP. NE TOUCHE PAS la cause profonde (manque conditionnement multi-équipes A02 §3.7.b). | **REJETÉ ici** : patch symptomatique. Logique SOTA = corriger architecture ALI d'abord (Phase 4a) ; si robustness toujours insuffisante post-Phase 4a, ALORS Phase 3.6. |

**Décision retenue** : Option A. Implémenter Phase 4a, re-run D8 Phase A après. Si robustness post-Phase 4a toujours insuffisante (FAIL famille 1 résiduels) → Phase 3.6 adversarial training **planifiée comme contingency**, pas pré-emptive.

---

## 8. Dettes ouvertes traceables (no silent debt)

| Dette ID | Description | Phase résolution | Status |
|----------|-------------|------------------|--------|
| D-P3-19 / R-ALI-06 | ALI multi-équipes joint conditionnel (D-P3-19 existing 2026-04-28, **confirmation empirique 2026-05-16 cette session**) | **Phase 4a** | OPEN — confirmation empirique ajoutée |
| D-2026-05-14-top16-v4-validation | Top 16 v4 output download + valide n_matches | cette session | **RÉSOLUE** (output downloaded, n=82, schema d8.v1 valide) |
| D-2026-05-16-aggregator-phase-a-mode | Aggregator main n'avait pas le wire `--mode=phase-a` (load_audit_reports orphelin) | cette session | **RÉSOLUE** (commit ADR-022 ce jour) |
| D-2026-05-16-rob07-threshold-absolute | G_ROB_07 conformal_set_size threshold absolu 3.0 sans context support. Angelopoulos 2023 §4.2 définit efficiency relatif. | **Phase 3.5b cleanup** | OPEN |
| D-2026-05-16-dro-aggregation-min-vs-percentile | G_ROB_08/09 kernel `dro.py` agrégation `min over per-match worst` → 1 outlier domine. Sinha 2018 §4 prévu pour distribution shift continu. | **Phase 3.5b cleanup** | OPEN |
| D-2026-05-16-aggregator-fairness-uses-by-ronde | `_fairness_metrics` réimplémente proxy `_by_ronde` au lieu de consommer `r["breakdowns"]` (déjà calculés par division avec n statistique). Aggregator perd info structurée + dégrade signal. | **Phase 3.5b cleanup** | OPEN (non-bloquant : conclusions D-P3-19 inchangées car ECE_ali confirmé uniforme par strata Elo) |
| D-2026-05-16-lineage-code-sha-disparate-phase-a | N1-N4 v3 = CODE_SHA `11db85f` vs Top 16 v4 = `84d2f6d`. Fonctionnellement équivalents (ADR-021 isolé Top 16 rondes_default), mais lineage ISO 5259 §lineage non-cohérent. | **Phase 5+ (re-deploy uniform)** ou closure design-decision (ADR-021 isolé) | OPEN |

---

## 9. ISO compliance summary

| Norme | Coverage | Note |
|-------|----------|------|
| ISO 24029-2 (robustness) | ✓ mesuré stress Elo + roster + DRO Wasserstein | 7 gates FAIL = signal réel à corriger Phase 4a |
| ISO 24027 (fairness) | ✓ mesuré per-strata Elo + per-niveau + per-genre | 5 gates FAIL = signal réel D-P3-19 |
| ISO 42001 (lifecycle) | ✓ ADR-022 traçabilité décision + lineage SHA-256 invariant cross-divisions | RAS |
| ISO 42005 (impact) | ✓ implication produit §5 (stabilité prod Top 16 = stakes max) | R-ALI-01 / R-ALI-06 mis à jour avec confirmation empirique |
| ISO 5259 (data lineage) | ⚠ disparate CODE_SHA cross-divisions Phase A | D-2026-05-16-lineage-code-sha-disparate-phase-a OPEN |
| ISO 25059 (quality) | ✓ 19 gates G-A measured + cause attribution + phase resolution | RAS |
| ISO 23894 (risk) | ✓ R-ALI-06 mise à jour empirique | RAS |
| ISO 5055 (architecture) | ✓ aggregator patch ISO 5055 SRP préservé | 29 tests aggregator PASS |

---

## 10. Cross-references

- **ADR-019** `docs/architecture/adr/ADR-019-d8-phase-a-multi-divisions.md` — Phase A spec
- **ADR-020** `docs/architecture/adr/ADR-020-d8-groupe-filter-and-conformal-support-fix.md` — Groupe + conformal support
- **ADR-021** `docs/architecture/adr/ADR-021-d8-rondes-default-division-specific.md` — Top 16 rondes
- **ADR-022** `docs/architecture/adr/ADR-022-d8-phase-a-acceptance-verdict-ali-conditional-phase-4a.md` — **ce verdict**
- **D-P3-19** `memory/project_debt_current.md` — ALI joint conditionnel (mise à jour empirique 2026-05-16)
- **R-ALI-01 / R-ALI-06** `docs/iso/AI_RISK_REGISTER.md` — quantification SOTA cross-niveau
- **Spec D8** `docs/superpowers/specs/2026-04-30-d8-fairness-robustness-design.md` §5 (19 gates G-A SOTA)
- **Spec Phase 4a** `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` §Phase 4a Approche A
- **Champion lineage** `docs/iso/ALI_MODEL_CARD.md` §11 + `models/cache/mlp_champion_metadata.json`

---

## 11. Decision user formelle

**Decision** : **Phase 4a entry strategy retained** — Option A (avancer Phase 4a ALI joint conditionnel Approche A SOTA). 11/13 FAIL résolus structurellement par Phase 4a (D-P3-19). 3 dettes mineures Phase 3.5b cleanup tracées explicitement (D-2026-05-16-rob07/dro/aggregator-fairness). Phase 3.6 retraining adversarial **NON RETENU** comme première intention ; **planifié comme contingency** si robustness post-Phase 4a insuffisante.

**Validation** : user 2026-05-16 (cette session, pivot diagnostique post REPRISE).

---

**Generated** : 2026-05-16 par session post-REPRISE ADR-021 + coverage restoration
**Aggregator run** : `outputs/d8/2024/{top-16,nationale-1,2,3,4}/d8_2024_{div}.json` → `reports/d8/phase_a/d8_full_report.json` + `D8_FINDINGS.md` + `D8_FAILURE_ANALYSIS_LOG.md` + `gates_19_status.json`
