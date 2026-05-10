# Pre-V1 Production Audit — Template ALICE Engine

**Document ID** : ALICE-PROCESS-PRE-V1-AUDIT
**Version** : 1.0.0
**Owner** : ML Senior Engineer (Claude) + User (final decision)
**Status** : NON NÉGOCIABLE — exécuté avant TOUT passage en prod (v1, vN+1, deploy SaaS, public release)

---

## Pourquoi cet audit existe

Le rôle d'ingénieur ML senior dans ALICE exige de **challenger ses propres choix** avant production. Sans gate formelle, le biais SOTA / le biais de cohérence pousse à shipper l'état actuel sans confronter gain/effort des alternatives non explorées. Cet audit force la confrontation systématique sur 8 dimensions.

Ce template est **ALICE-specific** : les sections 1-3 référencent le champion MLP(32,16) 18f, le CE OR-Tools multi-équipes, les contraintes FFE A02, les 14 normes ISO. Une réutilisation hors-Alice est secondaire.

Trigger formel : avant **chaque passage en prod**, créer
`reports/pre_v1_audit/<YYYY-MM-DD>-<scope>.md` en remplissant ce template intégralement.

---

## Section 1 — ML Model self-challenge

**Question maître** : *Le champion actuel est-il optimal-wireable sur (data, features, base ensemble) ?*

### Sous-questions obligatoires

- [ ] **Quel est le champion actuel ?** Citer modèle + lineage SHA-256 + métriques canonical (log_loss, ECE_draw, draw_bias, RPS, E[score] MAE).
- [ ] **Quelles alternatives non explorées avec ROI > 0.005 log_loss ou +0.5pt ECE ?** Liste exhaustive (Deep Ensemble, Bayesian MLP, TTA, MoE, conformal wrapping, …).
- [ ] **Pour chaque alternative, 2+ sources publiées** (arxiv, conférence, journal — pas blogposts).
- [ ] **Test empirique recommandé** : SI gain > 0.005 log_loss attendu, refit + comparer OOF AVANT swap.

### Output Section 1 — tableau ROI

| # | Alternative | Source | Gain attendu (log_loss / ECE) | Effort (CPU + dev) | Risque (regression) | ROI | Decision |
|---|---|---|---|---|---|---|---|
| | | | | | | | ship-as-is / pre-V1 / post-V1 / never |

---

## Section 2 — Feature / Data self-challenge

**Question maître** : *Tous les signaux disponibles dans data/echiquiers.parquet + data/joueurs.parquet sont-ils exploités ?*

### Sous-questions obligatoires

- [ ] **Features deferred** : revue dette ouverte D13 (zone_enjeu), F4 (streak), F5 (LHS/antithetic), F6 (copule), D11 (NLP PDF FFE).
- [ ] **Drift FFE** : schema joueurs/echiquiers a-t-il évolué depuis dernier scrape ? Vérifier sync `chess-app/scripts/sync_ffe_rules.py`.
- [ ] **Coverage temporal** : data va jusqu'à quelle saison ? Manque-t-il du current ?
- [ ] **Feature importance audit** : SHAP sur champion → top-20 features expliquent X% prédiction. Reste-t-il signal résiduel non capté ?

### Output Section 2

- Liste features deferred prioritaires (par SHAP-attendu)
- Dette FFE schema drift (open / closed)
- Decision : feature engineering avant V1 ou post-V1

---

## Section 3 — CE / Optimization self-challenge

**Question maître** : *Le CE multi-équipes (OR-Tools) encode-t-il toutes les contraintes FFE A02 §3.7 ?*

### Sous-questions obligatoires

- [ ] **Contraintes FFE A02** : ordre Elo descendant (§3.7.b), joueur brûlé (§3.7.c), même groupe (§3.7.d), noyau (§3.7.f) — encode-t-il les 4 ?
- [ ] **Coupes FFE** : J02 jeunes / S65 vétérans / Loubatière supportés ?
- [ ] **Sensitivity analysis** : varier les poids objective E[score] pondération win/draw/loss → décision change-t-elle de manière catastrophique sur edge cases ?
- [ ] **Adversarial inputs** : pool minimal (just team_size), pool huge (200+), Elo identique pour tous, club étranger inconnu, ronde 0/inexistante.
- [ ] **Compatibilité ALI Phase 4a** : joint conditionné multi-équipes intégré au CE ?

### Output Section 3

- Edge cases découverts non couverts → list bugs ou clarifications spec
- Sensitivity report (objective weights)
- Decision : ship CE current ou Phase 4a-required ?

---

## Section 4 — Performance / UX self-challenge

**Question maître** : *L'API FastAPI tient-elle un SLA réaliste sous charge ?*

### Sous-questions obligatoires

- [ ] **Latency P95/P99** : `/compose` + `/recompose` mesurés sous load (10/50/100 req/s) ? Cible : <2s P99 (CE on-demand spec).
- [ ] **Cold start** : durée boot complet (cache + rules + MLP load) ? Cible : <30s.
- [ ] **Memory footprint** : RSS au runtime sur Oracle VM ARM 24GB ? Marge ?
- [ ] **Error handling user-visible** : 4xx vs 5xx propres ? Messages métier (ronde invalide, club inconnu) vs stack traces ?
- [ ] **Rate limiting** (`slowapi`) : quotas réalistes pour user solo vs SaaS multi-tenant ?

### Output Section 4

- SLA report (mesures actuelles vs cible)
- Liste UX defects (messages erreurs cryptiques, etc.)
- Decision : optimisations pre-V1 vs accepter SLA actuel

---

## Section 5 — Security / Compliance ISO

**Question maître** : *Les 14 normes ISO sont-elles toutes auditées un dernier coup avant prod ?*

### Sous-questions obligatoires (par norme)

- [ ] **ISO 5055** : `wc -l` ≤300, `radon cc` ≤B, `radon mi` A — full repo scan ?
- [ ] **ISO 27001** : secrets en env vars uniquement (gitleaks pre-push) ? Audit logs MongoDB / structlog ?
- [ ] **ISO 27034** : pydantic validation TOUS endpoints API ? Bandit scan clean ?
- [ ] **ISO 25010** : tests + benchmarks couvrent fiabilité/perf/sécurité ?
- [ ] **ISO 29119** : pytest coverage >70% global ? Tests fast (pre-push <90s) + slow (CI) splittés ?
- [ ] **ISO 42010** : ADRs documentent toutes les décisions architecturales (à jour) ?
- [ ] **ISO 15289** : MkDocs build --strict OK ? Documentation cycle de vie complète ?
- [ ] **ISO 42001** : Model Card complet (sections G1-G11) ? Lineage SHA-256 wire end-to-end ?
- [ ] **ISO 42005** : Impact assessment AI à jour ? (`docs/iso/AI_RISK_ASSESSMENT.md`)
- [ ] **ISO 23894** : Risk register à jour ? Mitigations validées ?
- [ ] **ISO 5259** : Pandera schemas + lineage compute_data_lineage() ?
- [ ] **ISO 25059** : 15 quality gates F1-F12/T1-T12 PASS ?
- [ ] **ISO 24029** : Robustness D8 — gates G_ROB PASS ?
- [ ] **ISO 24027** : Fairness D8 — gates G_FAIR PASS ?

### Output Section 5

- Tableau 14 normes : status PASS / FAIL / GAP / DEFERRED
- Liste gaps avec phase cible

---

## Section 6 — Risk (ISO 23894) self-challenge

**Question maître** : *AI_RISK_REGISTER à jour ? Mitigations empiriquement validées ?*

### Sous-questions obligatoires

- [ ] **R-ALI-01** PRIVATE rules unverifiable — D8 G_FAIR_05/06 quantifie l'impact ?
- [ ] **R-ALI-02** Pool too small — D8 by_pool_size validé à N=280 ?
- [ ] **R-ALI-04** Drift FFE rules + roster turnover — D8 stress_roster + PSI ?
- [ ] **R-ALI-06** ALI multi-équipes joint conditionné — Phase 4a status ?
- [ ] **Nouveaux risques** identifiés depuis dernier audit ?

### Output Section 6

- Risk register up-to-date status
- Mitigations validées empiriquement (yes/no/pending)
- Risques résiduels que user accepte explicitement avant prod

---

## Section 7 — Cost / Effort matrix (synthèse)

Compile findings sections 1-6 dans **UN seul tableau de décision** :

| # | Amélioration | Section | Gain mesurable | Effort (CPU + dev jours) | Risque regression | ROI | Senior reco | User decision |
|---|---|---|---|---|---|---|---|---|
| | | | | | | HIGH/MED/LOW | pre-V1 / post-V1 / never | OK / postpone / reject |

**Règle ROI** :
- HIGH : gain > 0.5σ benchmark + effort < 1 sem + risque LOW
- MED : gain ~ 0.2-0.5σ OU effort 1-3 sem
- LOW : gain < 0.2σ OU effort > 3 sem OU risque HIGH

---

## Section 8 — Senior recommendation finale + User decision

### Recommandation Senior (Claude)

> Texte libre, 200-400 mots. Doit contenir :
> - **Avis ML senior synthétique** : ship-as-is OR adopter N améliorations avec quelles deadlines
> - **Raisons techniques** (pas un rationnel marketing)
> - **Risques résiduels** acceptés
> - **Hypothèses** qui pourraient invalider la reco

### User decision (à remplir)

- [ ] **Accept Senior reco** sans modification
- [ ] **Override** : ship as-is malgré reco contraire (raison documentée : ____)
- [ ] **Adopt subset** des améliorations recommandées : items # ____
- [ ] **Reject totalement** : reporter V1 (raison : ____)

**Date décision** : _____
**Signed-off** : User + Claude

---

## Annexes

### A.1 Sources documentaires obligatoires à consulter avant audit

- `config/MODEL_SPECS.md` (ML champion specs)
- `docs/architecture/DECISIONS.md` (ADR-001..N)
- `docs/iso/AI_RISK_REGISTER.md`
- `docs/iso/IMPLEMENTATION_STATUS.md`
- `memory/project_debt_current.md` (dettes ouvertes)
- `docs/requirements/QUALITY_GATES.md` (F1-F12 / T1-T12)
- `docs/superpowers/specs/2026-03-23-alice-prod-roadmap-design.md` (roadmap)

### A.2 Commandes utiles

```bash
# ISO 5055 scan
.venv/Scripts/python.exe -m radon cc scripts/ services/ app/ -a -s
.venv/Scripts/python.exe -m radon mi scripts/ services/ app/ -s

# ISO 25059 quality gates
make test-cov

# Latency benchmark
.venv/Scripts/python.exe scripts/benchmarks/api_latency.py  # à créer si absent

# SHAP top-20 features champion
.venv/Scripts/python.exe scripts/explain/shap_top_features.py  # à créer si absent
```

### A.3 Délivrable attendu

1. `reports/pre_v1_audit/<YYYY-MM-DD>-<scope>.md` (ce template rempli intégralement)
2. Mise à jour `memory/project_debt_current.md` avec dettes nouvelles identifiées
3. ADR proposé pour chaque amélioration adoptée pre-V1
4. Commit signé `audit(pre-v1): <scope> — N items pre-V1, M items post-V1`

---

**END OF TEMPLATE**
