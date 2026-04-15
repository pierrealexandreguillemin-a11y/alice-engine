# Quality Gates — ALICE Engine ML Pipeline

> Chaque kernel et chaque etape du pipeline DOIT passer ses quality gates avant
> de passer a l'etape suivante. Pas de gate = pas de push.
>
> **Principe** : si on n'evalue pas selon des standards, on produit du garbage
> meme avec un pipeline ISO complet.

## References

| Ref | Source | Usage |
|-----|--------|-------|
| ISO 5259-2 | ISO/IEC 5259-2:2024 — Data Quality for ML (24 DQC, 65+ QM) | FE gates F1-F12 |
| Guo2017 | Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | T4, T5, T11 |
| Const2012 | Constantinou & Fenton, *Solving the Problem of Inadequate Scoring Rules*, JRSS 2012 | T2 (RPS) |
| Wheat2019 | Wheatcroft, *Evaluating probabilistic forecasts of football matches*, arXiv:1908.08980 | T12 (RPS vs log loss) |
| GoogleML | Google, *Rules of ML* (rules #9, #29, #32, #37) | T1, T10, S1, S5 |
| Deepchecks | Deepchecks Train-Test Validation Suite (defaults) | F4, F5, F9 |
| TFDV | TensorFlow Data Validation — schema + skew detection | F7, F12, S1, S2 |
| PSI | Fiddler AI + NannyML — Population Stability Index thresholds | F4, F5, S2 |
| Samb2021 | Sambasivan et al., *Data Cascades in High-Stakes AI*, CHI 2021 | Motivation |
| Epstein1969 | Epstein, *A Scoring System for Probability Forecasts*, J. Appl. Meteor. 1969 | T2 (RPS original) |

---

## Stage 1 — FE Kernel Outputs (features parquets)

Applique sur train.parquet, valid.parquet, test.parquet apres chaque run FE.

| Gate | Check | Metrique ISO 5259-2 | Seuil | Ref |
|------|-------|---------------------|-------|-----|
| **F1** | Pas de feature >99% NaN par split | Com-ML-1 (Value completeness) | 0 feature dead par split | ISO 5259-2 |
| **F2** | NaN% coherent cross-split | Com-ML-1 + Rep-ML-1 | Ecart NaN% train vs valid/test < 20pp par feature | ISO 5259-2 |
| **F3** | Pas de feature zero-variance | Eft-ML-1 (Effectiveness) | unique_count > 1 par feature par split | ISO 5259-2 |
| **F4** | Feature drift train->valid | Rep-ML-1 (Representativeness) | PSI < 0.2 par feature numerique | PSI, Deepchecks |
| **F5** | Feature drift train->test | Rep-ML-1 | PSI < 0.2 par feature numerique | PSI, Deepchecks |
| **F6** | Distribution target equilibree | Bal-ML-7 (Label balance) | 3 classes presentes dans chaque split, ratios +-5% vs global | ISO 5259-2 |
| **F7** | Schema consistency cross-split | Con-ML-3 (Format consistency) | Memes colonnes, memes dtypes | ISO 5259-2, TFDV |
| **F8** | Feature ranges valides | Acc-ML-6 (Accuracy range) | Valeurs dans plages domaine (elo in [0,3500], taux in [0,1], etc.) | ISO 5259-2 |
| **F9** | Pas de multicollinearite excessive | Feature correlation | Pairwise |r| < 0.95 (warn 0.9) | Deepchecks |
| **F10** | Row counts coherents | Com-ML-4 (Record completeness) | Splits dans ratios attendus (train ~79%, valid ~5%, test ~16%) | ISO 5259-2 |
| **F11** | Target valide | Com-ML-5 (Label completeness) | 0 NaN dans target, valeurs in {0, 1, 2} | ISO 5259-2 |
| **F12** | Pas de leakage temporel | Sim-ML-3 (Independency) | Aucune saison de test dans train | ISO 5259-2, TFDV |

### Notes

- **F2 est le gate qui aurait detecte le bug des 61 features mortes** (NaN 0% train, 100% valid/test).
- **F4/F5** : PSI se calcule sur 10-20 bins equi-quantile. Ajouter 0.01 aux bins vides.
  Seuils : < 0.1 OK, 0.1-0.2 warning, >= 0.2 FAIL.
- **F9** : VIF > 10 est le seuil academique, mais pour tree models la multicollinearite
  est moins critique que pour modeles lineaires. On warn a 0.9, fail a 0.95.

---

## Stage 2 — Training Kernel Outputs (modeles + predictions)

Applique sur les modeles entraines, predictions test/valid, et artefacts.

### Gates existants (valides)

| Gate | Check | Seuil | Ref |
|------|-------|-------|-----|
| **T1** | log_loss < Elo baseline | Strict (model < baseline) | GoogleML #9, Codecentric |
| **T2** | RPS < Elo baseline | Strict | Epstein1969, Const2012 |
| **T3** | E[score] MAE < Elo baseline | Strict | Domaine ALICE |
| **T4** | ECE < 0.05 (classwise) | 5% | Guo2017 |
| **T5** | Calibration P(draw) +-2% | Observe vs predit | Guo2017, domaine |
| **T6** | mean_p_draw > 1% | Sanity check | Domaine |
| **T7** | 0 NaN/Inf dans predictions | Strict | Sanity |
| **T8** | Probas sum to 1 | abs(sum-1) < 1e-6 par row | Sanity |
| **T9** | >5 features importance > 0 | Cross-modele (SHAP + gain) | Postmortem ALICE |

### Gates ajoutes (litterature)

| Gate | Check | Seuil | Ref |
|------|-------|-------|-----|
| **T10** | Train-test gap log_loss | < 0.05 (pas de surfit) | GoogleML #37 |
| **T11** | Reliability diagram par classe | Visuel : courbe proche diagonale | Guo2017, scikit-learn |
| **T12** | Reporter RPS ET log_loss | Les deux | Wheat2019 |

### Gates infrastructure — protection timeout (2026-04-15)

Ces gates ne verifient pas la qualite du MODELE mais la robustesse du PIPELINE.
Un kernel sans checkpoint qui timeout = 100% du travail perdu. Constate 5 fois
sur le projet ALICE (V8 v17, AG v1-v4). ISO 25010 (fiabilite), ISO 42001 (tracabilite).

| Gate | Check | Seuil | Ref |
|------|-------|-------|-----|
| **T13** | Checkpoint apres chaque modele/fold | 0 artefact perdu si timeout a tout moment | ALICE postmortem v17, kaggle-deployment skill |
| **T14** | Time guard avant post-processing | `time_left > budget_post` avant chaque etape | ALICE AG v4 timeout, SESSION_HARD_LIMIT |
| **T15** | Artefacts sauves incrementalement | Chaque save independant (pas batch en fin de script) | ALICE postmortem 3 timeouts consecutifs |
| **T16** | Budget temps calcule AVANT push | Docstring avec temps reel par composant | feedback_time_budget_kernels, feedback_calculate_dont_guess |

**T13 — Checkpoints par modele/fold :**
- Training Final : `_checkpoint_model()` apres chaque modele (deja implemente)
- OOF : `_save_oof_checkpoint()` apres chaque fold (deja implemente)
- AG : AG gere ses propres checkpoints internes (`ag_models/`)
- Meta-learner : checkpoints apres predictions, modele, metadata (deja implemente)
- **Verification** : grep `checkpoint` ou `save` dans le kernel. Si absent = FAIL.

**T14 — Time guard :**
```python
SESSION_HARD_LIMIT = 32400  # 9h GPU ou 43200 12h CPU
def _time_left(t0): return SESSION_HARD_LIMIT - (time.time() - t0)
# Avant chaque etape post-fit :
if _time_left(t0) < 300:  # 5 min minimum
    logger.warning("TIME GUARD: skipping remaining artifacts")
    return
```
- **Verification** : grep `time_left` ou `time_guard` dans le kernel. Si absent = FAIL.

**T15 — Sauvegarde incrementale :**
- MAUVAIS : `save_all()` en fin de script → timeout = 0 outputs
- BON : chaque artefact sauve IMMEDIATEMENT apres production
- **Verification** : chaque `save`/`to_parquet`/`json.dump` doit etre AVANT le prochain compute.

**T16 — Budget temps :**
- AVANT push : calculer temps par composant avec donnees empiriques (pas estimation)
- Documenter dans le docstring : "fit ~7h, post ~30min, total ~7h30 / 9h budget"
- Si total > 80% du budget session → restructurer AVANT de push
- **Verification** : docstring du kernel contient budget temps avec chiffres reels.

### Notes

- **T2 (RPS)** : score de reference en prediction sportive ordinale. RPS = 0 parfait.
  Benchmarks football : uniform ~0.22, bon ML ~0.205, meilleur ~0.192 (Constantinou 2012).
  Chess interclubs : pas de benchmark public, comparer vs Elo baseline uniquement.
- **T4 (ECE)** : utiliser classwise-ECE (pas confidence-ECE). 15 bins.
  Modeles non calibres typiques : ECE 4-10%. Apres temperature scaling : < 1-2%.
- **T10** : Google Rule #37 — mesurer 3 types de skew : train vs holdout, holdout vs
  next-period, next-period vs live. Si train-test gap > 0.05, le modele surtfit.
- **T12** : Wheatcroft 2019 argumente CONTRE le RPS seul — il peut preferer des
  previsions moins informatives. Log loss mesure le gain d'information en bits.
  Reporter les deux permet de detecter les cas ou RPS et log loss divergent.

---

## Stage 3 — Inference / Serving (API predict)

Applique a chaque appel ou en monitoring continu.

| Gate | Check | Seuil | Ref |
|------|-------|-------|-----|
| **S1** | Input schema match training | Memes features, memes dtypes | GoogleML #32, TFDV |
| **S2** | Feature drift vs training | PSI < 0.2 par feature | Vertex AI (default 0.3, conservatif 0.2) |
| **S3** | Output probas valides | sum=1, >=0, no NaN | Sanity |
| **S4** | Prediction distribution stable | mean P(draw) +-5% vs training | Evidently AI |
| **S5** | Log features at serving | Pour detection skew futur | GoogleML #29 |

---

## Versioning et Tracabilite (ISO 5259 + 42001)

Chaque run de kernel DOIT etre trace :

| Element | Outil | Ref |
|---------|-------|-----|
| Code source | git commit hash | ISO 42001 |
| Dataset (parquets) | DVC hash + git tag | ISO 5259 lineage |
| Features (outputs FE) | DVC hash | ISO 5259 |
| Modeles (outputs training) | DVC hash + metadata.json | ISO 42001 model card |
| Kernel version Kaggle | Log dans metadata.json | Tracabilite |

**Pas de DVC = pas de reproductibilite. C'est un TODO bloquant pour Phase C.**

---

## Findings recherche (2026-03-29)

### Multicollinearite et tree models
- Tree models **tolerent** la multicollinearite pour la prediction (Hooker et al. 2021)
- VIF n'est **pas pertinent** pour GBT — concu pour modeles lineaires (Xu et al. 2014 KDD)
- SHAP est **corrompu** par la multicollinearite (Aas et al. 2021, Lundberg issue #1120)
- **Action** : F9 = tester pairwise top-20 SHAP, grouper si >0.9, ne pas supprimer

### Init score alpha (prior Elo)
- Alpha est un **hyperparametre legitime** (coefficient de confiance dans le prior)
- F_0 (init_score) n'est PAS soumis au learning rate (Friedman 2001) — asymetrie architecturale
- Alpha et LR sont **complementaires** : alpha = confiance prior, LR = capacite apprentissage
- Justification theorique : Ash & Adams 2020 (NeurIPS) — shrink initialization
- **Pas de methode built-in** dans XGBoost/CatBoost/LightGBM pour scaler init_score

### Strategie v17/v18
- v17 : split fix + alpha=0.7 + hyperparams inchanges (isoler l'effet data)
- Diagnostic : best_iter indique si les features apportent du signal (>300 = oui)
- v18 : ajuster alpha ou LR selon diagnostic

---

## Application pratique

### Avant push FE kernel
```bash
# Verifier F1-F12 localement si possible (subset des donnees)
# Sinon : valider sur outputs Kaggle AVANT de push le training kernel
python scripts/validate_fe_outputs.py --gates F1-F12
```

### Avant push Training kernel
```bash
# Le training kernel integre les gates T1-T12 dans quality_gate()
# Verifier que TOUS les gates sont dans le code AVANT push
grep -c "gate" scripts/cloud/train_kaggle.py
```

### En production
```bash
# S1-S5 dans le middleware FastAPI
# Monitoring continu avec alertes
```
