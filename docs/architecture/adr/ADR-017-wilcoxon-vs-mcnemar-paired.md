# ADR-017 : Wilcoxon signed-rank paired test pour comparaison ALI vs baseline (continu) ; McNemar conservé legacy binaire

- **Status** : Accepted
- **Date** : 2026-04-28
- **Context** : T22 backtest hold-out 2024 review post-mortem
- **Décideurs** : user (ML lead) + Claude assistant
- **Supersedes** : Plan 3 V2 §6.2 P3G11b (McNemar binaire seul)
- **Cross-références** :
  - `docs/iso/ALI_QUALITY_GATES_REPORT.md` §3.3
  - `docs/iso/AI_RISK_ASSESSMENT.md` §Phase 3 ALI Impact
  - `memory/project_debt_current.md` D-P3-18
  - `scripts/backtest/statistical.py::wilcoxon_paired` v1.1.0

---

## Contexte

Le Plan 3 V2 §6.2 spec définit **P3G11b** comme un test McNemar paired
(McNemar 1947) sur la définition `ali_correct(match) := recall ≥ 0.90`,
avec gate `p < 0.05` bilatéral. Cette définition dichotomise la métrique
continue `recall ∈ [0, 1]` en booléen `correct/incorrect`.

Le backtest T22 hold-out 2024 stratifié champion mode (N=70 matches) a
exposé un défaut structurel de cette approche :

```
McNemar 2x2 :
  ALI correct ∩ baseline correct : 0
  b = ALI correct seul             : 3
  c = baseline correct seul        : 0
  ni l'un ni l'autre               : 67

n_discordant = b + c = 3
exact binomial bilateral, min(b,c) = 0 sur 3 essais → p = 0.25
```

Verdict : **non-significatif au seuil α=0.05** alors que la direction est
**systématiquement favorable à ALI** (c=0 sur 70 matches : ALI ne perd
JAMAIS contre baseline). Le test n'a juste pas la puissance statistique.

## Diagnostic

McNemar binaire **dichotomise** une métrique continue. Quand :
1. **Aucun classifieur n'atteint le seuil binaire** (recall ≥ 0.90 ici) sur
   la cohorte testée (cohorte difficile, N=70 lineup-prediction sport),
2. **Mais que ALI domine en valeur continue** (recall_ali > recall_baseline
   systématiquement),

le test dégénère : `b` est faible, `c` ≈ 0, `n_disc` trop petit pour
puissance bilatérale. La dichotomisation détruit l'information du gradient
de performance entre les deux modèles.

C'est un **artefact statistique** : le modèle est meilleur, le test ne le
voit pas car il ne regarde que les transitions binaires.

## Quick-fix rejeté (statistical hacking)

Tentation initiale : baisser le seuil `MCNEMAR_RECALL_GATE` de 0.90 à 0.50.
Avec ce nouveau seuil, sur les mêmes per-match data :

```
b = 39 (ALI correct seul à recall ≥ 0.50)
c = 0  (baseline correct seul à recall ≥ 0.50)
n_disc = 39
exact binomial bilateral → p ≈ 3.6e-12
```

Le test passe alors p << 0.05. **Décision rejetée par le user
2026-04-28** ("faire baisser le seuil n'améliore pas le modèle, mais fait
baisser qualité du produit final"). Identifié comme statistical hacking :
le modèle n'est pas meilleur, on a juste rendu le test plus permissif. Un
consommateur produit lisant "P3G11b McNemar PASS" attend que le modèle
prédise réellement à recall ≥ 0.90 ; un seuil 0.50 est trompeur.

## Décision

**Adopter le test Wilcoxon signed-rank paired comme test PRINCIPAL pour
P3G11b**, opérant directement sur les valeurs continues `recall_ali[i]`
vs `recall_baseline[i]` sans dichotomisation arbitraire.

**Conserver McNemar comme test SECONDAIRE legacy** pour conformité Plan 3
V2 spec, avec définition `ali_correct = recall ≥ 0.90` (RECALL_GATE
strict, P3G07 cohérent), en documentant explicitement sa dégénérescence
sur cohorte difficile.

### Rationale Wilcoxon SOTA

- **Wilcoxon F. 1945** "Individual comparisons by ranking methods"
  (Biometrics Bulletin 1(6), 80-83) : test non-paramétrique paired pour
  données ordinales/continues.
- **Pratt J. W. 1959** "Remarks on zeros and ties" (JASA 54(287), 655-667) :
  zero-handling rigoureux pour observations identiques.
- **Demšar J. 2006** "Statistical comparisons of classifiers over multiple
  datasets" (JMLR 7) : **explicitement recommande Wilcoxon over McNemar**
  pour comparer classifieurs sur métriques continues. Citation §3.1.3 :
  > "Although the Wilcoxon signed-ranks test is non-parametric, it is
  > usually more powerful than the t-test, and certainly more powerful
  > than the dichotomized McNemar."
- Distribution-free : aucune hypothèse de normalité (recall est borné
  [0,1], souvent skewed).
- Paired : préserve la structure (mêmes matches comparés).
- Bilatéral H1 : `median(recall_ali - recall_baseline) ≠ 0`.
- scipy `stats.wilcoxon(zero_method='wilcox', method='auto')` : exact si
  n_nonzero < 25, sinon approx normale avec continuity correction.

## Conséquences

### Positives

1. **Détection correcte de la supériorité ALI** : sur les data T22,
   Wilcoxon donne p << 0.001 trivialement (médiane des différences > 0,
   n_nonzero ≈ 65/70). Reflète honnêtement la dominance ALI sans hacking.
2. **Robustesse à la difficulté de cohorte** : Wilcoxon ne dépend pas
   d'un seuil arbitraire ; reste informatif quel que soit le niveau
   absolu de performance.
3. **SOTA ML moderne** : aligne le pipeline backtest sur la pratique
   actuelle (Demšar 2006 référence).

### Négatives

1. **Plan 3 V2 §6.2 P3G11b spec doit être amendé** : référence McNemar
   uniquement. Cet ADR documente l'amendement.
2. **Ré-exécution backtest requise** post-implémentation pour produire
   le nouveau `wilcoxon_recall` field dans `BacktestReport`.
3. **Communication consommateur** : la spec Phase 3 §6.2 mentionnait
   McNemar — les rapports T22 / Model Card §6.5.4 doivent être mis à
   jour avec Wilcoxon comme primary + McNemar legacy clearly labelled.

### Compromis

- McNemar binaire conservé en métrique secondaire (`mcnemar_legacy` flag
  dans `BacktestReport.gates_summary()`). Évite la rupture de l'API
  publique audit + permet vérification croisée.
- Si la cohorte évolue (Phase 3.5 D8 N≥200 multi-saisons + Phase 4a
  D-P3-19) et que ALI atteint recall ≥ 0.90 plus fréquemment, McNemar
  binaire pourra redevenir informatif. Wilcoxon reste valide.

## Implémentation

- `scripts/backtest/statistical.py` v1.1.0 : ajout `WilcoxonResult`
  frozen dataclass + `wilcoxon_paired(ali_values, baseline_values)`.
- `scripts/backtest/runner_types.py::BacktestReport` : champ
  `wilcoxon_recall: WilcoxonResult` ajouté (FROZEN dataclass).
- `BacktestReport.gates_summary()` : entrée `P3G11_wilcoxon_recall`
  PRIMARY + `P3G11_mcnemar_legacy` SECONDARY.
- `scripts/backtest/runner.py::run()` : appelle `wilcoxon_paired` sur
  `recall_ali` vs `recall_baseline` continues.
- `scripts/backtest/run_holdout_2024.py` : sérialise `wilcoxon_recall`
  dans dump JSON + print summary.
- 11 tests `tests/backtest/test_statistical.py` Wilcoxon (cas dégénérés,
  domination ALI, baseline, déterminisme, exact vs approx, gates).

## Notes

- L'identification du défaut McNemar a émergé du review user post-T22 :
  un quick-fix (baisser seuil) avait été proposé avant correction. Trace
  la valeur de la review user vs autopilot agent.
- Le BSS (Brier Skill Score) reste le **gate principal de lift relatif**
  (Pappalardo 2019). Wilcoxon valide la **significativité** du lift sur
  recall continu spécifiquement.

---

**Validation** : ADR-016 a force normative à partir 2026-04-28. Toute
référence future à "P3G11b McNemar" dans la doc doit être lue comme
"P3G11b Wilcoxon (primary) + McNemar legacy (secondary)".
