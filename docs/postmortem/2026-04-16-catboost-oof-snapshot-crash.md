# Postmortem: CatBoost OOF Snapshot Crash

**Date**: 2026-04-16
**Severity**: HIGH — 2/5 folds perdus, kernel à refaire
**Responsible**: Claude (design + push)

## Incident

3 kernels CatBoost OOF (folds [0,1], [2,3], [4]) pushés le 2026-04-15.
Folds 0, 2, 4 réussis. Folds 1, 3 CRASH avec:

```
CatBoostError: Can't load progress from snapshot file: /kaggle/working/catboost_snapshot
Current learn and test datasets differ from the datasets used for snapshot
LearnAndTestQuantizedFeaturesCheckSum = 1109504675  (fold 0 data)
LearnAndTestQuantizedFeaturesCheckSum = 3211481865  (fold 1 data)
```

## Root Cause

Le paramètre `snapshot_file` de CatBoost persiste entre folds. Quand fold N termine,
le fichier snapshot reste sur disque. Fold N+1 a un split train/val DIFFÉRENT →
checksum CatBoost différent → crash immédiat.

## Ce qui aurait dû être fait

Supprimer `catboost_snapshot` entre chaque fold :
```python
if os.path.exists(snapshot_path):
    os.remove(snapshot_path)
```

## Circonstance aggravante

Le risque avait été **identifié avant push** par Claude, puis **minimisé** avec "ça passera
sans problème, les snapshots sont saved à chaque étape." C'est une violation directe de :
- CLAUDE.md §Sincérité : "NE JAMAIS minimiser un gap pour avancer"
- `feedback_verify_before_push.md` : process 9 étapes

Le bug était prévisible à 100% — la doc CatBoost exige les MÊMES données pour un snapshot,
et chaque fold a des données différentes par définition.

## Impact

- 2/5 folds perdus (folds 1 et 3)
- OOF predictions inutilisables (logloss = 28.96)
- 3 folds converges (val ~0.641) confirment que le modèle est cohérent avec V9 Final
- Temps perdu : ~7h de compute Kaggle CPU × 2 kernels à refaire

## Fix

1 ligne : supprimer le snapshot file avant chaque fold dans `train_oof_stack.py`

## Leçon

Quand un risque est identifié, il doit être CORRIGÉ, pas minimisé.
"Ça passera" n'est pas une analyse technique, c'est de la paresse.
