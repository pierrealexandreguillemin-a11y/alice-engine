# Kaggle Cloud Architecture

## Constraints
- CPU illimite (12h/session, pas de quota). GPU = 30h/semaine P100+T4.
- Tous kernels training = CPU (`enable_gpu: false`). Tree models n'utilisent pas GPU.
- Datasets : `/kaggle/input/datasets/{user}/{slug}/`
- kernel_sources : `/kaggle/input/notebooks/{user}/{slug}/`
- Env var `ALICE_MODEL` dans entry point (PAS `KAGGLE_KERNEL_RUN_SLUG`).
- Entry points DOIVENT setup sys.path AVANT import.
- Secrets impossibles en batch push (Kaggle API issue #582).
- Toujours re-uploader alice-code AVANT push si fichiers modifies.

## V8 Architecture 4-kernel (ADR-003)
FE kernel → XGBoost canary → CatBoost + LightGBM en parallele.
n_estimators=50K, early_stopping=200. CPU uniquement.

## V9 Architecture Optuna (2026-04-07)
3 kernels Optuna (1 par modele) → 3 kernels Training Final → 3 kernels OOF.
SQLite storage pour resume. Pruning callbacks (`optuna_integration` v4+).
Search spaces audites : `config/hyperparameters.yaml` section optuna.
init_score_alpha in [0.3, 0.8] joint search space.

## AutoGluon — ÉLIMINÉ (ADR-011, 2026-04-16)
Pas de residual learning, calibration incompatible CE, test 0.5716 > V9 LGB 0.5619.
NE PAS relancer de kernel AG. Voir `docs/architecture/DECISIONS.md` §ADR-011.

## OOF Architecture (2026-04-16)
1 fold = 1 kernel. CatBoost snapshot_file incompatible avec multi-fold sequentiel
(checksum mismatch entre folds). Postmortem : `docs/postmortem/2026-04-16-catboost-oof-snapshot-crash.md`.

## Checkpoints par librairie
- CatBoost : snapshot_file (binaire, secondes). MEMES params requis.
- LightGBM : callback text (lent: 65K=3h22m startup). From scratch si >3h.
- XGBoost : TrainingCheckPoint (binaire). EarlyStopping(save_best=True) OBLIGATOIRE.
