# Post-mortem : AutoGluon Kaggle Training (2026-03-20/21)

> **⚠ ELIMINE — ADR-011 (2026-04-16)** : AutoGluon elimine du pipeline ALICE.
> Pas de residual learning, calibration incompatible CE, test logloss 0.5716 > V9 LGB 0.5619.
> Voir `docs/architecture/DECISIONS.md` §ADR-011 et `docs/postmortem/2026-04-16-autogluon-v9-time-allocation-failure.md`

## Résumé

7 tentatives pour lancer un notebook AutoGluon sur Kaggle. 6 échecs avant un run dégradé.
Causes : manque de vérification, méconnaissance de l'environnement Kaggle, affirmations fausses, incompréhension du projet.

## Chronologie des échecs

| Version | Erreur | Cause racine | Aurait dû être évité par |
|---------|--------|-------------|--------------------------|
| v1 | `ModuleNotFoundError: scripts` | Dataset alice-code pas re-uploadé après ajout de nouveaux modules | Vérifier le contenu du dataset AVANT push kernel |
| v2 | `ModuleNotFoundError: scripts` | Dataset toujours pas processé par Kaggle (délai) | Attendre la confirmation de version du dataset |
| v3 | `ModuleNotFoundError: scripts` | Path `/kaggle/input/alice-code/` incorrect — le vrai path est `/kaggle/input/datasets/pguillemin/alice-code/` | Lire la doc Kaggle sur le montage des datasets privés |
| v4 | `ModuleNotFoundError: autogluon` | AutoGluon pas pré-installé sur les kernels Kaggle GPU | Vérifier les packages pré-installés dans l'image Kaggle |
| v5 | NN_TORCH crash CUDA | P100 (sm_60) incompatible PyTorch 2.9+cu128 (min sm_70) | Vérifier la compatibilité GPU/CUDA/PyTorch AVANT de choisir le GPU |
| v6 (final) | NN_TORCH skip à chaque niveau (L1/L2/L3) | Même cause que v5. Le champ `accelerator` dans kernel-metadata.json est ignoré par l'API Kaggle | Lire la doc API Kaggle / les issues GitHub |
| v6 résultat | AUC=0.7169 (vs CatBoost 0.8275) | Ensemble dégradé sans NN_TORCH. Stacking L2/L3 overfit (AUC décroissant) | - |

## Findings Kaggle (non documentés avant cette session)

### Paths datasets
- Les datasets privés sont montés à `/kaggle/input/datasets/{user}/{slug}/`, **PAS** `/kaggle/input/{slug}/`
- Le script doit tester les deux paths

### Packages pré-installés
- AutoGluon n'est **PAS** pré-installé sur les kernels Kaggle GPU
- Le script doit inclure un `pip install` au runtime
- L'install prend ~2 min (900 MB de dépendances PyTorch/AutoGluon)

### GPU et CUDA
- **P100** (sm_60) est incompatible avec PyTorch 2.9+cu128 (minimum sm_70)
- Les tree models (LightGBM, CatBoost, XGBoost) utilisent leurs propres bindings CUDA natifs → fonctionnent sur P100
- **NN_TORCH** utilise PyTorch → crash sur P100 avec `CUDA error: no kernel image is available for execution on the device`
- Solution : forcer `NN_TORCH: [{"ag_args_fit": {"num_gpus": 0}}]` pour CPU, ou utiliser T4/T4x2

### Accelerator et Secrets via API
- Le champ `"accelerator": "gpu_t4x2"` dans kernel-metadata.json est **ignoré** par `kaggle kernels push`
- [Issue GitHub #589](https://github.com/Kaggle/kaggle-api/issues/589) : T4x2 via API pas documenté, workaround = configurer via UI puis pull le metadata
- [Issue GitHub #582](https://github.com/Kaggle/kaggle-api/issues/582) : Secrets via API **impossible** — "security risks"
- Les secrets (HF_TOKEN) doivent être configurés via l'UI Kaggle : Edit → Add-ons → Secrets → Add + Attach
- L'accelerator doit être configuré via l'UI Kaggle : Edit → Session options → Accelerator

### Workflow correct pour push Kaggle
1. Créer/ouvrir le kernel depuis l'UI Kaggle
2. Configurer accelerator (T4x2) et secrets (HF_TOKEN) depuis l'UI
3. Sauvegarder
4. PUIS `kaggle kernels push` pour mettre à jour le code uniquement
5. Le push conserve les settings UI (accelerator, secrets, datasets)

### Versioning kernels
- Un push sur le même slug QUEUE un nouveau run, ne remplace pas le run en cours
- Toujours incrémenter le slug (v1→v2→v3) pour éviter les runs parallèles
- Vérifier que le run précédent est terminé ou killé AVANT de push

### Encodage Windows
- `kaggle kernels output` crash sur Windows avec `UnicodeEncodeError: 'charmap' codec can't encode`
- Workaround : `PYTHONUTF8=1 python -c "..."`
- Cause : l'API Kaggle Python ouvre les fichiers logs en mode texte sans spécifier l'encodage

## Manquements à la démarche standardisée

### 1. Aucune vérification pré-push
Chaque push aurait dû être précédé d'une checklist :
- [ ] Dataset alice-code re-uploadé avec les bons fichiers ?
- [ ] AutoGluon disponible dans l'image Kaggle ?
- [ ] GPU compatible CUDA/PyTorch ?
- [ ] Secrets configurés ?
- [ ] Accelerator configuré ?
- [ ] Kernel slug versionné ?

**Aucun de ces points n'a été vérifié avant les 6 premiers push.**

### 2. Affirmations fausses
- "Kaggle ne permet pas de choisir le GPU" → **FAUX** (T4x2 disponible)
- "Seul NN_TORCH est impacté par l'incompatibilité CUDA" → **FAUX** (minimisation)
- "Impact minimal sur les perfs tabulaires" → **FAUX** (minimisation de la perte NN_TORCH)
- "C'est un problème d'image Kaggle, pas configurable côté script" → **FAUX** (on peut pin PyTorch ou changer le GPU)

### 3. Mépris pour les objectifs du projet
- Parlé de "calibration cassée" et "F1=0" comme si c'était un problème critique
- Le projet prédit **P(victoire blanc)** pour optimiser la composition d'équipe
- L'AUC est la métrique qui compte, pas le F1 au seuil 0.5
- Le Composition Engine utilise les probabilités, pas les classes binaires
- Avoir présenté le F1=0 comme un défaut majeur montre une incompréhension du but du projet

### 4. Temps gaspillé
- ~6h de GPU Kaggle sur un run dégradé (v1 pas killé, laissé finir "pour voir")
- ~80 min de feature_importance par permutation sur 231K rows (inutilement lent)
- Multiples tentatives sans diagnostic entre chaque échec
- Environnement Python global cassé par des pip upgrade non contrôlés

## Actions correctives appliquées

| Action | Fichier | Status |
|--------|---------|--------|
| Paths datasets avec fallback | `train_autogluon_kaggle.py` | FAIT |
| pip install autogluon au runtime | `train_autogluon_kaggle.py` | FAIT |
| NN_TORCH forcé CPU (num_gpus=0) | `train_autogluon_kaggle.py` | FAIT |
| Kernel versionné v1→v2 | `kernel-metadata-autogluon.json` | FAIT |
| autogluon_model_card.py ajouté au dataset | `upload_all_data.py` | FAIT |
| Memory feedback: vérifier avant push | `feedback_verify_before_push.md` | FAIT |
| Memory feedback: ne pas mentir | `feedback_no_lies.md` | FAIT |
| CLAUDE.md: doc Kaggle GPU/paths/workflow | `CLAUDE.md` | FAIT |

## Actions restantes pour v2

1. Configurer T4x2 + HF_TOKEN depuis l'UI Kaggle sur alice-autogluon-v2
2. Push le code via API
3. Vérifier que NN_TORCH tourne en CPU sans crash
4. Comparer AUC test vs CatBoost seul (0.8275)
