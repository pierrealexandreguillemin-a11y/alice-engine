# ADR-015 : `services/ffe_rules.py` maintenu Plan 2, suppression Plan 3

**Date** : 2026-04-19
**Status** : ACCEPTED
**Context** : Plan 2 Phase 3 Task 9 audit + peer review Plan 2

## Contexte

Le plan Plan 2 Task 9 spécifie :
> Supprimer `services/ffe_rules.py`. Vérifier `app/api/routes.py` n'importe plus de l'ancien fichier. Tests Phase 2 existants doivent toujours passer (refactor compatible).

Le plan suppose implicitement que `RuleEngine` (livré Plan 1) couvre l'intégralité de la logique FFE. Audit du code `app/api/routes.py` révèle :

- `_apply_pre_filters` (lignes 69-73) utilise `filter_brule`, `filter_match_count`, `filter_same_group`
- `_validate_composition` (lignes 140-154) utilise 7 `check_*` fonctions : `check_elo_order`, `check_team_size`, `check_unique_assignment`, `check_noyau`, `check_mutes_limit`, `check_foreign_quota`, `check_fr_gender`, `check_elo_max`
- Ligne 343 : `sort_by_elo`

**État actuel de `RuleEngine`** : implémente 4 articles (`_check_team_size`, `_check_elo_order`, `_check_mutes_limit`, `_check_elo_max`). **8 articles/helpers manquants** pour migrer `routes.py` sans perte de fonctionnalité :
- `filter_brule` (3.7.c)
- `filter_match_count` (3.7.e)
- `filter_same_group` (3.7.d)
- `check_noyau` (3.7.f)
- `check_unique_assignment` (trivial)
- `check_foreign_quota` (3.7.h)
- `check_fr_gender` (3.7.i)
- `sort_by_elo` (helper, trivial)

## Décision

**`services/ffe_rules.py` est maintenu tel quel dans Plan 2.** Suppression différée à Plan 3.

Plan 3 couvrira :
1. Extension `RuleEngine` avec `_check_*` / `_filter_*` pour les 7 articles FFE manquants (brule, match_count, same_group, noyau, unique, foreign_quota, fr_gender)
2. Migration `routes.py::_apply_pre_filters` vers `RuleEngine.filter_candidates`
3. Migration `routes.py::_validate_composition` vers `RuleEngine.validate_lineup`
4. Suppression effective de `services/ffe_rules.py`
5. Tests Phase 2 backward-compat + Plan 2 ALI doivent continuer à passer

## Conséquences

**Positives :**
- Plan 2 scope reste focalisé sur ALI SOTA (pas d'expansion rétroactive)
- Cohérence ISO 42010 : déviation tracée explicitement via ADR plutôt que dette silencieuse
- Migration Plan 3 peut être planifiée proprement (estimation ~150 lignes ajoutées dans RuleEngine + refactor ciblé `routes.py`)

**Négatives :**
- `services/ffe_rules.py` coexiste avec `RuleEngine` pendant la durée de Plan 2 → double source de vérité temporaire
- Risque de drift si nouvelle règle FFE ajoutée uniquement à un des deux endroits (mitigation : ADR-013 et ce ADR signalent la transition en cours)
- Le script `verify_plan2_dod.sh` ne vérifie PAS l'absence de `services/ffe_rules.py` (gate structural retiré)

## Alternatives rejetées

### Option A : extension `RuleEngine` complète dans Plan 2
**Rejetée.** Ajouter 7 nouveaux `_check_*` méthodes + refactoriser `routes.py` en Plan 2 Task 9 étendrait le scope de ~150-200 lignes + tests. Cela :
- Retarde la livraison Plan 2 (ALI generator SOTA prêt mais bloqué sur task cleanup)
- Mélange la mission Plan 2 (ALI) avec un nettoyage Phase 2 legacy (mission distincte)
- Augmente le risque de régression sur Phase 2 backward-compat (11/11 tests actuels)

### Option B : inline fonctions dans `routes.py`
**Rejetée.** Déplacer la logique de `ffe_rules.py` dans `routes.py` violerait ISO 5055 (`routes.py` exploserait au-delà de 300 lignes) et serait une migration cosmétique (juste relocation) au lieu d'une consolidation vers RuleEngine.

## Implémentation Plan 3

- **P3-Task 1** (nouvelle) : extension `RuleEngine` avec 7 articles FFE manquants, tests UUID-level par article
- **P3-Task 2** (nouvelle) : migration `routes.py::_apply_pre_filters` + `_validate_composition` vers `RuleEngine`
- **P3-Task 3** (nouvelle) : suppression `services/ffe_rules.py` + `tests/test_ffe_rules.py`, vérification tests Phase 2 + Plan 2
- **P3-Task 4** (nouvelle) : mise à jour `verify_plan2_dod.sh` → `verify_plan3_dod.sh` avec gate `! test -f services/ffe_rules.py`

## Références

- Plan 2 Task 9 : `docs/superpowers/plans/2026-04-19-phase3-plan2-generator-sota.md` §Task 9
- Peer review Plan 2 finding important #1 (2026-04-19)
- ADR-013 : RuleEngine JSON-driven replaces Python FFE rules
- ISO 42010 : architecture decisions documented
