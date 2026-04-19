# ADR-013 : RuleEngine JSON-driven remplace les règles Python FFE

**Date** : 2026-04-19
**Status** : ACCEPTED
**Context** : Phase 3 brainstorming 2026-04-18/19

## Contexte

Phase 2 a introduit 11 règles FFE en Python (`services/ffe_rules.py`).
Parallèlement, chess-app/flat-six/rules/ contient 46 règles FFE structurées
en JSON (format canonique avec UUID RFC4122, source_ref PDF, article, conditions
formalisées). pocket-arbiter parse ces PDFs via docling.

Les 11 règles Python = **réimplémentation duplicate** des 14 A02 (dont 10 seulement
couvertes). Risque de drift entre les deux sources.

## Décision

**Remplacer** `services/ffe_rules.py` par un **RuleEngine générique** qui interprète
les JSON chess-app vendorés dans `config/ffe_rules/`.

ALICE classe PUBLIC/PRIVATE chaque règle dans `alice_verifiability.json`
(annotations locales, indépendantes de chess-app).

## Conséquences

**Positives :**
- Source de vérité unique (JSON chess-app, normatif)
- Traçabilité ISO 42001/5259 via UUID + source_ref
- Extensible sans nouveau code (ajouter un JSON = nouvelle compétition)
- Cohérent avec l'écosystème (pocket-arbiter → chess-app JSON → ALICE)

**Négatives :**
- ~250 lignes nouvelles (RuleEngine générique)
- Dette D10 : sync manuelle JSON chess-app → ALICE (traitée dans ce plan)

## Alternatives rejetées

- **Statu quo** : garder les 11 Python → drift fatal
- **Migrer une à une** : conserve double source → risque divergent
- **Vendor sans RuleEngine** : nécessite réimplémentation par règle → pas DRY

## Implémentation

- Plan 1 : RuleEngine + vendoring + classifier + CI drift alert
- Plan 2 : wire dans /compose, suppression `services/ffe_rules.py`
