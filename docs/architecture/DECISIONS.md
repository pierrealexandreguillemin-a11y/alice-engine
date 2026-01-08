# Architecture Decision Records (ADR)

> **Norme**: ISO 42010 - Architecture Decisions
> **Format**: MADR (Markdown Any Decision Records)

---

## ADR-001: CatBoost vs XGBoost pour le modele ML

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix du framework ML pour la prediction des compositions adverses.

### Decision
Utiliser **CatBoost** comme framework principal.

### Raisons
1. Gestion native des features categoriques (licence, division, ligue_code)
2. Inference 30-60x plus rapide que XGBoost
3. Moins de tuning requis (bon out-of-the-box)
4. Meilleure gestion des valeurs manquantes

### Consequences
- XGBoost garde en fallback
- Format modele: `.cbm` (CatBoost Model)
- Dependance: `catboost>=1.2.7`

---

## ADR-002: FastAPI vs Flask pour l'API

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix du framework web Python.

### Decision
Utiliser **FastAPI**.

### Raisons
1. Validation automatique avec Pydantic v2
2. Documentation OpenAPI auto-generee
3. Support async natif (Motor MongoDB)
4. Performance superieure a Flask

### Consequences
- Schemas Pydantic obligatoires
- Typage strict
- Dependance: `fastapi>=0.109.0`

---

## ADR-003: OR-Tools pour l'optimisation

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix du solver pour le Composition Engine (CE).

### Decision
Utiliser **Google OR-Tools**.

### Raisons
1. Solver CP-SAT performant
2. Adapte aux problemes d'assignment
3. Gratuit et open-source
4. Meilleur que PuLP pour notre cas

### Consequences
- Contraintes FFE modelisables
- Dependance: `ortools>=9.8`

---

## ADR-004: Architecture 3 couches SRP

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Organisation du code Python.

### Decision
Architecture en 3 couches: Controller → Service → Repository.

### Raisons
1. Separation des responsabilites (ISO 42010)
2. Testabilite (services purs)
3. Coherence avec chess-app backend

### Consequences
- `app/api/` = Controllers
- `services/` = Logique metier
- `services/data_loader.py` = Repository

---

## ADR-005: Render vs Vercel pour le deploiement

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Choix de la plateforme de deploiement (budget 0€).

### Decision
Utiliser **Render** (free tier).

### Raisons
1. Support Python natif (pas de serverless)
2. Long-running processes supportes
3. Blueprint YAML pour IaC
4. Region Frankfurt (proche France)

### Consequences
- Cold start apres 15 min inactivite
- Solution keep-alive requise
- 750h/mois partagees

### Alternatives evaluees
- Vercel: Timeout 10s, pas adapte ML
- Koyeb: 1 service gratuit seulement
- Railway: Plus de free tier

---

## ADR-006: MongoDB partage avec chess-app

**Date**: 3 Janvier 2026
**Statut**: Accepte

### Contexte
Acces aux donnees joueurs/clubs.

### Decision
Reutiliser le cluster MongoDB Atlas de chess-app.

### Raisons
1. Donnees deja presentes (joueurs, clubs)
2. Pas de duplication
3. Cout 0€
4. Lecture seule pour ALICE

### Consequences
- Meme connexion string
- Pas de collections propres a ALICE
- Dependance a chess-app pour les donnees

---

## ADR-007: Layered + SRP vs Domain-Driven Design (DDD)

**Date**: 8 Janvier 2026
**Statut**: Accepte

### Contexte

Choix du paradigme architectural pour structurer le code d'Alice-Engine.
Deux approches principales considerees:

1. **Layered Architecture + SRP** (actuel)
   - Controller → Service → Repository
   - Single Responsibility Principle par couche
   - Simple, explicite

2. **Domain-Driven Design (DDD)**
   - Bounded Contexts, Aggregates, Entities, Value Objects
   - Ubiquitous Language
   - Architecture hexagonale/onion

### Decision

Conserver **Layered Architecture + SRP** et ne pas adopter DDD.

### Raisons

#### 1. Complexite du domaine insuffisante pour DDD

| Critere | Alice-Engine | Seuil DDD |
|---------|--------------|-----------|
| Bounded Contexts | 1 (Composition) | 3+ |
| Regles metier | ~10 regles FFE | 50+ |
| Workflows | Lineaire (predict→optimize) | Multiples, branches |
| Equipe | 1-2 devs | 5+ devs |

#### 2. DDD serait over-engineering

Le domaine Alice-Engine est **algorithmique**, pas **metier complexe**:
- ALI: Inference ML (CatBoost) → calcul statistique
- CE: Optimisation (OR-Tools) → probleme mathematique
- Regles FFE: ~10 regles, pas de processus metier

DDD brille pour: banque, assurance, e-commerce, ERP.
DDD est excessif pour: API ML, microservices simples, CRUD.

#### 3. Cout cognitif non justifie

| Element DDD | Cout | Benefice Alice |
|-------------|------|----------------|
| Bounded Contexts | Eleve | Nul (1 seul contexte) |
| Aggregates | Moyen | Faible |
| Domain Events | Eleve | Nul (pas d'evenements) |
| Ubiquitous Language | Moyen | Deja present (termes FFE) |
| Repository pattern | Faible | Deja implemente |

#### 4. Layered suffit pour les besoins actuels

```
app/api/routes.py      # Controller: HTTP validation
services/inference.py  # Service: Logique ML pure
services/composer.py   # Service: Logique optimisation pure
services/data_loader.py # Repository: I/O MongoDB/Parquet
```

- **Testabilite**: Services sans I/O, facilement mockables
- **Lisibilite**: Flux lineaire, pas d'indirection
- **Maintenance**: 1 dev peut comprendre tout le code

### Quand reconsiderer DDD

Adopter DDD si Alice-Engine evolue vers:

1. **Multi-domaines**: Gestion licences, calendrier, paiements, notifications
2. **Equipe elargie**: 5+ developpeurs necessitant bounded contexts
3. **Regles metier complexes**: Workflows avec etats, branches, rollbacks
4. **Event-driven**: Besoin de Domain Events, CQRS, Event Sourcing

Indicateurs de bascule:
- Plus de 3 Bounded Contexts identifies
- Plus de 50 regles metier
- Plus de 5 developpeurs
- Couplage fort entre modules

### Consequences

#### Positif
- Code simple et explicite
- Onboarding rapide (< 1 jour)
- Pas de framework DDD a maitriser
- Performance: pas d'indirection inutile

#### Negatif
- Si le domaine se complexifie, refactoring necessaire
- Moins de patterns "enterprise-ready"

#### Neutre
- ISO 42010 respecte (documentation architecture)
- SRP respecte (separation des couches)
- Testabilite equivalente

### Alternatives evaluees

| Approche | Verdict | Raison |
|----------|---------|--------|
| DDD complet | Rejete | Over-engineering |
| DDD-lite (tactique only) | Rejete | Benefice insuffisant |
| Hexagonal | Rejete | Ports/Adapters excessifs |
| Clean Architecture | Rejete | 4 couches = trop pour le projet |
| **Layered + SRP** | **Accepte** | Equilibre complexite/benefice |

### References

- ISO 42010: Architecture Description
- Martin Fowler: "Is Design Dead?" (simplicite vs patterns)
- Eric Evans: "DDD - Tackling Complexity" (quand utiliser DDD)
- YAGNI: You Aren't Gonna Need It

---

*Derniere mise a jour: 8 Janvier 2026*
