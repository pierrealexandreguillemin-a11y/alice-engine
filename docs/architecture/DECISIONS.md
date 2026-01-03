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

*Derniere mise a jour: 3 Janvier 2026*
