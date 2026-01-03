# ALICE Engine - Architecture Description

> **Version**: 0.1.0
> **Date**: 3 Janvier 2026
> **Norme**: ISO 42010 - Architecture Description

---

## 1. Introduction

### Objectif
Ce document decrit l'architecture logicielle d'ALICE (Adversarial Lineup Inference & Composition Engine).

### Audience
- Developpeurs
- Architectes
- Mainteneurs

### Documents lies
- [CDC_ALICE.md](../requirements/CDC_ALICE.md) - Cahier des charges
- [API_CONTRACT.md](../api/API_CONTRACT.md) - Contrat API

---

## 2. Vue d'ensemble

### Diagramme de contexte

```
┌─────────────────────────────────────────────────────────────────┐
│                         CHESS-APP                               │
│  ┌─────────────────┐                                            │
│  │  Frontend       │                                            │
│  │  (React/Vite)   │                                            │
│  │  Vercel         │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐         ┌─────────────────────────────┐   │
│  │  Backend        │────────▶│  ALICE Engine               │   │
│  │  (Node.js)      │  REST   │  (Python/FastAPI)           │   │
│  │  Render         │◀────────│  Render                     │   │
│  └────────┬────────┘         └──────────┬──────────────────┘   │
│           │                             │                       │
│           └──────────┬──────────────────┘                       │
│                      ▼                                          │
│           ┌─────────────────┐                                   │
│           │  MongoDB Atlas  │                                   │
│           │  (Shared)       │                                   │
│           └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Composants principaux

| Composant | Responsabilite | Technologie |
|-----------|---------------|-------------|
| **ALI** | Prediction composition adverse | CatBoost |
| **CE** | Optimisation composition propre | OR-Tools |
| **API** | Interface REST | FastAPI |
| **Data Loader** | Acces MongoDB | Motor (async) |

---

## 3. Architecture en couches (SRP)

```
┌─────────────────────────────────────────────────────────┐
│                    Controller Layer                      │
│  app/api/routes.py - HTTP routes, validation Pydantic   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     Service Layer                        │
│  services/inference.py - ALI (prediction)               │
│  services/composer.py  - CE (optimisation)              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Repository Layer                       │
│  services/data_loader.py - MongoDB I/O                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  MongoDB Atlas (chess-app database)                     │
│  models/*.cbm (CatBoost models)                         │
└─────────────────────────────────────────────────────────┘
```

### Regles SRP

1. **Controller** : HTTP uniquement, pas de logique metier
2. **Service** : Logique pure, testable, sans I/O direct
3. **Repository** : Seul a toucher la base de donnees

---

## 4. Flux de donnees

### Endpoint /predict

```
Client Request
      │
      ▼
┌─────────────────┐
│ routes.py       │ 1. Validation Pydantic (PredictRequest)
│ POST /predict   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ inference.py    │ 2. ALI: Charger modele CatBoost
│ predict_lineup  │    Generer features
│                 │    Predire composition adverse
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ composer.py     │ 3. CE: Calculer probabilites Elo
│ optimize_lineup │    Optimiser avec OR-Tools
│                 │    Generer alternatives
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ routes.py       │ 4. Serialisation PredictResponse
│ return JSON     │
└─────────────────┘
```

---

## 5. Securite (ISO 27001/27034)

### Secrets

| Secret | Stockage | Usage |
|--------|----------|-------|
| `MONGODB_URI` | Env var | Connexion DB |
| `API_KEY` | Env var | Auth endpoint /train |

### Validation

- Input: Pydantic v2 avec contraintes strictes
- Output: Schemas Pydantic (pas de donnees brutes)

### Rate Limiting

- Gere par Render/Vercel (niveau infra)
- Pas de rate limiting applicatif pour v0.1

---

## 6. Deploiement

### Environnements

| Env | URL | Infra |
|-----|-----|-------|
| Dev | localhost:8000 | Local |
| Prod | alice-engine.onrender.com | Render Free |

### CI/CD

```
git push master
      │
      ▼
┌─────────────────┐
│ Pre-commit      │ Lint, Format, Type, Security
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pre-push        │ Tests, Coverage, Audit
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GitHub          │ Push to origin
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Render          │ Auto-deploy Blueprint
└─────────────────┘
```

---

## 7. Decisions architecturales

Voir [DECISIONS.md](./DECISIONS.md) pour les ADR (Architecture Decision Records).

---

*Derniere mise a jour: 3 Janvier 2026*
