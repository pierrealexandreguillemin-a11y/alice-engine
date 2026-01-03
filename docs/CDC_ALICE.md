# Cahier des Charges : ALICE
## Adversarial Lineup Inference & Composition Engine

**Version** : 1.0.0  
**Date** : 30 décembre 2025  
**Statut** : Nouveau projet - Phase de création

---

## 1. Présentation du Projet

### 1.1 Nom et Signification

**ALICE** = **A**dversarial **L**ineup **I**nference & **C**omposition **E**ngine

> *"J'ai fait les compos avec Alice"* - Usage naturel prévu

**Définition technique** : Module de prédiction par ensemble learning (XGBoost) sur données tabulaires structurées, implémentant une inférence probabiliste multi-output pour la modélisation adverse, couplé à un optimiseur combinatoire maximisant l'espérance du score de match via la fonction de densité Elo.

### 1.2 Contexte

Ce module est un **composant satellite** de l'application principale `chess-app`, une solution SaaS de gestion de clubs d'échecs français. 

**chess-app** (projet parent) :
- Application web Next.js/React (frontend) + Express/Fastify (backend)
- Base de données MongoDB Atlas
- Déployé sur Vercel (frontend) et Render (backend)
- Gère les joueurs, équipes, compositions, convocations pour les interclubs FFE
- Intègre un moteur de règles "Flat-Six" validant 73+ règles FFE

**ALICE** (ce projet) :
- Service Python autonome
- Communique avec chess-app via API REST
- Fournit des prédictions de compositions adverses et optimisations
- Nom technique : `alice-engine` (repo et déploiement)

### 1.3 Objectif Principal

Créer un service de **prédiction intelligente** permettant aux capitaines d'équipes de :

1. **Anticiper la composition adverse** probable pour une ronde donnée (Adversarial Lineup Inference)
2. **Optimiser leur propre composition** pour maximiser le score attendu (Composition Engine)
3. **Explorer des scénarios** avec différentes probabilités

### 1.4 Différenciateur Commercial

ALICE est le **cœur différenciant** du SaaS chess-app. Aucune solution concurrente n'offre de prédictions probabilistes avancées pour les capitaines d'interclubs français.

---

## 2. Spécifications Fonctionnelles

### 2.1 Fonctionnalités Principales

#### F.1 Prédiction de Composition Adverse

**Entrées** :
- Identifiant du club adverse (clubId FFE)
- Numéro de ronde
- Historique des compositions passées (via scraping FFE déjà fait par chess-app)

**Traitement** :
- Analyse des patterns de présence de chaque joueur adverse
- Calcul de probabilité de présence par joueur
- Estimation de la position probable sur l'échiquier (ordre Elo)

**Sorties** :
- Liste ordonnée des joueurs adverses probables avec probabilités
- 20-50 scénarios de lineup pondérés

#### F.2 Optimisation de Composition

**Entrées** :
- Liste des joueurs disponibles du club utilisateur (avec Elo)
- Contraintes de composition (règles FFE simplifiées)
- Prédiction adverse (issue de F.1)

**Traitement** :
- Calcul du score attendu pour chaque configuration possible
- Formule Elo standard : P(victoire) = 1 / (1 + 10^((Elo_adv - Elo_mon)/400))
- Ajustement pour matchs nuls (~0.35 points)

**Sorties** :
- Composition optimale recommandée
- Score attendu total du match
- Alternatives classées

#### F.3 Respect des Contraintes FFE

Le module reçoit les contraintes en entrée (pas de duplication du moteur de règles) :
- Ordre Elo décroissant obligatoire
- Nombre de joueurs min/max
- Autres contraintes passées en paramètres

**Note** : Le moteur Flat-Six complet reste dans chess-app. Ce module applique des contraintes simplifiées pour l'optimisation.

### 2.2 Multi-Tenant

- Isolation stricte par `clubId`
- Modèle global par défaut
- Modèle spécifique par club si données suffisantes (>50-100 matchs historiques)
- Aucune fuite de données entre clubs

---

## 3. Spécifications Techniques

### 3.1 Stack Technologique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Langage | Python | 3.11+ |
| Framework API | FastAPI | latest |
| Serveur ASGI | Uvicorn | latest |
| Machine Learning | XGBoost | 3.1.2 |
| Data Processing | Pandas, NumPy | latest |
| ML Utils | scikit-learn | latest |
| Persistance Modèles | joblib | latest |
| MongoDB (async) | motor | latest |
| Validation | Pydantic | v2 |

### 3.2 Architecture

```
alice-engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # Point d'entrée FastAPI
│   ├── config.py            # Configuration (env vars)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # Endpoints API
│   │   └── schemas.py       # Modèles Pydantic (entrées/sorties)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference.py     # Adversarial Lineup Inference (ALI)
│   │   ├── composer.py      # Composition Engine (CE)
│   │   └── data_loader.py   # Chargement données MongoDB
│   ├── models/
│   │   ├── __init__.py
│   │   └── xgboost_model.py # Wrapper modèle XGBoost
│   └── utils/
│       ├── __init__.py
│       └── elo_calculator.py # Formules Elo
├── models/                   # Modèles entraînés (.joblib)
│   └── global_model.joblib
├── tests/
│   ├── __init__.py
│   ├── test_inference.py
│   └── test_composer.py
├── scripts/
│   └── train_model.py       # Script d'entraînement
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── render.yaml              # Config déploiement Render
```

### 3.3 API Contract

#### Endpoint Principal

**POST /predict**

```json
// Requête
{
  "clubId": "2976",
  "opponentClubId": "1234", 
  "competitionId": "N5-2025",
  "roundNumber": 3,
  "availablePlayers": [
    {
      "ffeId": "A12345",
      "elo": 1850,
      "name": "Dupont Jean",
      "category": "Sen",
      "isFemale": false
    }
  ],
  "constraints": {
    "minPlayers": 8,
    "maxPlayers": 8,
    "eloDescending": true,
    "minFemales": 0
  },
  "options": {
    "scenarioCount": 20,
    "includeAlternatives": true
  }
}
```

```json
// Réponse
{
  "success": true,
  "version": "1.0.0",
  "predictedOpponentLineup": [
    {
      "board": 1,
      "ffeId": "B98765",
      "name": "Martin Pierre",
      "elo": 2100,
      "probability": 0.85
    }
  ],
  "scenarios": [
    {
      "id": 1,
      "probability": 0.25,
      "lineup": [...]
    }
  ],
  "recommendedLineup": [
    {
      "board": 1,
      "ffeId": "A12345",
      "name": "Dupont Jean",
      "elo": 1850,
      "expectedScore": 0.35,
      "opponent": {
        "ffeId": "B98765",
        "elo": 2100
      }
    }
  ],
  "expectedMatchScore": 4.5,
  "confidence": 0.72,
  "alternatives": [
    {
      "rank": 2,
      "lineup": [...],
      "expectedScore": 4.3
    }
  ],
  "metadata": {
    "processingTimeMs": 245,
    "modelVersion": "global-v1",
    "dataPointsUsed": 150
  }
}
```

#### Endpoints Secondaires

**GET /health** - Vérification santé du service

**GET /models/{clubId}** - Info sur le modèle utilisé pour un club

**POST /train** - Déclencher un réentraînement (protégé par API key)

### 3.4 Déploiement

**Plateforme** : Render (Free/Hobby tier)

**URL Production** : `https://alice-engine.onrender.com`

**Configuration** :
- Build command : `pip install -r requirements.txt`
- Start command : `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Auto-deploy depuis GitHub

**Contraintes Free Tier** :
- Spin down après 15 min d'inactivité
- Cold start ~10-30 secondes
- Solution : ping cron via UptimeRobot (gratuit)

### 3.5 Variables d'Environnement

```env
# MongoDB (lecture seule des données historiques)
MONGODB_URI=mongodb+srv://...

# Sécurité
API_KEY=xxx  # Pour endpoint /train

# Configuration ALICE
MODEL_PATH=./models
DEFAULT_SCENARIO_COUNT=20
LOG_LEVEL=INFO

# Optionnel - métriques
SENTRY_DSN=xxx
```

---

## 4. Intégration avec chess-app

### 4.1 Principe de Découplage

```
┌─────────────────┐         HTTP/REST         ┌──────────────────┐
│                 │  ───────────────────────► │                  │
│   chess-app     │    POST /predict          │     ALICE        │
│   (TypeScript)  │  ◄─────────────────────── │  (Python)        │
│                 │         JSON              │                  │
└─────────────────┘                           └──────────────────┘
     Vercel                                        Render
```

### 4.2 Côté chess-app (futur)

Fichier à créer dans chess-app quand prêt :

```javascript
// services/aliceService.js

const ALICE_URL = process.env.ALICE_URL || 'https://alice-engine.onrender.com';

export async function getAlicePrediction(clubId, opponentClubId, roundNumber, availablePlayers, constraints = {}) {
  try {
    const response = await fetch(`${ALICE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Version': '1.0'
      },
      body: JSON.stringify({
        clubId,
        opponentClubId,
        roundNumber,
        availablePlayers,
        constraints
      }),
      timeout: 35000 // Cold start Render
    });
    
    if (!response.ok) {
      throw new Error(`ALICE error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('ALICE service unavailable:', error);
    return { success: false, error: 'Service indisponible' };
  }
}
```

### 4.3 Points de Coordination

| Aspect | Responsabilité chess-app | Responsabilité ALICE |
|--------|--------------------------|-------------------------------|
| Données joueurs | Scraping FFE, stockage MongoDB | Lecture seule MongoDB |
| Règles FFE | Moteur Flat-Six complet (73 règles) | Contraintes simplifiées en entrée |
| Authentification | JWT utilisateur, gestion clubId | Validation clubId reçu |
| UI | Bouton "Demander à Alice", affichage | Aucune |
| Historique matchs | Stockage via scraping | Utilisation pour entraînement |

---

## 5. Plan de Développement

### Phase 1 : Squelette (Sprint 1)
- [ ] Structure projet Python `alice-engine`
- [ ] FastAPI avec endpoint /health
- [ ] Configuration environnement
- [ ] Déploiement Render (vide mais fonctionnel)

### Phase 2 : Adversarial Lineup Inference (Sprint 2)
- [ ] Connexion MongoDB (lecture)
- [ ] Chargement données historiques adverses
- [ ] Modèle XGBoost (prédiction présence joueurs)
- [ ] Endpoint /predict - partie inference

### Phase 3 : Composition Engine (Sprint 3)
- [ ] Calcul score attendu Elo
- [ ] Algorithme d'optimisation composition
- [ ] Génération scénarios multiples
- [ ] Endpoint /predict - partie composition
- [ ] Tests unitaires

### Phase 4 : Production (Sprint 4)
- [ ] Multi-tenant (modèles par club)
- [ ] Script d'entraînement automatisé
- [ ] Monitoring et logs
- [ ] Documentation API complète

---

## 6. Contraintes et Principes

### 6.1 Budget
**0€ supplémentaire** - Utilisation exclusive des free tiers :
- Render Hobby (750h/mois)
- MongoDB Atlas existant (lecture)

### 6.2 Performance
- Inférence < 1 seconde (hors cold start)
- Cold start acceptable (< 30s) avec fallback gracieux

### 6.3 Principes de Développement

Alignés avec ceux de chess-app :
- **UN COMMIT = UNE FONCTIONNALITÉ TESTÉE**
- Diagnostic avant correction
- Tests avant déploiement
- Documentation à jour

### 6.4 Indépendance

ALICE doit pouvoir :
- Être développé sans toucher à chess-app
- Être testé de manière isolée
- Évoluer à son propre rythme

---

## 7. Ressources

### Documentation XGBoost
- https://xgboost.readthedocs.io/

### FastAPI
- https://fastapi.tiangolo.com/

### Render Deployment
- https://render.com/docs/deploy-fastapi

### Formule Elo
- P(A gagne) = 1 / (1 + 10^((Elo_B - Elo_A)/400))
- Score nul ≈ 0.35 points en moyenne

---

## 8. Glossaire

| Terme | Définition |
|-------|------------|
| **ALICE** | Adversarial Lineup Inference & Composition Engine - nom du module |
| **ALI** | Adversarial Lineup Inference - composant de prédiction adverse |
| **CE** | Composition Engine - composant d'optimisation |
| FFE | Fédération Française des Échecs |
| Elo | Système de classement des joueurs (1000-2800 typiquement) |
| Interclubs | Compétition par équipes entre clubs |
| Ronde | Une journée de compétition (match aller ou retour) |
| Échiquier | Position d'un joueur dans l'équipe (1 = meilleur joueur) |
| Lineup | Composition d'équipe ordonnée |
| Flat-Six | Nom du moteur de règles FFE dans chess-app |
| clubId | Identifiant FFE unique du club (ex: "2976") |
| ffeId | Identifiant FFE unique du joueur (ex: "A12345") |
