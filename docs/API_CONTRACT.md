# API Contract : ALICE
## Adversarial Lineup Inference & Composition Engine

**Version** : 1.0.0
**Base URL** : `https://alice-engine.onrender.com` (production)
**Base URL** : `http://localhost:8000` (développement)

---

## Authentification

### Endpoints Publics (pas d'auth requise)
- `GET /health`
- `GET /`

### Endpoints Protégés (API Key requise)
- `POST /train`

Header requis :
```
X-API-Key: {API_KEY}
```

### Endpoints Standard (clubId requis)
- `POST /predict`
- `GET /models/{clubId}`

Le clubId est validé dans le payload, pas d'authentification JWT (chess-app gère ça).

---

## Endpoints

### GET /

**Description** : Page d'accueil / info service

**Réponse** :
```json
{
  "service": "ALICE",
  "fullName": "Adversarial Lineup Inference & Composition Engine",
  "version": "1.0.0",
  "status": "running",
  "documentation": "/docs"
}
```

---

### GET /health

**Description** : Vérification santé du service (pour monitoring)

**Réponse 200** :
```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T10:30:00Z",
  "checks": {
    "mongodb": "connected",
    "model": "loaded"
  }
}
```

**Réponse 503** (service dégradé) :
```json
{
  "status": "degraded",
  "timestamp": "2025-12-30T10:30:00Z",
  "checks": {
    "mongodb": "disconnected",
    "model": "loaded"
  },
  "message": "Database connection lost"
}
```

---

### POST /predict

**Description** : Endpoint principal - Prédiction de composition adverse et optimisation

#### Requête

**Headers** :
```
Content-Type: application/json
X-API-Version: 1.0  (optionnel, pour évolution future)
```

**Body** :
```json
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
      "isFemale": false,
      "isReserve": false,
      "isMuted": false,
      "matchesPlayed": 5
    }
  ],
  "constraints": {
    "teamSize": 8,
    "eloDescending": true,
    "minFemales": 0,
    "maxMuted": 2
  },
  "options": {
    "scenarioCount": 20,
    "includeAlternatives": true,
    "alternativeCount": 3
  }
}
```

#### Paramètres Détaillés

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `clubId` | string | ✅ | ID FFE du club utilisateur |
| `opponentClubId` | string | ✅ | ID FFE du club adverse |
| `competitionId` | string | ❌ | ID compétition (pour affiner) |
| `roundNumber` | integer | ✅ | Numéro de la ronde |
| `availablePlayers` | array | ✅ | Joueurs disponibles |
| `constraints` | object | ❌ | Contraintes de composition |
| `options` | object | ❌ | Options de calcul |

**availablePlayers[]** :

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `ffeId` | string | ✅ | ID FFE du joueur |
| `elo` | integer | ✅ | Elo actuel |
| `name` | string | ❌ | Nom complet |
| `category` | string | ❌ | Catégorie (Pou, Pup, Ben, Min, Cad, Jun, Sen, Vet) |
| `isFemale` | boolean | ❌ | true si joueuse |
| `isReserve` | boolean | ❌ | true si joueur réserve/occasionnel |
| `isMuted` | boolean | ❌ | true si muté |
| `matchesPlayed` | integer | ❌ | Matchs joués cette saison |

**constraints** :

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `teamSize` | integer | 8 | Nombre de joueurs requis |
| `eloDescending` | boolean | true | Ordre Elo décroissant obligatoire |
| `minFemales` | integer | 0 | Minimum de joueuses |
| `maxMuted` | integer | null | Maximum de mutés |

**options** :

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `scenarioCount` | integer | 20 | Nombre de scénarios adverses |
| `includeAlternatives` | boolean | true | Inclure compositions alternatives |
| `alternativeCount` | integer | 3 | Nombre d'alternatives |

#### Réponse Succès (200)

```json
{
  "success": true,
  "version": "1.0.0",

  "predictedOpponentLineup": [
    {
      "board": 1,
      "ffeId": "B98765",
      "name": "Martin Pierre",
      "elo": 2100,
      "probability": 0.85,
      "reasoning": "Present 17/20 last matches"
    },
    {
      "board": 2,
      "ffeId": "B87654",
      "name": "Durand Paul",
      "elo": 1950,
      "probability": 0.72,
      "reasoning": "Usually plays away matches"
    }
  ],

  "scenarios": [
    {
      "id": 1,
      "probability": 0.25,
      "lineup": [
        { "board": 1, "ffeId": "B98765", "elo": 2100 },
        { "board": 2, "ffeId": "B87654", "elo": 1950 }
      ]
    }
  ],

  "recommendedLineup": [
    {
      "board": 1,
      "ffeId": "A12345",
      "name": "Dupont Jean",
      "elo": 1850,
      "expectedScore": 0.35,
      "winProbability": 0.28,
      "drawProbability": 0.35,
      "lossProbability": 0.37,
      "opponent": {
        "ffeId": "B98765",
        "name": "Martin Pierre",
        "elo": 2100
      }
    }
  ],

  "expectedMatchScore": 4.5,
  "scoreRange": {
    "pessimistic": 3.5,
    "expected": 4.5,
    "optimistic": 5.5
  },
  "confidence": 0.72,

  "alternatives": [
    {
      "rank": 2,
      "description": "Renforcement échiquier 3",
      "lineup": [...],
      "expectedScore": 4.3,
      "tradeoff": "Sacrifie échiquier 1 pour sécuriser échiquier 3"
    }
  ],

  "warnings": [
    {
      "code": "LOW_DATA",
      "message": "Only 15 historical matches found for opponent"
    }
  ],

  "metadata": {
    "processingTimeMs": 245,
    "modelVersion": "global-v1",
    "dataPointsUsed": 150,
    "predictionDate": "2025-12-30T10:30:00Z"
  }
}
```

#### Réponse Erreur Validation (400)

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "availablePlayers",
        "error": "Must contain at least 8 players"
      }
    ]
  }
}
```

#### Réponse Erreur Club Non Trouvé (404)

```json
{
  "success": false,
  "error": {
    "code": "CLUB_NOT_FOUND",
    "message": "Opponent club not found in database",
    "details": {
      "opponentClubId": "9999"
    }
  }
}
```

#### Réponse Données Insuffisantes (200 avec warning)

```json
{
  "success": true,
  "warning": "Insufficient historical data, using fallback model",
  "confidence": 0.35,
  "predictedOpponentLineup": [...],
  "recommendedLineup": [...],
  "metadata": {
    "modelVersion": "fallback-global",
    "dataPointsUsed": 8
  }
}
```

#### Réponse Erreur Serveur (500)

```json
{
  "success": false,
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "requestId": "abc123"
  }
}
```

---

### GET /models/{clubId}

**Description** : Informations sur le modèle utilisé pour un club

**Paramètres URL** :
- `clubId` : ID FFE du club

**Réponse 200** :
```json
{
  "clubId": "2976",
  "modelType": "club-specific",
  "modelVersion": "club-2976-v3",
  "lastTrainedAt": "2025-12-28T15:00:00Z",
  "trainingDataPoints": 250,
  "accuracy": 0.78,
  "features": [
    "elo",
    "round_number",
    "home_away",
    "recent_form",
    "opponent_strength"
  ]
}
```

**Réponse 200 (modèle global)** :
```json
{
  "clubId": "1234",
  "modelType": "global-fallback",
  "modelVersion": "global-v1",
  "reason": "Insufficient data for club-specific model (23 matches, minimum 50)",
  "lastTrainedAt": "2025-12-01T00:00:00Z"
}
```

---

### POST /train

**Description** : Déclencher un réentraînement du modèle (protégé)

**Headers** :
```
Content-Type: application/json
X-API-Key: {API_KEY}
```

**Body** :
```json
{
  "clubId": "2976",
  "forceRetrain": false
}
```

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `clubId` | string | ❌ | Si absent, réentraîne le modèle global |
| `forceRetrain` | boolean | ❌ | Force même si données insuffisantes |

**Réponse 202** (tâche lancée) :
```json
{
  "success": true,
  "message": "Training job started",
  "jobId": "train-2976-20251230",
  "estimatedDuration": "5 minutes"
}
```

**Réponse 401** (non autorisé) :
```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key"
  }
}
```

---

## Codes d'Erreur

| Code | HTTP | Description |
|------|------|-------------|
| `VALIDATION_ERROR` | 400 | Paramètres invalides |
| `CLUB_NOT_FOUND` | 404 | Club non trouvé |
| `INSUFFICIENT_PLAYERS` | 400 | Pas assez de joueurs disponibles |
| `UNAUTHORIZED` | 401 | API key manquante/invalide |
| `RATE_LIMITED` | 429 | Trop de requêtes |
| `INTERNAL_ERROR` | 500 | Erreur serveur |
| `MODEL_LOADING_ERROR` | 503 | Modèle non chargé |
| `DATABASE_ERROR` | 503 | Erreur MongoDB |

---

## Rate Limiting

### Limites par défaut

| Endpoint | Limite |
|----------|--------|
| `/predict` | 100 requêtes/minute par clubId |
| `/train` | 1 requête/heure |
| `/health` | Illimité |

### Headers de réponse

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1703934600
```

---

## Versioning

### Header de version

```
X-API-Version: 1.0
```

### Politique de compatibilité

- Les nouveaux champs optionnels peuvent être ajoutés (non-breaking)
- Les champs existants ne seront jamais supprimés sans nouvelle version majeure
- Les réponses incluent toujours `version` pour traçabilité

---

## Exemples cURL

### Test health
```bash
curl https://alice-engine.onrender.com/health
```

### Prédiction simple
```bash
curl -X POST https://alice-engine.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "clubId": "2976",
    "opponentClubId": "1234",
    "roundNumber": 3,
    "availablePlayers": [
      {"ffeId": "A12345", "elo": 1850, "name": "Dupont Jean"},
      {"ffeId": "A23456", "elo": 1720, "name": "Martin Paul"},
      {"ffeId": "A34567", "elo": 1680, "name": "Durand Marie"},
      {"ffeId": "A45678", "elo": 1620, "name": "Bernard Luc"},
      {"ffeId": "A56789", "elo": 1580, "name": "Petit Anne"},
      {"ffeId": "A67890", "elo": 1540, "name": "Robert Jean"},
      {"ffeId": "A78901", "elo": 1490, "name": "Simon Claire"},
      {"ffeId": "A89012", "elo": 1450, "name": "Laurent Marc"}
    ],
    "constraints": {
      "teamSize": 8,
      "eloDescending": true
    }
  }'
```

### Déclencher entraînement
```bash
curl -X POST https://alice-engine.onrender.com/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"clubId": "2976"}'
```

---

## OpenAPI/Swagger

Documentation interactive disponible à :
- `/docs` - Swagger UI
- `/redoc` - ReDoc
- `/openapi.json` - Spécification OpenAPI brute
