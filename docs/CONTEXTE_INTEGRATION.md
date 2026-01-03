# Contexte d'Intégration : ALICE ↔ chess-app

## Vue d'Ensemble

Ce document décrit comment ALICE (Adversarial Lineup Inference & Composition Engine) s'intègre avec chess-app, sans créer de dépendance rigide.

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UTILISATEUR                                  │
│                    (Capitaine d'équipe)                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FRONTEND (Vercel)                               │
│                         chess-app                                    │
│                      React / Vite / PWA                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Page Composition d'Équipe                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │   │
│  │  │ Mode Manuel │  │ Mode Cascade│  │ Demander à ALICE     │ │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND chess-app (Render)                      │
│                      Express.js / Fastify                            │
│                                                                      │
│  Routes existantes:              Nouvelle route:                     │
│  - /api/players                  - /api/alice/predict ◄── NEW       │
│  - /api/teams                                                        │
│  - /api/compositions                                                 │
│  - /api/validate (Flat-Six)                                         │
└─────────────────────────────────────────────────────────────────────┘
                │                               │
                │                               │
                ▼                               ▼
┌──────────────────────────┐    ┌─────────────────────────────────────┐
│     MongoDB Atlas        │    │          ALICE (Render)             │
│                          │    │        Python / FastAPI              │
│  Collections:            │◄───│                                      │
│  - players               │    │  Endpoints:                          │
│  - matches               │    │  - POST /predict                     │
│  - compositions          │    │  - GET /health                       │
│  - teams                 │    │  - GET /models/{clubId}              │
│                          │    │  - POST /train                       │
└──────────────────────────┘    └─────────────────────────────────────┘
```

---

## Flux de Données

### Scénario : Capitaine demande à Alice une composition optimale

```
1. Capitaine clique "Demander à Alice" dans chess-app
                    │
                    ▼
2. Frontend envoie requête au backend chess-app
   POST /api/alice/predict
   {
     opponentClubId: "1234",
     roundNumber: 3,
     competitionId: "N5-2025"
   }
                    │
                    ▼
3. Backend chess-app:
   a) Récupère les joueurs disponibles (MongoDB)
   b) Récupère les contraintes (rules.json simplifié)
   c) Appelle ALICE
                    │
                    ▼
4. ALICE:
   a) ALI : Charge historique adversaire (MongoDB, lecture)
   b) ALI : Exécute prédiction XGBoost → lineup adverse probable
   c) CE : Optimise composition utilisateur
   d) Retourne résultat
                    │
                    ▼
5. Backend chess-app:
   a) Reçoit la réponse d'Alice
   b) Valide avec Flat-Six (optionnel)
   c) Retourne au frontend
                    │
                    ▼
6. Frontend affiche:
   - "Alice prédit cette composition adverse..."
   - "Alice recommande cette composition..."
   - Score attendu
   - Alternatives
```

---

## Contrat d'Interface

### Ce que chess-app envoie à chess-predictor

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
      "isMuted": false
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
    "includeAlternatives": true
  }
}
```

### Ce que chess-predictor retourne

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
      "probability": 0.85
    },
    {
      "board": 2,
      "ffeId": "B87654",
      "name": "Durand Paul",
      "elo": 1950,
      "probability": 0.72
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
      "description": "Version défensive",
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

---

## Responsabilités Séparées

### chess-app est responsable de :

| Fonction | Détails |
|----------|---------|
| **Scraping FFE** | Import des joueurs, résultats, historique |
| **Stockage MongoDB** | Toutes les collections |
| **Authentification** | JWT, gestion users, rôles |
| **Règles FFE complètes** | Moteur Flat-Six (73+ règles) |
| **Interface utilisateur** | Toutes les pages React |
| **Validation finale** | Appel Flat-Six après réponse d'Alice |
| **Gestion multi-clubs** | ClubId dans JWT et requêtes |

### ALICE est responsable de :

| Fonction | Composant | Détails |
|----------|-----------|---------|
| **Prédiction présence** | ALI | Probabilité de chaque joueur adverse |
| **Génération scénarios** | ALI | 20-50 lineups possibles adverses |
| **Optimisation composition** | CE | Maximisation score attendu |
| **Calculs Elo** | CE | Formule standard |
| **Gestion modèles ML** | - | Entraînement, stockage, versioning |
| **Contraintes simplifiées** | CE | Ordre Elo, taille équipe |

### Ce qu'ALICE NE fait PAS :

- ❌ Authentification utilisateur
- ❌ Validation complète règles FFE
- ❌ Stockage de données (lecture seule)
- ❌ Interface utilisateur
- ❌ Scraping FFE

---

## Gestion des Erreurs

### Côté ALICE

```json
// Erreur de validation
{
  "success": false,
  "error": {
    "code": "INVALID_CLUB_ID",
    "message": "Club ID not found in database",
    "details": { "clubId": "9999" }
  }
}

// Erreur de prédiction
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_DATA",
    "message": "Not enough historical data for prediction",
    "details": { "matchesFound": 5, "minimumRequired": 10 }
  }
}

// Service dégradé
{
  "success": true,
  "warning": "Using fallback global model",
  "predictedOpponentLineup": [...],
  "confidence": 0.45
}
```

### Côté chess-app (gestion des erreurs ALICE)

```javascript
// Dans chess-app : services/aliceService.js

async function askAlice(params) {
  try {
    const response = await fetch(`${ALICE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal: AbortSignal.timeout(35000) // Cold start Render
    });
    
    const data = await response.json();
    
    if (!data.success) {
      // Log l'erreur mais ne bloque pas l'utilisateur
      console.warn('Alice prediction failed:', data.error);
      return {
        success: false,
        fallback: true,
        message: 'Alice est indisponible, utilisez le mode manuel'
      };
    }
    
    return data;
    
  } catch (error) {
    // Service inaccessible (cold start trop long, etc.)
    console.error('Alice unreachable:', error);
    return {
      success: false,
      fallback: true,
      message: 'Alice est temporairement indisponible'
    };
  }
}
```

---

## Accès MongoDB

### Configuration ALICE

ALICE accède à la même base MongoDB Atlas que chess-app, mais en **lecture seule**.

```python
# Dans alice-engine : app/config.py

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "chess-app"  # Même base que chess-app

# Collections utilisées (lecture seule)
COLLECTIONS = {
    "players": "players",
    "matches": "matches", 
    "compositions": "compositions",
    "teams": "teams"
}
```

### Données Lues par ALICE

```python
# Exemple de requête pour historique adversaire (composant ALI)
async def get_opponent_history(db, opponent_club_id: str, limit: int = 50):
    """
    Récupère les compositions passées d'un club adverse.
    Utilisé par le composant Adversarial Lineup Inference.
    """
    cursor = db.compositions.find(
        {"clubId": opponent_club_id},
        {"_id": 0, "roundNumber": 1, "players": 1, "date": 1}
    ).sort("date", -1).limit(limit)
    
    return await cursor.to_list(length=limit)
```

---

## Évolutions Futures

### Phase 1 (MVP)
- ALI basique avec modèle global
- CE par force brute

### Phase 2
- Modèles par club (si >50 matchs historiques)
- Feature engineering avancé (forme récente, domicile/extérieur)

### Phase 3
- Suggestions de transferts stratégiques
- Analyse comparative entre clubs

### Phase 4 (chess-app)
- Intégration UI complète ("Demander à Alice")
- Historique des prédictions vs réalité
- Amélioration continue du modèle

---

## Variables d'Environnement Partagées

### chess-app (.env)
```env
ALICE_URL=https://alice-engine.onrender.com
ALICE_TIMEOUT=35000
```

### ALICE (.env)
```env
MONGODB_URI=mongodb+srv://readonly:xxx@cluster.mongodb.net/chess-app
API_KEY=xxx_for_train_endpoint
```

---

## Tests d'Intégration

### Test de bout en bout (manuel)

1. Démarrer ALICE localement
2. Envoyer requête curl :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"clubId":"2976","opponentClubId":"1234","roundNumber":3,"availablePlayers":[{"ffeId":"A12345","elo":1850}],"constraints":{"teamSize":8}}'
```

3. Vérifier la réponse JSON

### Test depuis chess-app (futur)

```javascript
// tests/integration/alice.test.js
describe('ALICE Integration', () => {
  it('should get prediction from Alice for valid request', async () => {
    const result = await askAlice({
      clubId: '2976',
      opponentClubId: '1234',
      roundNumber: 3,
      availablePlayers: mockPlayers
    });
    
    expect(result.success).toBe(true);
    expect(result.recommendedLineup).toHaveLength(8);
  });
  
  it('should handle Alice timeout gracefully', async () => {
    // Simuler timeout
    const result = await askAlice({...}, { timeout: 1 });
    
    expect(result.success).toBe(false);
    expect(result.fallback).toBe(true);
  });
});
```

---

## Checklist d'Intégration

### Avant de connecter les deux projets

- [ ] ALICE déployé sur Render
- [ ] Endpoint /health répond OK
- [ ] Endpoint /predict fonctionne avec données de test
- [ ] MongoDB accessible en lecture
- [ ] Cold start < 30 secondes

### Côté chess-app (quand prêt)

- [ ] Variable ALICE_URL configurée
- [ ] Service aliceService.js créé
- [ ] Gestion timeout et fallback
- [ ] Route /api/alice/predict créée
- [ ] UI bouton "Demander à Alice" ajouté
- [ ] Tests d'intégration passent
