# Deploiement ALICE sur Render

> **Auto-genere** - Voir `render.yaml` pour config complete
> **Derniere mise a jour**: 3 Janvier 2026

---

## 1. Architecture Multi-Services

```
┌─────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  Frontend       │────▶│  Backend Node.js        │────▶│  ALICE Engine           │
│  Vercel         │     │  Render (teamsync)      │     │  Render (alice-engine)  │
│  teamsync-iota  │     │  teamsync-ziz4          │     │  alice-engine           │
└─────────────────┘     └──────────┬──────────────┘     └──────────┬──────────────┘
                                   │                               │
                                   └───────────────┬───────────────┘
                                                   ▼
                                        ┌─────────────────────────┐
                                        │  MongoDB Atlas          │
                                        │  cluster0.yhzms1m       │
                                        │  (lecture seule ALICE)  │
                                        └─────────────────────────┘
```

---

## 2. Configuration Render

### Variables d'environnement (Dashboard Render)

| Variable | Valeur | Source |
|----------|--------|--------|
| `MONGODB_URI` | `mongodb+srv://...` | chess-app backend/.env |
| `API_KEY` | Generer une cle | `openssl rand -hex 32` |
| `LOG_LEVEL` | `INFO` | - |

### Limites Free Tier

| Ressource | Limite | Note |
|-----------|--------|------|
| Heures/mois | 750h partagees | ~31 jours si 1 service |
| RAM | 512 MB | Suffisant pour CatBoost |
| Cold start | 15 min inactivite | Voir solution ci-dessous |
| Requetes | Illimitees | - |

---

## 3. Solution Cold Start (Keep-Alive)

### Option A: Ping depuis Frontend (Recommande)

Ajouter dans `chess-app/frontend/src/App.tsx` :

```typescript
// Keep-alive ALICE pendant que l'app est ouverte
useEffect(() => {
  const ALICE_URL = import.meta.env.VITE_ALICE_URL || 'https://alice-engine.onrender.com';

  // Ping initial
  fetch(`${ALICE_URL}/health`).catch(() => {});

  // Ping toutes les 10 minutes (avant le cold start de 15 min)
  const interval = setInterval(() => {
    fetch(`${ALICE_URL}/health`).catch(() => {});
  }, 10 * 60 * 1000); // 10 minutes

  return () => clearInterval(interval);
}, []);
```

### Option B: UptimeRobot (Gratuit, externe)

1. Creer compte sur https://uptimerobot.com
2. Ajouter moniteur HTTP(s)
3. URL: `https://alice-engine.onrender.com/health`
4. Intervalle: 5 minutes (gratuit)

### Option C: Cron-job.org (Gratuit)

1. Creer compte sur https://cron-job.org
2. URL: `https://alice-engine.onrender.com/health`
3. Schedule: `*/10 * * * *` (toutes les 10 min)

---

## 4. Deploiement Step-by-Step

### Etape 1: Connecter GitHub

1. Aller sur https://dashboard.render.com
2. New > Blueprint
3. Connecter repo `pierrealexandreguillemin-a11y/alice-engine`
4. Render detecte automatiquement `render.yaml`

### Etape 2: Configurer Secrets

Dans Dashboard Render > alice-engine > Environment :

```
MONGODB_URI = mongodb+srv://pierrealexandreguillemin:****@cluster0.yhzms1m.mongodb.net/chess-app?retryWrites=true&w=majority
API_KEY = (generer avec: openssl rand -hex 32)
```

### Etape 3: Deployer

- Auto-deploy active par defaut
- Push sur `master` → deploiement automatique

### Etape 4: Verifier

```bash
curl https://alice-engine.onrender.com/health
```

---

## 5. Integration avec chess-app Backend

Dans `chess-app/backend/`, ajouter :

```typescript
// services/aliceClient.ts
const ALICE_URL = process.env.ALICE_URL || 'https://alice-engine.onrender.com';
const ALICE_API_KEY = process.env.ALICE_API_KEY;

export async function predictLineup(params: PredictParams) {
  const response = await fetch(`${ALICE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': ALICE_API_KEY
    },
    body: JSON.stringify(params)
  });
  return response.json();
}
```

---

## 6. URLs de Production

| Service | URL |
|---------|-----|
| Health | https://alice-engine.onrender.com/health |
| Swagger | https://alice-engine.onrender.com/docs |
| Predict | https://alice-engine.onrender.com/predict |

---

*Document auto-genere - Ne pas modifier manuellement*
