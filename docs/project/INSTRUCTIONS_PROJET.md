# Instructions Projet Claude : ALICE
## Adversarial Lineup Inference & Composition Engine

## Contexte Utilisateur

### Profil de Pierre-Alexandre
- **Niveau** : Débutant en programmation
- **Préférence** : Explications en langage simple, pas de jargon technique non expliqué
- **Style de travail** : Étape par étape, un fichier à la fois
- **Principe clé** : "UN COMMIT = UNE FONCTIONNALITÉ TESTÉE ET VALIDÉE"
- **Environnement** : Windows, VS Code avec Claude Code, terminal PowerShell

### Ce qu'il apprécie
- Instructions claires avec chemins de fichiers exacts
- Commandes console prêtes à copier-coller
- Diagnostic approfondi avant toute correction
- Comprendre le "pourquoi" avant le "comment"
- Recevoir UN fichier à la fois, pas des blocs combinés

### Ce qu'il n'apprécie PAS
- Les réponses "optimistes" qui cachent des problèmes
- Les corrections rapides sans diagnostic
- Le jargon non expliqué
- Les fichiers combinés ou trop d'infos d'un coup
- Contourner la sécurité "pour simplifier"

---

## Contexte Technique

### Projet Parent : chess-app
- **Localisation** : `C:\Dev\chess-app` (sur la machine de Pierre-Alexandre)
- **Stack** : React/Vite (frontend) + Express.js (backend) + MongoDB Atlas
- **Déploiement** : Vercel (frontend) + Render (backend)
- **État** : ~80-95% complet, frontend en cours de finition

### Ce Projet : ALICE (alice-engine)
- **Nom complet** : Adversarial Lineup Inference & Composition Engine
- **Usage naturel** : *"J'ai fait les compos avec Alice"*
- **Localisation prévue** : `C:\Dev\alice-engine`
- **Stack** : Python 3.11+ / FastAPI / XGBoost
- **Déploiement** : Render (service Python séparé)
- **URL** : `https://alice-engine.onrender.com`
- **État** : Nouveau projet, phase de création

### Composants d'ALICE

| Composant | Acronyme | Fonction |
|-----------|----------|----------|
| **Adversarial Lineup Inference** | ALI | Prédire la composition adverse probable |
| **Composition Engine** | CE | Optimiser sa propre composition |

### Relation entre les projets

```
chess-app (TypeScript)          ALICE (Python)
        │                               │
        │      HTTP POST /predict       │
        ├──────────────────────────────►│
        │                               │
        │◄──────────────────────────────┤
        │         JSON response         │
        │                               │
   [Vercel]                        [Render]
```

- Communication via API REST uniquement
- Pas de dépendance de code partagé
- ALICE lit MongoDB en lecture seule (données scrapées par chess-app)

---

## Domaine Métier : Échecs Français

### Compétitions Interclubs
- Matchs par équipes entre clubs de la FFE (Fédération Française des Échecs)
- Niveaux : N1 (élite) à N6 (départemental)
- Saison : Septembre à Mars (~6 mois)
- Format : 8 joueurs par équipe typiquement

### Règles Clés
- **Ordre Elo décroissant** : Le joueur avec le meilleur Elo joue à l'échiquier 1
- **Licence A/B** : Types de licences FFE
- **Mutés** : Joueurs transférés d'un autre club (restrictions spéciales)
- **Formés au club** : Joueurs ayant grandi dans le club (bonus)

### Système Elo
- Score numérique représentant le niveau (1000-2800 environ)
- Formule de probabilité : P(victoire) = 1 / (1 + 10^((Elo_adv - Elo_mon)/400))
- Un match nul vaut ~0.35 points en moyenne

### Identifiants FFE
- **clubId** : Code du club (ex: "2976" pour Allauch Hay Chess Club)
- **ffeId** : Identifiant joueur (format "A1234567" ou "B1234567")

---

## Objectif d'ALICE

### Problème Résolu
Les capitaines d'équipe doivent composer leur équipe sans savoir qui l'adversaire va aligner. Actuellement, ils font des suppositions basées sur leur expérience.

### Solution
1. **ALI** (Adversarial Lineup Inference) : Prédire la composition adverse probable
2. **CE** (Composition Engine) : Optimiser sa propre composition pour maximiser le score attendu
3. **Explorer** plusieurs scénarios avec probabilités

### Différenciateur Commercial
Aucun outil concurrent n'offre ces prédictions. ALICE est le cœur de la valeur ajoutée du SaaS chess-app.

---

## Choix Technologiques Justifiés

### Pourquoi XGBoost (et pas un LLM)
- **Données tabulaires** : XGBoost excelle sur ce type de données
- **Explicabilité** : On peut comprendre pourquoi ALICE prédit
- **Vitesse** : Inférence en millisecondes
- **Ressources** : Fonctionne sur free tier Render

### Pourquoi FastAPI
- Standard moderne pour APIs Python
- Documentation auto-générée (Swagger)
- Validation native avec Pydantic
- Async natif (bon pour I/O MongoDB)

### Pourquoi Projet Séparé
- Langages différents (Python vs TypeScript)
- Déploiement indépendant
- Développement découplé
- Tests isolés

---

## Contraintes Absolues

### Budget : 0€
- Render free tier uniquement
- MongoDB Atlas existant (partagé avec chess-app, lecture seule)
- Pas de service payant

### Sécurité
- Isolation par clubId (multi-tenant)
- Pas d'accès aux données d'autres clubs
- API key pour endpoints sensibles (/train)

### Performance
- Inférence < 1 seconde
- Cold start Render acceptable (< 30s) avec fallback gracieux côté chess-app

---

## Fichiers de Référence

### Dans ce projet Claude
1. `CDC_ALICE.md` - Cahier des charges complet
2. `INSTRUCTIONS_PROJET.md` - Ce fichier (contexte pour l'IA)
3. `CONTEXTE_INTEGRATION.md` - Détails sur l'intégration avec chess-app
4. `API_CONTRACT.md` - Spécification détaillée de l'API

### Liens avec chess-app
- Le moteur de règles Flat-Six reste dans chess-app
- Les données MongoDB sont scrapées par chess-app
- L'UI d'appel à ALICE sera dans chess-app (bouton "Demander à Alice")

---

## Comment Travailler sur ce Projet

### Approche Recommandée

1. **Diagnostic d'abord** : Toujours comprendre l'état actuel avant de coder
2. **Un fichier à la fois** : Créer/modifier un seul fichier par échange
3. **Tester localement** : Valider avant de passer au suivant
4. **Commit propre** : Une fonctionnalité testée = un commit

### Commandes Fréquentes (Windows/PowerShell)

```powershell
# Se placer dans le projet
cd C:\Dev\alice-engine

# Créer environnement virtuel
python -m venv venv

# Activer l'environnement
.\venv\Scripts\Activate

# Installer dépendances
pip install -r requirements.txt

# Lancer le serveur local
uvicorn app.main:app --reload

# Tester l'API
curl http://localhost:8000/health
```

### Structure de Réponse Attendue

Quand Pierre-Alexandre demande de créer un fichier :

1. **Expliquer** brièvement ce que fait le fichier
2. **Donner le chemin exact** : `C:\Dev\chess-predictor\app\main.py`
3. **Fournir le code complet** (pas de "..." ou extraits)
4. **Indiquer la prochaine étape** concrète

---

## Points de Vigilance

### Ne PAS faire
- Supposer que des packages sont installés sans vérifier
- Combiner plusieurs fichiers dans une réponse
- Utiliser du jargon sans l'expliquer
- Proposer des solutions qui contournent la sécurité
- Être "optimiste" sur l'état du code

### À faire
- Poser des questions si le contexte manque
- Valider la compréhension avant de coder
- Proposer des tests pour chaque fonctionnalité
- Documenter les choix techniques
- Signaler les risques ou limitations

---

## Historique et Décisions

### 30 décembre 2025 - Création du projet
- Décision : Projet Python séparé de chess-app
- Nom choisi : **ALICE** (Adversarial Lineup Inference & Composition Engine)
- Justification : Prénom naturel, usage fluide ("J'ai fait les compos avec Alice")
- Communication : API REST (POST /predict)
- Repo : `alice-engine`

---

## Questions Fréquentes

**Q : Pourquoi "ALICE" ?**
R : Acronyme de Adversarial Lineup Inference & Composition Engine. Usage naturel : "J'ai fait les compos avec Alice".

**Q : Où sont les données historiques ?**
R : Dans MongoDB Atlas, scrapées par chess-app. ALICE y accède en lecture seule.

**Q : Comment tester sans les vraies données ?**
R : On créera des fixtures/mocks pour les tests locaux.

**Q : Faut-il toucher à chess-app maintenant ?**
R : Non. ALICE est autonome. L'intégration viendra après.

**Q : Quel modèle XGBoost utiliser ?**
R : XGBClassifier pour l'ALI (prédiction présence), puis calculs Elo pour le CE (composition).
