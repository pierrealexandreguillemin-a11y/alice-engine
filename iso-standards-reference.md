# ISO Standards Reference — ALICE Engine

## Normes Actives

| Norme | Focus | Priorité |
|-------|-------|----------|
| **ISO 27001** | ISMS, gestion risques sécurité | 🔴 Critique |
| **ISO 27034** | Secure coding, OWASP | 🔴 Critique |
| **ISO 5055** | Qualité code, CWE auto | 🔴 Critique |
| **ISO 25010** | Qualité système (FURPS+) | 🟠 Important |
| **ISO 25012** | Qualité données | 🟠 Important |
| **ISO 25019** | SaaS/Cloud | 🟠 Important |
| **ISO 29119** | Tests logiciels | 🟠 Important |
| **ISO 42010** | Architecture | 🟠 Important |
| **ISO 12207** | Cycle de vie | 🟡 Utile |
| **ISO 90003** | Qualité processus | 🟡 Utile |
| **ISO 15289** | Contenu documentation cycle de vie | 🟠 Important |
| **ISO 26514** | Information utilisateur logiciel | 🟠 Important |
| **ISO 26515** | Documentation en environnement agile | 🟡 Utile |
| **ISO 25065** | UX/Accessibilité | 🟡 Utile |

---

## Documentation (ISO 15289 + ISO 26514)

### Structure docs/ conforme ISO 15289

```
docs/
├── architecture/       # ISO 42010 - Architecture Description
│   ├── ARCHITECTURE.md         # Vue d'ensemble architecture
│   ├── DATA_MODEL.md           # Modèle de données
│   └── DECISIONS.md            # ADR (Architecture Decision Records)
│
├── api/                # ISO 26514 - Information for Users (API)
│   └── API_CONTRACT.md         # Contrat API OpenAPI
│
├── requirements/       # ISO 15289 - Requirements Specification
│   ├── CDC_ALICE.md            # Cahier des charges
│   └── CONTEXTE_*.md           # Contextes métier
│
├── operations/         # ISO 15289 - Operations Documentation
│   ├── DEPLOIEMENT_RENDER.md   # Guide déploiement
│   └── MAINTENANCE.md          # Procédures maintenance
│
├── development/        # ISO 15289 - Development Documentation
│   ├── CONTRIBUTING.md         # Guide contribution
│   └── PYTHON-HOOKS-SETUP.md   # Setup développeur
│
├── iso/                # ISO 15289 - Quality Records
│   └── IMPLEMENTATION_STATUS.md # Auto-généré
│
└── project/            # ISO 15289 - Project Documentation
    ├── ANALYSE_INITIALE_ALICE.md # Analyse initiale
    └── CHANGELOG.md              # Journal des modifications
```

### Types de documents ISO 15289

| Type | Code | Exemples ALICE |
|------|------|----------------|
| **Concept of Operations** | ConOps | CDC_ALICE.md |
| **System Requirements** | SyRS | CONTEXTE_*.md |
| **Architecture Description** | AD | ARCHITECTURE.md |
| **Interface Design** | IDD | API_CONTRACT.md |
| **Software User Documentation** | SUD | README.md |
| **Operations Manual** | OpsMan | DEPLOIEMENT_RENDER.md |
| **Quality Records** | QR | IMPLEMENTATION_STATUS.md |

### Contenu minimal par document (ISO 26514)

Chaque document technique doit contenir :

1. **En-tête**
   - Titre
   - Version
   - Date dernière mise à jour
   - Auteur/Responsable

2. **Introduction**
   - Objectif du document
   - Audience cible
   - Prérequis

3. **Corps**
   - Contenu structuré avec titres hiérarchiques
   - Exemples de code si applicable
   - Schémas/diagrammes si nécessaire

4. **Références**
   - Documents liés
   - Normes applicables

---

## Architecture SRP

```
feature/
├── feature.controller.ts   # Routes, HTTP uniquement
├── feature.service.ts      # Logique métier pure
├── feature.repository.ts   # Accès données uniquement
├── feature.validator.ts    # Validation Zod
├── feature.types.ts        # Interfaces/Types
└── feature.test.ts         # Tests unitaires
```

**Règles :**
- 1 fichier = 1 responsabilité
- Controller → Service → Repository (jamais l'inverse)
- Service = pur, testable, sans I/O direct
- Repository = seul à toucher MongoDB

---

## Pyramide de Tests

```
        ╱╲
       ╱E2E╲         5%  — Playwright (flux critiques)
      ╱──────╲
     ╱ Intég. ╲     15%  — API routes, DB memory
    ╱──────────╲
   ╱  Unitaires ╲   80%  — Vitest, logique pure
  ╱──────────────╲
```

| Type | Cible | Outil |
|------|-------|-------|
| Unit | Services, Flat-Six rules | Vitest |
| Intégration | Routes API, DB | Supertest + mongodb-memory-server |
| E2E | Workflows complets | Playwright |
| Accessibilité | WCAG 2.1 AA | axe-core |
| Sécurité | OWASP Top 10 | ZAP DAST |

---

## Checklist Sécurité (27034)

- [ ] Input validation (Zod)
- [ ] Output encoding
- [ ] Auth/Authz chaque route
- [ ] SQL/NoSQL injection (mongo-sanitize)
- [ ] XSS (Helmet CSP)
- [ ] CSRF tokens
- [ ] Rate limiting
- [ ] Secrets en env vars
- [ ] Logs sans données sensibles
- [ ] Dépendances à jour

---

## Qualité Code (5055)

```bash
# Avant chaque commit
npm run lint          # 0 erreurs
npm run typecheck     # 0 erreurs
npm run test          # 100% pass
npm audit             # 0 critical/high
```

**CWE prioritaires :**
- CWE-89: Injection
- CWE-79: XSS
- CWE-287: Auth bypass
- CWE-522: Credentials faibles
- CWE-798: Hardcoded secrets

---

## Qualité Données (25012)

| Critère | Implémentation |
|---------|----------------|
| Exactitude | Validation Zod stricte |
| Complétude | Required fields + defaults |
| Cohérence | Transactions MongoDB |
| Unicité | Indexes unique (ffeId, email) |
| Traçabilité | createdAt, updatedAt, audit logs |

---

## Multi-tenant (25019)

```typescript
// TOUJOURS filtrer par clubId
const players = await Player.find({ clubId: req.user.clubId });

// JAMAIS
const players = await Player.find({ _id: id }); // ❌ Fuite données
```

---

## JSDoc Standard

```typescript
/**
 * @description Valide composition via Flat-Six
 * @param {string} compositionId - ID MongoDB
 * @param {string} clubId - Isolation tenant
 * @returns {Promise<ValidationResult>}
 * @throws {NotFoundError} Composition inexistante
 * @throws {ValidationError} Règles FFE violées
 * @see ISO 25010 - Fiabilité
 */
```

---

## Commandes Rapides

```bash
# Dev
npm run dev

# Tests
npm run test              # Unitaires
npm run test:integration  # Intégration
npm run test:e2e          # E2E
npm run test:coverage     # Couverture

# Qualité
npm run lint
npm run typecheck
npm audit
npm run security:check    # Snyk

# Build
npm run build
```

---

## Contexte Hostile — Rappels

- **Mineurs** → RGPD renforcé, consentement parental
- **Paiements** → PCI-DSS awareness
- **Concurrence** → Obfuscation, rate limiting agressif
- **Attaques** → WAF, monitoring, alertes
- **Données FFE** → Scraping légal, cache, respect robots.txt

---

*Dernière MAJ: 2025-01-02 | Chess App v1.4.2*
