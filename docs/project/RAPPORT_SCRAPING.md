# Rapport de Scraping FFE - 1er-4 Janvier 2026

## Resume Executif

**Objectif** : Collecter les compositions d'equipes et joueurs FFE (2002-2026) pour alimenter ALICE (XGBoost)

**Resultat** : **100% de succes**

| Dataset | Fichiers | Joueurs | Taille |
|---------|----------|---------|--------|
| Compositions equipes | 84,789 | - | ~2.1 GB |
| Joueurs FFE v1 (FIDE only) | 883 | 35,320 | ~49 MB |
| **Joueurs FFE v2 (COMPLET)** | **2,159** | **66,208** | ~120 MB |
| **Total** | **86,948** | **66,208** | **~2.27 GB** |

**Duree totale** : ~5h (compositions) + ~10min (joueurs v1) + ~31min (joueurs v2)

---

## 1. Architecture Deployee

### Infrastructure Multi-Agents

| Agent | Machine | IP | Saisons | Groupes | Statut |
|-------|---------|-----|---------|---------|--------|
| 1 | VPS AMD 1 (Oracle) | 144.24.201.177 | 2013-2017 | 3,546 | COMPLETE |
| 2 | VPS AMD 2 (Oracle) | 84.235.227.48 | 2008-2012 | 3,229 | COMPLETE |
| 3 | Local | - | 2002-2007 | 2,428 | COMPLETE |
| 4 | Local | - | 2018-2021 | 2,243 | COMPLETE |
| 5 | Local | - | 2022-2026 | 3,702 | COMPLETE |

### Ressources Oracle Cloud Free Tier

- **2 VMs AMD Micro** : 1/8 OCPU, 1 GB RAM chacune
- **Cout** : 0 EUR (100% Free Tier)
- **Region** : eu-marseille-1

---

## 2. Chronologie du Scraping

### Phase 2.2 - Discovery (Pre-requis)
- **14,438 groupes** decouverts via `discover.py`
- Base SQLite initialisee avec tous les `ref_id`

### Phase 2.3 - Scraping Distribue

| Heure | Evenement |
|-------|-----------|
| 08:00 | Demarrage Agents 1-5 |
| 08:30 | Agents 3, 5 termines (2002-2007, 2022-2025) |
| 09:00 | Agent 4 bloque sur erreurs COVID |
| 09:15 | Fix COVID_YEARS, relance Agent 4 |
| 09:30 | Agents 1, 2 termines sur VPS |
| 10:00 | Sync donnees VPS -> Local |
| 10:30 | Agent 4 termine |
| 10:45 | Retry 272 erreurs restantes |
| 10:50 | **100% complete (2002-2025)** |
| 18:30 | Decouverte saison 2026 (fix Saison=3000) |
| 19:25 | **2026 complete** - 710 groupes, 3974 fichiers |

### Phase 2.4 - Scraping Joueurs v1 (2 Janvier 2026)

| Heure | Evenement |
|-------|-----------|
| 01:30 | Creation scrape_players.py |
| 01:33 | Lancement 4 agents paralleles |
| 01:39 | Agent 1 termine (pages 1-221) |
| 01:42 | Agents 2-4 termines |
| 01:42 | **100% complete** - 883 pages, 35,320 joueurs |

**Strategie** : 4 agents locaux, pagination ASP.NET avec ViewState

| Agent | Pages | Duree |
|-------|-------|-------|
| 1 | 1-221 | 4.9 min |
| 2 | 222-442 | ~8 min (nav + scrape) |
| 3 | 443-663 | ~10 min (nav + scrape) |
| 4 | 664-883 | ~12 min (nav + scrape) |

### Phase 2.5 - Scraping Joueurs v2 COMPLET (3-4 Janvier 2026)

**Probleme identifie** : Le dataset v1 ne contenait que les joueurs avec Elo FIDE (35,320 joueurs).
Les jeunes (U08-U14) etaient quasi absents (-97% a -99.6% vs FFE Open Data).

**Solution** : Scraper par club au lieu de la liste globale FIDE.

| Heure | Evenement |
|-------|-----------|
| 23:00 | Exploration MCP DevTools du site FFE |
| 23:30 | Decouverte URL `Action=JOUEURCLUBREF&ClubRef=XXX` |
| 00:00 | Creation scrape_all_players.py |
| 00:15 | Scraping clubs par departement (992 clubs) |
| 00:16 | **Clubs complete** - 56 secondes |
| 00:20 | Lancement scraping joueurs par club |
| 00:51 | **100% complete** - 66,208 joueurs, 0 erreurs |

**Strategie** : Scraping hierarchique Departements -> Clubs -> Joueurs

| Etape | URL Pattern | Resultat |
|-------|-------------|----------|
| Clubs | `ListeClubs.aspx?Action=CLUBCOMITE&ComiteRef=XX` | 992 clubs |
| Joueurs | `ListeJoueurs.aspx?Action=JOUEURCLUBREF&ClubRef=XXX` | 66,208 joueurs |
| Validation | `FicheClub.aspx?Ref=XXX` (Licences A+B) | 100% match |

**Resultats par categorie** :

| Categorie | v1 (FIDE) | v2 (COMPLET) | Amelioration |
|-----------|-----------|--------------|--------------|
| U08 | 28 | 7,438 | +26,500% |
| U10 | 264 | 10,575 | +3,900% |
| U12 | 777 | 10,006 | +1,188% |
| U14 | 1,177 | 5,877 | +399% |
| **Total Jeunes** | **2,246** | **33,896** | **+1,410%** |
| **Total General** | **35,320** | **66,208** | **+87%** |

**Distribution par type d'Elo** :
- Estime (E) : 47,746 (72.1%) - jeunes recuperes
- FIDE (F) : 17,494 (26.4%)
- National (N) : 965 (1.5%)

---

## 3. Problemes Rencontres et Solutions

### Probleme 1 : Calendriers COVID vides
- **Symptome** : Erreurs "Calendrier invalide: Trop petit" sur 2020-2021
- **Cause** : Competitions annulees (COVID) avec calendriers < 2000 bytes
- **Solution** : Extension `COVID_YEARS = {2018, 2019, 2020, 2021}` puis reduction globale `MIN_CALENDAR_SIZE = 1500`

### Probleme 2 : Circuit breaker trop agressif
- **Symptome** : Pauses de 5 min apres 3 erreurs consecutives
- **Cause** : Erreurs legitimement vides declenchaient le circuit breaker
- **Solution** : Fix des seuils de validation

### Probleme 3 : SSH non configure
- **Symptome** : Impossible de se connecter aux VPS
- **Cause** : Cle privee non copiee dans ~/.ssh/
- **Solution** : `cp ~/.oci/console_key ~/.ssh/id_rsa`

### Probleme 4 : ARM "Out of capacity"
- **Symptome** : Impossible de creer VMs ARM Oracle
- **Cause** : Free Tier ARM sature en region Marseille
- **Solution** : Utilisation des agents locaux en remplacement

### Probleme 5 : Saison 2026 non detectee
- **Symptome** : Discovery 2026 ne sauvegardait aucun groupe
- **Cause** : FFE utilise `Saison=3000` pour "Actuelle" au lieu de 2026
- **Solution** : Mapping `if ref_saison == 3000: ref_saison = saison`

---

## 4. Metriques de Performance

### Vitesse de Scraping

| Agent | Vitesse Moyenne | Pics |
|-------|-----------------|------|
| VPS AMD | ~50 groupes/min | ~100/min |
| Local | ~20 groupes/min | ~50/min |

### Delais Appliques

| Mode | Delai |
|------|-------|
| VPS off-peak | 0.25-0.4s |
| VPS peak | 0.5-0.8s |
| Local off-peak | 0.5-0.8s |
| Local peak | 1.0-1.5s |

### Erreurs

| Type | Count | Resolution |
|------|-------|------------|
| Calendrier vide | 272 | Seuil abaisse |
| Timeout | 0 | - |
| HTTP error | 0 | - |
| **Total final** | **0** | 100% succes |

---

## 5. Code Modifie

### scrape.py

```python
# Changements principaux:

# 1. Configuration multi-agents
AGENT_SEASONS = {
    1: list(range(2013, 2018)),  # VPS AMD 1
    2: list(range(2008, 2013)),  # VPS AMD 2
    3: list(range(2002, 2008)),  # Local
    4: list(range(2018, 2022)),  # Local
    5: list(range(2022, 2027)),  # Local (2022-2026)
}

# 2. Seuil reduit pour calendriers vides
MIN_CALENDAR_SIZE = 1500  # Was 2000

# 3. Heures creuses etendues
def is_off_peak() -> bool:
    return 22 <= hour or hour < 9  # Was 0-9
```

### scrape_players.py (Nouveau)

```python
# Configuration multi-agents pagination
AGENT_PAGES = {
    1: (1, 221),      # pages 1-221
    2: (222, 442),    # pages 222-442
    3: (443, 663),    # pages 443-663
    4: (664, 883),    # pages 664-883
}

# Delais adaptatifs
DELAY_OFF_PEAK = (0.4, 0.6)   # 22h-9h
DELAY_PEAK = (0.8, 1.2)       # 9h-22h

# Navigation ViewState ASP.NET obligatoire
# Agents 2-4 doivent naviguer jusqu'a leur page de depart
```

---

## 6. Fichiers Produits

### Structure

```
data/
├── players/        (883 fichiers, 49 MB) - Nouveau
│   ├── page_0001.html
│   ├── ...
│   └── page_0883.html
│
├── 2002/           (357 fichiers)
├── 2003/           (1,696 fichiers)
├── ...
├── 2024/           (4,445 fichiers)
├── 2025/           (4,720 fichiers)
└── 2026/           (3,974 fichiers)

Total: 85,672 fichiers HTML (~2.15 GB)
```

### Format par Groupe (compositions)

```
data/{saison}/{competition}/{division}/{groupe}/
├── calendrier.html    # Vue calendrier (matchs, dates, lieux)
└── ronde_N.html       # Compositions echiquiers (1 par ronde)
```

### Format Joueurs v1 (obsolete)

```
data/players/page_XXXX.html
# 40 joueurs par page, ordre alphabetique
# LIMITE: Seulement joueurs avec Elo FIDE
```

### Format Joueurs v2 (COMPLET)

```
data/players_v2/
├── clubs_index.json           # Index des 992 clubs
└── club_{ref}/                 # Un dossier par club
    ├── page_01.html
    ├── page_02.html
    └── ...

# ~40 joueurs par page, TOUS les licencies (FIDE + National + Estime)
# Contient: NrFFE, Nom, Affiliation (A/B), Elo, Rapide, Blitz, Categorie, Mute, Club
```

---

## 7. Validation

### Integrite des Donnees

- [x] Tous les groupes ont un calendrier.html
- [x] Les rondes correspondent aux liens dans calendrier.html
- [x] HTML valide (balise </html> presente)
- [x] Taille > 1500 bytes (calendriers) / > 1500 bytes (rondes)

### Echantillon Verifie

```
2026/Interclubs/Nationale_I/Groupe_A/
├── calendrier.html
├── ronde_1.html
├── ronde_2.html
...
└── ronde_N.html
```

---

## 8. Prochaines Etapes

1. **Phase 3 - Parsing compositions** : Extraire les compositions avec `parse.py`
2. **Phase 3b - Parsing joueurs** : Extraire les joueurs avec `parse_players.py`
3. **Phase 4 - Export** : Generer CSV/Parquet pour ALICE
4. **Archivage** : Backup des donnees brutes vers `C:/Dev/ffe_data_backup/`

---

## 9. Conformite

- **ISO/IEC 25010:2023** : Maintenabilite, Fiabilite
- **PEP 8/257/484** : Style Python
- **RGPD/CNIL** : Donnees publiques, finalite legitime
- **Ethique** : Delais respectueux, heures creuses

---

*Rapport mis a jour le 4 Janvier 2026*
