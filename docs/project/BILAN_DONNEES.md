# Bilan des Donnees FFE Scrapees

## Vue d'Ensemble

| Metrique | Valeur |
|----------|--------|
| **Periode couverte** | 2002 - 2026 (25 saisons) |
| **Groupes scraped** | 15,148 |
| **Fichiers HTML compositions** | 84,789 |
| **Fichiers HTML joueurs** | 2,159 |
| **Total fichiers** | 86,948 |
| **Joueurs licencies** | 66,208 |
| **Taux de succes** | 100% |

---

## Joueurs FFE v2 (COMPLET - 4 Janvier 2026)

| Metrique | Valeur |
|----------|--------|
| **Clubs scraped** | 992 |
| **Pages scrapees** | 2,159 |
| **Joueurs total** | 66,208 |
| **Taille** | ~120 MB |
| **Stockage** | `data/players_v2/club_XXX/page_XX.html` |

### Comparaison v1 vs v2

| Metrique | v1 (FIDE only) | v2 (COMPLET) | Amelioration |
|----------|----------------|--------------|--------------|
| Pages | 883 | 2,159 | +145% |
| Joueurs | 35,320 | 66,208 | +87% |
| Jeunes (U08-U14) | 2,246 | 33,896 | +1,410% |
| Couverture FFE | 53% | 100% | +47pp |

### Distribution par type d'Elo

| Type | Count | % |
|------|-------|---|
| Estime (E) | 47,746 | 72.1% |
| FIDE (F) | 17,494 | 26.4% |
| National (N) | 965 | 1.5% |

### Distribution par categorie

| Categorie | Count | % |
|-----------|-------|---|
| Jeunes (U08-U14) | 33,896 | 51.2% |
| Adolescents (U16-U20) | 5,278 | 8.0% |
| Adultes (X20-X65) | 27,034 | 40.8% |

### Donnees par Joueur

| Champ | Description |
|-------|-------------|
| NrFFE | Numero FFE unique (ex: K59857) |
| Nom | Nom complet |
| Affiliation | A (complete) ou B (partielle) |
| Elo | Classement standard + type (F/N/E) |
| Rapide | Classement rapide + type |
| Blitz | Classement blitz + type |
| Categorie | PpoM, PouF, SenM, VetF, etc. |
| Mute | Transfere d'un autre club (M) |
| Club | Nom du club |

### Parsing v2 (4 Janvier 2026)

| Metrique | Valeur |
|----------|--------|
| **Pages parsees** | 2,159 |
| **Joueurs extraits** | 66,208 |
| **Doublons nr_ffe** | 0 (unicite OK) |
| **Fichier Parquet** | `data/joueurs.parquet` (3.0 MB) |
| **Colonnes** | 19 |

#### Validation ISO 25012

| Critere | Statut | Details |
|---------|--------|---------|
| Exactitude | ✅ | Types corrects (int64, object, bool) |
| Completude | ✅ | Nulls attendus: age_min (U8), age_max (S65) |
| Unicite | ✅ | 0 doublons nr_ffe sur 66,208 |
| Coherence | ✅ | Categories mappees vers codes FFE officiels |

#### Distribution par categorie (parsing)

| Categorie | Count |
|-----------|-------|
| SenM | 11,579 |
| PouM | 8,140 |
| PupM | 7,915 |
| PpoM | 5,476 |
| SepM | 5,457 |
| BenM | 5,081 |
| VetM | 4,761 |
| MinM | 3,603 |
| PouF | 2,435 |
| PupF | 2,091 |
| Autres | 10,270 |

---

## Donnees par Saison

| Saison | Groupes | Fichiers | Calendriers | Rondes | Observations |
|--------|---------|----------|-------------|--------|--------------|
| 2026 | 710 | 3,974 | 710 | 3,264 | Saison en cours |
| 2025 | 825 | 4,720 | 825 | 3,895 | Complete |
| 2024 | 762 | 4,445 | 762 | 3,683 | Complete |
| 2023 | 693 | 4,032 | 693 | 3,339 | Complete |
| 2022 | 712 | 3,857 | 712 | 3,145 | Complete |
| 2021 | 652 | 2,316 | 646 | 1,670 | COVID - competitions reduites |
| 2020 | 77 | 692 | 77 | 615 | COVID - saison tronquee |
| 2019 | 762 | 4,179 | 756 | 3,423 | Complete |
| 2018 | 752 | 4,208 | 752 | 3,456 | Complete |
| 2017 | 745 | 4,011 | 693 | 3,318 | Complete |
| 2016 | 747 | 4,045 | 697 | 3,348 | Complete |
| 2015 | 736 | 4,122 | 705 | 3,417 | Complete |
| 2014 | 685 | 3,921 | 670 | 3,251 | Complete |
| 2013 | 633 | 3,564 | 608 | 2,956 | Complete |
| 2012 | 662 | 3,658 | 625 | 3,033 | Complete |
| 2011 | 638 | 3,669 | 624 | 3,045 | Complete |
| 2010 | 631 | 3,664 | 608 | 3,056 | Complete |
| 2009 | 663 | 3,783 | 646 | 3,137 | Complete |
| 2008 | 635 | 3,651 | 615 | 3,036 | Complete |
| 2007 | 614 | 3,660 | 614 | 3,046 | Complete |
| 2006 | 599 | 3,571 | 599 | 2,972 | Complete |
| 2005 | 525 | 2,731 | 525 | 2,206 | Moins de competitions |
| 2004 | 394 | 2,262 | 394 | 1,868 | Moins de competitions |
| 2003 | 255 | 1,696 | 255 | 1,441 | Moins de competitions |
| 2002 | 41 | 357 | 41 | 316 | Premiere saison numerisee |

---

## Types de Competitions

### Interclubs Adultes (Nationales)
- Top 16, Nationale I, II, III, IV
- ~8 echiquiers par equipe
- ~7-11 rondes par saison

### Interclubs Regionaux
- Regionales, Departementales
- 34 ligues regionales
- ~4-8 echiquiers par equipe

### Coupes
- Coupe de France
- Coupe Jean-Claude Loubatiere
- Coupe de la Parite
- Coupe 2000

### Interclubs Jeunes
- Nationales Jeunes
- Competitions scolaires (UNSS)
- ~4-6 echiquiers

### Interclubs Feminins
- Nationale Feminine
- Divisions regionales

---

## Structure des Fichiers HTML

### calendrier.html
```
Contient:
- Hierarchie: Ligue / Division / Groupe
- Liste des rondes avec dates
- Resultats globaux des matchs (ex: 3-5)
- Lieux des rencontres

Taille: 1.5 KB - 60 KB
```

### ronde_N.html
```
Contient:
- Score global du match
- Pour chaque echiquier:
  - Numero (1-8)
  - Joueur Blancs (Nom, Elo)
  - Joueur Noirs (Nom, Elo)
  - Resultat (1-0, 0-1, 1/2-1/2, F)

Taille: 15 KB - 55 KB
```

---

## Estimations de Donnees Parsees

| Element | Estimation |
|---------|------------|
| **Matchs** | ~100,000 |
| **Echiquiers** | ~600,000 - 800,000 |
| **Joueurs licencies** | 66,208 |
| **Parties** | ~600,000 - 800,000 |

---

## Qualite des Donnees

### Points Positifs
- [x] Couverture complete 2002-2026
- [x] Tous les niveaux (Top 16 aux departementales)
- [x] Elo des joueurs disponibles
- [x] Dates et lieux des matchs

### Limitations Connues
- [ ] 2020 tres incomplet (COVID)
- [ ] Anciennes saisons (2002-2004) moins de competitions
- [ ] Quelques forfaits notes "F" sans details
- [ ] Elo parfois manquant pour joueurs non licencies

---

## Utilisation pour ALICE (XGBoost)

### Features Extractibles

```python
# Par echiquier
features = {
    'saison': int,
    'ronde': int,
    'echiquier': int,  # 1-8
    'elo_blanc': int,
    'elo_noir': int,
    'diff_elo': int,   # elo_blanc - elo_noir
    'resultat': float, # 1.0, 0.5, 0.0
    'couleur': str,    # 'B' ou 'N'

    # Contexte equipe
    'niveau_competition': str,  # N1, N2, R1, D1...
    'score_equipe_avant': float,
    'classement_equipe': int,
}
```

### Volume Estime pour Training

| Dataset | Lignes |
|---------|--------|
| Echiquiers (2002-2026) | ~750,000 |
| Echiquiers (2015-2026) | ~450,000 |
| Echiquiers (2020-2026) | ~180,000 |

---

## Stockage

### Taille sur Disque

```
data/              ~2.1 GB (HTML compositions equipes)
data/players_v2/   ~120 MB (HTML joueurs FFE - COMPLET)
data/players/      ~49 MB (HTML joueurs FIDE - OBSOLETE)
data/clubs/        ~200 KB (Index clubs JSON)
db/                ~15 MB (SQLite metadata)
exports/           (a generer - ~200 MB CSV)
```

### Backup Recommande

```bash
# Compression des donnees
tar -czvf ffe_data_backup_2026-01-01.tar.gz data/ db/

# Taille estimee: ~400 MB compresse
```

---

## Prochaines Etapes

1. **parse.py** : Extraction des compositions d'equipes
2. **parse_players_v2.py** : Extraction des donnees joueurs (nouveau format)
3. **Normalisation** : Noms joueurs, clubs
4. **Export CSV/Parquet** : Format ALICE
5. **Validation** : Echantillon manuel 5%
6. **Nettoyage** : Supprimer `data/players/` (v1 obsolete)

---

*Bilan mis a jour le 4 Janvier 2026*
