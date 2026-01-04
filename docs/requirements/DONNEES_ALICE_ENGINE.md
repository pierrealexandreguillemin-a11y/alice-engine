# Donnees FFE pour ALICE-Engine (XGBoost)

> **Document de reference** : Ce document decrit l'ensemble des donnees disponibles
> suite au scraping FFE 2002-2026 pour alimenter le modele XGBoost de prediction
> des compositions d'equipes.

---

## 1. Vue d'Ensemble du Dataset

### 1.1 Volumetrie

| Metrique | Valeur |
|----------|--------|
| **Periode** | 2002 - 2026 (25 saisons) |
| **Poids brut compositions** | ~2.1 GB (HTML) |
| **Poids brut joueurs v2** | ~120 MB (HTML) |
| **Fichiers HTML compositions** | 84,789 |
| **Fichiers HTML joueurs** | 2,159 (992 clubs) |
| **Groupes/Poules** | 15,148 |
| **Calendriers** | ~15,000 |
| **Pages rondes** | ~70,000 |
| **Joueurs FFE (licencies)** | 66,208 |
| **Backup** | `C:/Dev/ffe_data_backup/` |

### 1.2 Estimations Donnees Parsees

| Element | Estimation |
|---------|------------|
| **Matchs** | ~110,000 |
| **Echiquiers/Parties** | ~650,000 - 850,000 |
| **Joueurs licencies** | 66,208 |

---

## 2. Structure des Donnees Disponibles

### 2.1 Hierarchie des Fichiers

```
data/
├── clubs/
│   └── clubs_index.json               # Index 992 clubs (ref, nom, dept, licences)
│
├── players_v2/                         # COMPLET - 4 Janvier 2026
│   └── club_{ref}/                     # 992 dossiers clubs
│       └── page_XX.html                # ~40 joueurs/page, tous licencies
│
├── players/                            # OBSOLETE - FIDE only
│   └── page_XXXX.html                  # 883 pages (35,320 joueurs FIDE)
│
└── {saison}/                           # 2002-2026
    └── {competition}/                  # Interclubs, Coupe_de_France, Ligue_XXX
        └── {division}/                 # Nationale_1, Regionale_2, etc.
            └── {groupe}/               # Groupe_A, Poule_1, etc.
                ├── calendrier.html     # Dates, lieux, scores globaux
                └── ronde_N.html        # Compositions detaillees
```

### 2.2 Extraction Automatique depuis Arborescence

| Champ | Source | Exemple | Type |
|-------|--------|---------|------|
| `saison` | Dossier niveau 1 | `2025` | int |
| `competition` | Dossier niveau 2 | `Interclubs` | str |
| `division` | Dossier niveau 3 | `Nationale 1` | str |
| `groupe` | Dossier niveau 4 | `Groupe A` | str |
| `niveau` | Extrait de division | `1` | int |
| `type_competition` | Deduit | `National` ou `Regional` | str |
| `ligue` | Si regional | `Hauts-De-France` | str |
| `ligue_code` | Mapping | `HDF` | str |

---

## 3. Donnees par Fichier Source

### 3.1 calendrier.html

| Champ | Description | Exemple |
|-------|-------------|---------|
| `ronde` | Numero de la ronde | `1` |
| `equipe_dom` | Equipe a domicile (blancs) | `Cappelle-La-Grande` |
| `equipe_ext` | Equipe exterieure (noirs) | `Lille Universite Club` |
| `score_dom` | Score equipe domicile | `4` |
| `score_ext` | Score equipe exterieure | `1` |
| `date` | Date et heure (datetime) | `2024-10-12 16:00` |
| `date_str` | Format original | `samedi 12/10/24 16:00` |
| `heure` | Heure de debut | `16:00` |
| `jour_semaine` | Jour | `Samedi` |
| `lieu` | Ville du match | `LILLE` |

### 3.2 ronde_N.html - Niveau Match

| Champ | Description | Exemple |
|-------|-------------|---------|
| `ronde` | Numero ronde | `1` |
| `equipe_dom` | Equipe domicile | `Cappelle-La-Grande` |
| `equipe_ext` | Equipe exterieure | `Lille Universite Club` |
| `score_dom` | Points equipe dom | `4` |
| `score_ext` | Points equipe ext | `1` |
| `echiquiers[]` | Liste des 4-8 echiquiers | (voir ci-dessous) |

### 3.3 ronde_N.html - Niveau Echiquier

| Champ | Description | Exemple |
|-------|-------------|---------|
| `numero` | Numero echiquier | `1` |
| `blanc.nom` | Nom joueur blancs | `NIKOLOV` |
| `blanc.prenom` | Prenom joueur blancs | `Momchil` |
| `blanc.nom_complet` | Nom complet | `NIKOLOV Momchil` |
| `blanc.elo` | Classement Elo | `2424` |
| `blanc.titre` | Code titre | `g` |
| `blanc.titre_fide` | Titre FIDE | `GM` |
| `noir.nom` | Nom joueur noirs | `LESUEUR` |
| `noir.prenom` | Prenom joueur noirs | `Gabriel` |
| `noir.nom_complet` | Nom complet | `LESUEUR Gabriel` |
| `noir.elo` | Classement Elo | `2256` |
| `noir.titre` | Code titre | `` |
| `noir.titre_fide` | Titre FIDE | `` |
| `resultat_blanc` | Score blancs | `0.5` |
| `resultat_noir` | Score noirs | `0.5` |
| `resultat_text` | Texte original | `X - X` |
| `type_resultat` | Type categorise | `nulle` |
| `diff_elo` | Elo blanc - Elo noir | `+168` |
| `equipe_blanc` | Equipe du joueur blanc | `Cappelle-La-Grande` |
| `equipe_noir` | Equipe du joueur noir | `Lille Universite Club` |

### 3.4 players_v2/club_XXX/page_XX.html - Joueurs FFE (COMPLET)

| Champ | Description | Exemple |
|-------|-------------|---------|
| `nr_ffe` | Numero FFE unique | `K59857` |
| `nom_complet` | Nom et prenom | `AALIOULI Karim` |
| `affiliation` | Type licence | `A` (complete) ou `B` (partielle) |
| `elo` | Classement standard | `1567` |
| `elo_type` | Type Elo | `F` (FIDE), `N` (National), `E` (Estime) |
| `rapide` | Classement rapide | `1500` |
| `rapide_type` | Type Elo rapide | `F`, `N`, `E` |
| `blitz` | Classement blitz | `1500` |
| `blitz_type` | Type Elo blitz | `F`, `N`, `E` |
| `categorie` | Categorie legacy | `PpoM`, `PouF`, `SenM`, `VetF` |
| `mute` | Transfere d'un club | `M` ou vide |
| `club` | Club d'affiliation | `Echiquier de Bigorre` |

### 3.5 Distribution des Joueurs (66,208 total)

| Categorie | Count | % | Type Elo principal |
|-----------|-------|---|-------------------|
| Jeunes U08-U14 | 33,896 | 51.2% | Estime (E) |
| Ados U16-U20 | 5,278 | 8.0% | Estime/National |
| Adultes X20-X65 | 27,034 | 40.8% | FIDE (F) |

| Type Elo | Count | % |
|----------|-------|---|
| Estime (E) | 47,746 | 72.1% |
| FIDE (F) | 17,494 | 26.4% |
| National (N) | 965 | 1.5% |

---

## 4. Titres FIDE Disponibles

| Code HTML | Titre FIDE | Description |
|-----------|------------|-------------|
| `g` | GM | Grand-Maitre |
| `m` | IM | Maitre International |
| `f` | FM | Maitre FIDE |
| `gf` | WGM | Grand-Maitre Feminin |
| `mf` | WIM | Maitre International Feminin |
| `ff` | WFM | Maitre FIDE Feminin |

---

## 5. Types de Resultats

| Valeur | Description | Score B | Score N |
|--------|-------------|---------|---------|
| `victoire_blanc` | Blancs gagnent | 1.0 | 0.0 |
| `victoire_noir` | Noirs gagnent | 0.0 | 1.0 |
| `nulle` | Match nul | 0.5 | 0.5 |
| `forfait_blanc` | Forfait blancs | 0.0 | 1.0 |
| `forfait_noir` | Forfait noirs | 1.0 | 0.0 |
| `inconnu` | Non determine | 0.0 | 0.0 |

---

## 6. Classification Geographique

### 6.1 Competitions Nationales (sans region)

- Interclubs (Top 16, N1, N2, N3, N4)
- Interclubs_Feminins
- Interclubs_Jeunes
- Interclubs_Rapide
- Coupe_de_France
- Coupe_de_la_Parite
- Coupe_Jean-Claude_Loubatiere
- Competitions_Scolaires

### 6.2 Competitions Regionales (avec ligue)

| Ligue | Code | Region |
|-------|------|--------|
| Auvergne-Rhone-Alpes | ARA | Sud-Est |
| Bourgogne Franche-Comte | BFC | Est |
| Bretagne | BRE | Ouest |
| Centre Val de Loire | CVL | Centre |
| Corse | COR | Sud |
| Grand Est | GES | Est |
| Guyane | GUY | Outre-Mer |
| Hauts-De-France | HDF | Nord |
| Ile de France | IDF | Centre |
| Normandie | NOR | Nord-Ouest |
| Nouvelle Aquitaine | NAQ | Sud-Ouest |
| Occitanie | OCC | Sud |
| Pays de la Loire | PDL | Ouest |
| Provence Alpes Cote d'Azur | PACA | Sud-Est |
| Reunion | REU | Outre-Mer |

---

## 7. Features pour XGBoost

### 7.1 Features Directes (extraites)

```python
features_directes = {
    # Contexte temporel
    'saison': int,              # 2002-2026
    'ronde': int,               # 1-11
    'jour_semaine': str,        # Samedi, Dimanche
    'heure': str,               # 10:00, 14:00, 16:00

    # Contexte competition
    'type_competition': str,    # National, Regional
    'niveau': int,              # 1 (N1) a 4+ (dept)
    'ligue_code': str,          # HDF, IDF, etc.

    # Echiquier
    'numero_echiquier': int,    # 1-8
    'elo_blanc': int,           # 1000-2800
    'elo_noir': int,            # 1000-2800
    'diff_elo': int,            # -800 a +800
    'titre_blanc': str,         # GM, IM, FM, ''
    'titre_noir': str,          # GM, IM, FM, ''

    # Resultat (target)
    'resultat': float,          # 1.0, 0.5, 0.0
}
```

### 7.2 Features Derivees (a calculer)

```python
features_derivees = {
    # Historique joueur
    'nb_parties_joueur': int,        # Parties jouees cette saison
    'score_moyen_joueur': float,     # Performance moyenne
    'forme_recente': float,          # 5 dernieres parties

    # Historique equipe
    'classement_equipe': int,        # Position au classement
    'score_equipe_cumule': float,    # Points avant cette ronde
    'domicile': bool,                # Joue a domicile?

    # Historique adversaire
    'score_vs_adversaire': float,    # Historique face-a-face
    'elo_moyen_adversaire': int,     # Force moyenne equipe adverse

    # Contexte match
    'enjeu_match': str,              # Maintien, titre, milieu
    'difference_classement': int,    # Ecart au classement
}
```

---

## 8. Donnees par Saison

| Saison | Groupes | Fichiers | Observations |
|--------|---------|----------|--------------|
| 2026 | 710 | 3,974 | En cours |
| 2025 | 825 | 4,720 | Complete |
| 2024 | 762 | 4,445 | Complete |
| 2023 | 693 | 4,032 | Complete |
| 2022 | 712 | 3,857 | Complete |
| 2021 | 652 | 2,316 | COVID reduit |
| 2020 | 77 | 692 | COVID tronque |
| 2019 | 762 | 4,179 | Complete |
| 2018 | 752 | 4,208 | Complete |
| 2017 | 745 | 4,011 | Complete |
| 2016 | 747 | 4,045 | Complete |
| 2015 | 736 | 4,122 | Complete |
| 2014 | 685 | 3,921 | Complete |
| 2013 | 633 | 3,564 | Complete |
| 2012 | 662 | 3,658 | Complete |
| 2011 | 638 | 3,669 | Complete |
| 2010 | 631 | 3,664 | Complete |
| 2009 | 663 | 3,783 | Complete |
| 2008 | 635 | 3,651 | Complete |
| 2007 | 614 | 3,660 | Complete |
| 2006 | 599 | 3,571 | Complete |
| 2005 | 525 | 2,731 | Moins de comp. |
| 2004 | 394 | 2,262 | Moins de comp. |
| 2003 | 255 | 1,696 | Moins de comp. |
| 2002 | 41 | 357 | 1ere numerisee |

---

## 9. Qualite des Donnees

### 9.1 Points Forts

- [x] Couverture complete 2002-2026
- [x] Tous niveaux (Top 16 aux departementales)
- [x] Elo disponible pour 95%+ des joueurs
- [x] Dates et lieux des matchs
- [x] Titres FIDE quand disponibles
- [x] Scores individuels et par equipe

### 9.2 Limitations Connues

- [ ] 2020 tres incomplet (COVID)
- [ ] Elo parfois 0 pour joueurs non classes
- [ ] Quelques forfaits sans details (note "F")
- [ ] Anciens joueurs (pre-2010) parfois sans prenom
- [ ] Lieu parfois generique (nom ville seulement)

### 9.3 Nettoyage Recommande

1. **Elo = 0** : Imputer avec moyenne echiquier ou exclure
2. **Forfaits** : Exclure ou traiter separement
3. **COVID 2020** : Ponderer ou exclure
4. **Doublons joueurs** : Normaliser les noms (accents, casse)

---

## 10. Utilisation avec parse.py

### 10.1 Parser un Groupe

```python
from parse import parse_groupe
from pathlib import Path

groupe_dir = Path("data/2025/Interclubs/Nationale_1/Groupe_A")
result = parse_groupe(groupe_dir)

# Acceder aux metadata
print(result["metadata"]["saison"])      # 2025
print(result["metadata"]["niveau"])      # 1

# Parcourir les matchs
for ronde_num, matchs in result["rondes"].items():
    for match in matchs:
        print(f"R{ronde_num}: {match['equipe_dom']} vs {match['equipe_ext']}")
        print(f"  Date: {match['date_str']} - Lieu: {match['lieu']}")

        for ech in match["echiquiers"]:
            print(f"  Ech {ech['numero']}: {ech['blanc']['nom_complet']} vs {ech['noir']['nom_complet']}")
            print(f"    Diff Elo: {ech['diff_elo']:+d} - Resultat: {ech['type_resultat']}")
```

### 10.2 Export vers DataFrame

```python
import pandas as pd
from parse import parse_groupe
from pathlib import Path

def groupe_to_dataframe(groupe_dir):
    result = parse_groupe(groupe_dir)
    rows = []

    for ronde_num, matchs in result["rondes"].items():
        for match in matchs:
            for ech in match["echiquiers"]:
                row = {
                    # Metadata
                    "saison": result["metadata"]["saison"],
                    "competition": result["metadata"]["competition"],
                    "division": result["metadata"]["division"],
                    "niveau": result["metadata"]["niveau"],
                    "type_competition": result["metadata"]["type_competition"],
                    "ligue_code": result["metadata"]["ligue_code"],

                    # Match
                    "ronde": ronde_num,
                    "date": match["date"],
                    "heure": match["heure"],
                    "lieu": match["lieu"],
                    "equipe_dom": match["equipe_dom"],
                    "equipe_ext": match["equipe_ext"],

                    # Echiquier
                    "echiquier": ech["numero"],
                    "blanc_nom": ech["blanc"]["nom_complet"],
                    "blanc_elo": ech["blanc"]["elo"],
                    "blanc_titre": ech["blanc"]["titre_fide"],
                    "noir_nom": ech["noir"]["nom_complet"],
                    "noir_elo": ech["noir"]["elo"],
                    "noir_titre": ech["noir"]["titre_fide"],
                    "diff_elo": ech["diff_elo"],
                    "resultat": ech["resultat_blanc"],
                    "type_resultat": ech["type_resultat"],
                }
                rows.append(row)

    return pd.DataFrame(rows)
```

---

## 11. Conclusion : Faut-il Re-Scraper ?

### Verdict : **NON, les donnees sont COMPLETES** (2002-2026)

| Critere | Statut | Commentaire |
|---------|--------|-------------|
| Elo joueurs | OK | Disponible dans ronde_X.html |
| Titres FIDE | OK | Disponible (g/m/f) |
| Dates matchs | OK | Disponible dans calendrier.html |
| Lieux matchs | OK | Disponible dans calendrier.html |
| Scores | OK | Match + echiquier |
| Metadata geo | OK | Extractible de l'arborescence |
| Historique 20+ ans | OK | 2002-2025 |

**Toutes les donnees necessaires pour XGBoost sont presentes.**

Le parsing avec `parse.py` extrait maintenant 100% des champs disponibles.

---

## 12. Prochaines Etapes ALICE-Engine

1. **Parsing compositions** : Executer parse.py sur les 15,148 groupes
2. **Parsing joueurs v2** : Adapter parser pour nouveau format `players_v2/`
3. **Export** : Generer fichiers Parquet/CSV
4. **Nettoyage** : Normaliser noms, gerer Elo=0
5. **Features** : Calculer features derivees (historique)
6. **Training** : Entrainer XGBoost sur dataset complet
7. **Cleanup** : Supprimer `data/players/` (v1 obsolete)

---

*Document mis a jour le 4 Janvier 2026*
*Source: FFE Scraper - 86,948 fichiers HTML (84,789 compositions + 2,159 joueurs) - 100% succes*
*Joueurs: 66,208 licencies (100% couverture FFE Open Data)*
