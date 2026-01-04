# Contexte Dataset FFE - Resume pour Prochaines Taches

> **Objectif** : Resume des infos essentielles pour le parsing et l'entrainement ALICE
> **Source** : Scraping FFE 1er-4 Janvier 2026

---

## 1. Dataset Disponible

| Metrique | Valeur |
|----------|--------|
| **Emplacement** | `C:/Dev/Alice-Engine/dataset_alice/` (lien vers `ffe_data_backup`) |
| **Taille totale** | **2.5 GB** |
| **Format** | HTML brut (a parser) |
| **Periode** | 2002-2026 (25 saisons) |

### Fichiers

| Type | Quantite | Taille |
|------|----------|--------|
| Compositions (calendrier + rondes) | 84,789 | ~2.1 GB |
| Joueurs FFE v2 (COMPLET) | 2,159 pages (992 clubs) | ~120 MB |
| Index clubs | 1 (JSON) | ~200 KB |
| **Total** | **86,949** | **~2.5 GB** |

### Estimations apres parsing

| Element | Estimation |
|---------|------------|
| Matchs | ~110,000 |
| Echiquiers/Parties | ~750,000 |
| Joueurs licencies | 66,208 |

---

## 2. Structure des Fichiers HTML

### Arborescence

```
dataset_alice/
├── clubs/
│   └── clubs_index.json        # Index 992 clubs (ref, nom, dept, licences)
│
├── players_v2/                  # COMPLET - 66,208 joueurs
│   └── club_{ref}/              # 992 dossiers clubs
│       └── page_XX.html         # ~40 joueurs/page, tous licencies
│
├── players/                     # OBSOLETE - FIDE only (35,320 joueurs)
│   └── page_XXXX.html           # 883 pages
│
└── {saison}/                    # 2002-2026
    └── {competition}/           # Interclubs, Coupe_de_France, Ligue_XXX
        └── {division}/          # Nationale_1, Regionale_2...
            └── {groupe}/        # Groupe_A, Poule_1...
                ├── calendrier.html   # Dates, lieux, scores
                └── ronde_N.html      # Compositions detaillees
```

### Extraction depuis arborescence (metadata)

| Champ | Source | Exemple |
|-------|--------|---------|
| `saison` | Dossier niveau 1 | `2025` |
| `competition` | Dossier niveau 2 | `Interclubs` |
| `division` | Dossier niveau 3 | `Nationale_1` |
| `groupe` | Dossier niveau 4 | `Groupe_A` |
| `niveau` | Extrait de division | `1` |
| `type_competition` | Deduit | `National` ou `Regional` |
| `ligue_code` | Mapping | `HDF`, `IDF`... |

---

## 3. Champs a Extraire par Fichier

### calendrier.html

| Champ | Description | Exemple |
|-------|-------------|---------|
| `ronde` | Numero ronde | `1` |
| `equipe_dom` | Equipe domicile | `Cappelle-La-Grande` |
| `equipe_ext` | Equipe exterieure | `Lille Universite Club` |
| `score_dom` | Score domicile | `4` |
| `score_ext` | Score exterieur | `1` |
| `date` | Date (datetime) | `2024-10-12 16:00` |
| `lieu` | Ville | `LILLE` |

### ronde_N.html (par echiquier)

| Champ | Description | Exemple |
|-------|-------------|---------|
| `numero` | Echiquier | `1` |
| `blanc.nom_complet` | Joueur blancs | `NIKOLOV Momchil` |
| `blanc.elo` | Elo blancs | `2424` |
| `blanc.titre` | Titre FIDE | `g` → `GM` |
| `noir.nom_complet` | Joueur noirs | `LESUEUR Gabriel` |
| `noir.elo` | Elo noirs | `2256` |
| `resultat_blanc` | Score blancs | `0.5` |
| `type_resultat` | Categorise | `nulle` |
| `diff_elo` | Ecart Elo | `+168` |

### players_v2/club_XXX/page_XX.html (joueurs - NOUVEAU FORMAT)

| Champ | Description | Exemple |
|-------|-------------|---------|
| `nr_ffe` | ID FFE | `K59857` |
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
| `club` | Club | `Echiquier de Bigorre` |

### Distribution des Joueurs (66,208 total)

| Categorie | Count | % | Type Elo |
|-----------|-------|---|----------|
| Jeunes U08-U14 | 33,896 | 51.2% | Estime (E) |
| Ados U16-U20 | 5,278 | 8.0% | Estime/National |
| Adultes X20-X65 | 27,034 | 40.8% | FIDE (F) |

| Type Elo | Count | % |
|----------|-------|---|
| Estime (E) | 47,746 | 72.1% |
| FIDE (F) | 17,494 | 26.4% |
| National (N) | 965 | 1.5% |

---

## 4. Codes et Mappings

### Titres FIDE

| Code HTML | Titre |
|-----------|-------|
| `g` | GM (Grand-Maitre) |
| `m` | IM (Maitre International) |
| `f` | FM (Maitre FIDE) |
| `gf` | WGM (Grand-Maitre Feminin) |
| `mf` | WIM (Maitre International Feminin) |
| `ff` | WFM (Maitre FIDE Feminin) |

### Resultats

| Texte HTML | Type | Score B | Score N |
|------------|------|---------|---------|
| `1 - 0` | victoire_blanc | 1.0 | 0.0 |
| `0 - 1` | victoire_noir | 0.0 | 1.0 |
| `X - X` ou `1/2` | nulle | 0.5 | 0.5 |
| `F - +` | forfait_blanc | 0.0 | 1.0 |
| `+ - F` | forfait_noir | 1.0 | 0.0 |

### Categories d'age (mapping legacy -> FFE)

| HTML (legacy) | Code FFE | Nom | Age |
|---------------|----------|-----|-----|
| PpoM / PpoF | U08 / U08F | Petits Poussins | < 8 ans |
| PouM / PouF | U10 / U10F | Poussins | 8-9 ans |
| PupM / PupF | U12 / U12F | Pupilles | 10-11 ans |
| BenM / BenF | U14 / U14F | Benjamins | 12-13 ans |
| MinM / MinF | U16 / U16F | Minimes | 14-15 ans |
| CadM / CadF | U18 / U18F | Cadets | 16-17 ans |
| JunM / JunF | U20 / U20F | Juniors | 18-19 ans |
| SenM / SenF | X20 | Seniors | 20-49 ans |
| SepM / SepF | X50 | Seniors Plus | 50-64 ans |
| VetM / VetF | X65 | Veterans | 65+ ans |

### Ligues regionales

| Ligue | Code |
|-------|------|
| Hauts-De-France | HDF |
| Ile de France | IDF |
| Provence Alpes Cote d'Azur | PACA |
| Auvergne-Rhone-Alpes | ARA |
| *(15 ligues au total)* | ... |

---

## 5. Qualite des Donnees - Points d'attention

### A traiter lors du parsing

| Probleme | Frequence | Solution |
|----------|-----------|----------|
| **Elo = 0** | ~5% joueurs | Imputer moyenne echiquier ou flag |
| **Forfaits (F)** | ~2% parties | Exclure ou traiter separement |
| **COVID 2020** | 77 groupes | Ponderer ou exclure |
| **Doublons noms** | Variable | Normaliser (accents, casse) |
| **Pre-2010 sans prenom** | ~10% anciens | Garder tel quel |

### Saisons speciales

| Saison | Statut | Groupes | Note |
|--------|--------|---------|------|
| 2020 | Incomplet | 77 | COVID - saison tronquee |
| 2021 | Partiel | 652 | COVID - competitions reduites |
| 2002-2004 | Partiel | 41-394 | Moins de competitions |
| 2026 | En cours | 710 | Saison actuelle |

---

## 6. Parser existant

Le projet `ffe_scrapper` contient deja un parser :

```
C:/Dev/ffe_scrapper/src/parse.py
C:/Dev/ffe_scrapper/src/analyze_players_v2.py  # Nouveau - analyse joueurs v2
```

### Utilisation documentee

```python
from parse import parse_groupe
from pathlib import Path

groupe_dir = Path("data/2025/Interclubs/Nationale_1/Groupe_A")
result = parse_groupe(groupe_dir)

# result["metadata"]["saison"] → 2025
# result["rondes"][1][0]["echiquiers"][0]["blanc"]["elo"] → 2424
```

### Export DataFrame

```python
def groupe_to_dataframe(groupe_dir):
    result = parse_groupe(groupe_dir)
    rows = []
    for ronde_num, matchs in result["rondes"].items():
        for match in matchs:
            for ech in match["echiquiers"]:
                row = {
                    "saison": result["metadata"]["saison"],
                    "ronde": ronde_num,
                    "echiquier": ech["numero"],
                    "blanc_nom": ech["blanc"]["nom_complet"],
                    "blanc_elo": ech["blanc"]["elo"],
                    # ... autres champs
                }
                rows.append(row)
    return pd.DataFrame(rows)
```

---

## 7. Prochaines Taches (ordre recommande)

1. **Adapter parse_dataset.py** pour nouveau format `players_v2/`
2. **Parser les 15,148 groupes** → `exports/echiquiers.parquet`
3. **Parser les 66,208 joueurs** → `exports/joueurs.parquet`
4. **Nettoyage** : Elo=0, forfaits, normalisation noms
5. **Feature engineering** : Features derivees (historique)
6. **Export final** : Parquet optimise pour CatBoost/XGBoost
7. **Cleanup** : Supprimer `players/` (v1 obsolete)

---

## 8. Fichiers de sortie attendus

```
C:/Dev/Alice-Engine/exports/
├── echiquiers.parquet    # ~750k lignes, ~150-300 MB
├── joueurs.parquet       # 66,208 lignes, ~10 MB
└── metadata.json         # Stats parsing
```

---

*Resume mis a jour le 4 Janvier 2026*
*Source: 86,949 fichiers HTML - 66,208 joueurs (100% couverture FFE)*
