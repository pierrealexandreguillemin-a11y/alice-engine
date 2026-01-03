# Contexte Dataset FFE - Resume pour Prochaines Taches

> **Objectif** : Resume des infos essentielles pour le parsing et l'entrainement ALICE
> **Source** : Scraping FFE 1er-2 Janvier 2026

---

## 1. Dataset Disponible

| Metrique | Valeur |
|----------|--------|
| **Emplacement** | `C:/Dev/Alice-Engine/dataset_alice/` (lien vers `ffe_data_backup`) |
| **Taille totale** | **2.4 GB** |
| **Format** | HTML brut (a parser) |
| **Periode** | 2002-2026 (25 saisons) |

### Fichiers

| Type | Quantite | Taille |
|------|----------|--------|
| Compositions (calendrier + rondes) | 84,789 | ~2.1 GB |
| Joueurs FFE | 883 pages | ~49 MB |
| **Total** | **85,672** | **~2.4 GB** |

### Estimations apres parsing

| Element | Estimation |
|---------|------------|
| Matchs | ~110,000 |
| Echiquiers/Parties | ~750,000 |
| Joueurs uniques | ~55,000 |

---

## 2. Structure des Fichiers HTML

### Arborescence

```
dataset_alice/
├── players/                    # 883 pages joueurs
│   └── page_XXXX.html
│
└── {saison}/                   # 2002-2026
    └── {competition}/          # Interclubs, Coupe_de_France, Ligue_XXX
        └── {division}/         # Nationale_1, Regionale_2...
            └── {groupe}/       # Groupe_A, Poule_1...
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

### page_XXXX.html (joueurs)

| Champ | Description | Exemple |
|-------|-------------|---------|
| `nr_ffe` | ID FFE | `K59857` |
| `nom` | Nom complet | `AALIOULI Karim` |
| `elo` | Classement standard | `1567` |
| `rapide` | Classement rapide | `1500` |
| `blitz` | Classement blitz | `1500` |
| `categorie` | Age | `Sen` |
| `sexe` | Genre | `M` |
| `club` | Club | `Echiquier de Bigorre` |

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

1. **Copier/adapter parse.py** dans `scripts/parse_dataset.py`
2. **Parser les 15,148 groupes** → `exports/echiquiers.parquet`
3. **Parser les 883 pages joueurs** → `exports/joueurs.parquet`
4. **Nettoyage** : Elo=0, forfaits, normalisation noms
5. **Feature engineering** : Features derivees (historique)
6. **Export final** : Parquet optimise pour CatBoost/XGBoost

---

## 8. Fichiers de sortie attendus

```
C:/Dev/Alice-Engine/exports/
├── echiquiers.parquet    # ~750k lignes, ~150-300 MB
├── joueurs.parquet       # ~55k lignes, ~5 MB
└── metadata.json         # Stats parsing
```

---

*Resume genere le 3 Janvier 2026*
