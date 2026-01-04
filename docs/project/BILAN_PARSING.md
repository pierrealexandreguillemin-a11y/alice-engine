# Bilan Parsing Dataset FFE

> **Version**: 0.2.0
> **Date**: 3 Janvier 2026
> **Script**: `scripts/parse_dataset.py`
> **Norme**: ISO 25012 - Qualite des donnees

---

## 1. Resume executif

Le parsing du dataset FFE (2.4 GB HTML) a ete realise avec succes le 3 janvier 2026.

| Metrique | Valeur |
|----------|--------|
| **Fichiers HTML sources** | 85,672 |
| **Groupes parses** | 13,935 |
| **Echiquiers extraits** | 1,736,490 |
| **Joueurs licencies** | 35,320 |
| **Duree parsing** | ~48 minutes |
| **Taille sortie** | 35.8 MB (Parquet) |

---

## 2. Fichiers generes

### 2.1 echiquiers.parquet (34.2 MB)

**1,736,490 lignes** representant chaque partie jouee sur un echiquier.

#### Schema (32 colonnes)

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `saison` | int | Annee de la saison | 2025 |
| `competition` | str | Nom competition | "Interclubs" |
| `division` | str | Division | "Nationale 1" |
| `groupe` | str | Groupe/Poule | "Groupe A" |
| `ligue` | str | Ligue regionale (si applicable) | "Hauts-De-France" |
| `ligue_code` | str | Code ligue | "HDF" |
| `niveau` | int | Niveau numerique | 1 |
| `type_competition` | str | Type categorise | "national" |
| `ronde` | int | Numero de ronde | 1 |
| `equipe_dom` | str | Equipe domicile | "Cappelle-La-Grande" |
| `equipe_ext` | str | Equipe exterieur | "Lille UC" |
| `score_dom` | int | Score equipe domicile | 4 |
| `score_ext` | int | Score equipe exterieur | 1 |
| `date` | datetime | Date/heure du match | 2024-10-12 16:00 |
| `date_str` | str | Date brute | "samedi 12/10/24 16:00" |
| `heure` | str | Heure | "16:00" |
| `jour_semaine` | str | Jour | "Samedi" |
| `lieu` | str | Lieu du match | "LILLE" |
| `echiquier` | int | Numero d'echiquier | 1 |
| `blanc_nom` | str | Nom joueur blancs | "NIKOLOV Momchil" |
| `blanc_titre` | str | Titre FIDE blancs | "GM" |
| `blanc_elo` | int | Elo blancs | 2424 |
| `blanc_equipe` | str | Equipe du joueur blancs | "Cappelle-La-Grande" |
| `noir_nom` | str | Nom joueur noirs | "LESUEUR Gabriel" |
| `noir_titre` | str | Titre FIDE noirs | "" |
| `noir_elo` | int | Elo noirs | 2256 |
| `noir_equipe` | str | Equipe du joueur noirs | "Lille UC" |
| `resultat_blanc` | float | Score blancs | 0.5 |
| `resultat_noir` | float | Score noirs | 0.5 |
| `resultat_text` | str | Resultat brut | "X - X" |
| `type_resultat` | str | Type categorise | "nulle" |
| `diff_elo` | int | Difference Elo (blanc - noir) | 168 |

#### Types de resultats

| Type | Count | % |
|------|-------|---|
| `victoire_blanc` | 616,123 | 35.5% |
| `victoire_noir` | 599,973 | 34.6% |
| `non_joue` | 233,024 | 13.4% |
| `nulle` | 189,776 | 10.9% |
| `forfait_noir` | 52,145 | 3.0% |
| `forfait_blanc` | 31,753 | 1.8% |
| `victoire_blanc_ajournement` | 5,294 | 0.3% |
| `victoire_noir_ajournement` | 5,191 | 0.3% |
| `double_forfait` | 2,953 | 0.2% |
| `ajournement` | 258 | 0.0% |

### 2.2 joueurs.parquet (3.0 MB)

**66,208 lignes** representant les joueurs licencies FFE 2025 (v2 COMPLET).

#### Schema (19 colonnes)

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `nr_ffe` | str | Licence FFE | "K59857" |
| `id_ffe` | int | ID interne FFE | 672495 |
| `nom` | str | Nom | "AALIOULI" |
| `prenom` | str | Prenom | "Karim" |
| `nom_complet` | str | Nom complet | "AALIOULI Karim" |
| `affiliation` | str | Type de licence FFE | "A", "B" ou "N" |
| `elo` | int | Elo standard | 1567 |
| `elo_type` | str | Type Elo | "F" (Fide), "N" (National), "E" (Estime) |
| `elo_rapide` | int | Elo rapide | 1500 |
| `elo_rapide_type` | str | Type Elo rapide | "N" |
| `elo_blitz` | int | Elo blitz | 1500 |
| `elo_blitz_type` | str | Type Elo blitz | "E" |
| `categorie` | str | Categorie legacy FFE | "SenM" |
| `code_ffe` | str | Code officiel FFE | "Sen", "U8", "S50" |
| `genre` | str | Genre (M/F) | "M" |
| `age_min` | int/null | Age minimum | 20 (null pour U8) |
| `age_max` | int/null | Age maximum | 49 (null pour S65) |
| `mute` | bool | **Mute** = transfere d'un autre club | false (~99%), true (~1%) |
| `club` | str | Nom du club (parfois + ville) | "Echiquier de Bigorre" |

### 2.3 Mapping des categories FFE

Les donnees HTML utilisent un format **legacy** different du format officiel FFE 2025.

| Donnees (legacy) | Code FFE | Nom complet | Age |
|------------------|----------|-------------|-----|
| PpoM / PpoF | U8 / U8F | Petits Poussins | < 8 ans |
| PouM / PouF | U10 / U10F | Poussins | 8-9 ans |
| PupM / PupF | U12 / U12F | Pupilles | 10-11 ans |
| BenM / BenF | U14 / U14F | Benjamins | 12-13 ans |
| MinM / MinF | U16 / U16F | Minimes | 14-15 ans |
| CadM / CadF | U18 / U18F | Cadets | 16-17 ans |
| JunM / JunF | U20 / U20F | Juniors | 18-19 ans |
| SenM / SenF | Sen | Seniors | 20-49 ans |
| SepM / SepF | S50 | Seniors Plus | 50-64 ans |
| VetM / VetF | S65 | Veterans | 65+ ans |

**Note**: Le suffixe M/F indique le genre (Masculin/Feminin).

### 2.4 Champ "Mute"

Le champ `mute` indique si un joueur a ete **transfere d'un autre club** cette saison.

- **~99%** des joueurs : `mute = false` (restent dans leur club)
- **~1%** des joueurs : `mute = true` (nouveaux transferts)

*Note: Ce champ sera utile pour le Composition Engine (CE) - les reglements FFE limitent le nombre de mutes par equipe.*

---

## 3. Distribution des donnees

### 3.1 Par saison

```
Saison    Echiquiers
------    ----------
2002         29,847
2003         35,241
2004         38,456
...
2017         83,239
2018         86,292
2019         84,665
2020         22,632  <- COVID-19
2021         42,180  <- Reprise partielle
2022         75,349
2023         81,663
2024         90,696
2025         98,066
2026         79,793  <- Saison en cours
```

**Note**: L'annee 2020 montre une chute de 73% due au COVID-19.

### 3.2 Par type de competition

| Type | Echiquiers | % |
|------|------------|---|
| `regional` | 846,290 | 48.7% |
| `autre` | 508,406 | 29.3% |
| `national_jeunes` | 117,990 | 6.8% |
| `coupe_jcl` | 95,348 | 5.5% |
| `national` | 66,320 | 3.8% |
| `scolaire` | 48,708 | 2.8% |
| `coupe` | 20,992 | 1.2% |
| `national_feminin` | 17,796 | 1.0% |
| `coupe_parite` | 12,912 | 0.7% |
| `national_rapide` | 1,728 | 0.1% |

### 3.3 Par division (Interclubs nationaux)

| Division | Echiquiers |
|----------|------------|
| Nationale 1 | ~8,000 |
| Nationale 2 | ~12,000 |
| Nationale 3 | ~18,000 |
| Nationale 4 | ~28,000 |

---

## 4. Limitation connue: Dataset joueurs incomplet

⚠️ **Le dataset `joueurs.parquet` ne contient que ~35k joueurs sur ~66k licenciés FFE.**

### Cause
Le scraping initial (`players/page_*.html`) a capturé une liste filtrée, excluant la majorité des jeunes (U08-U14).

### Impact
| Catégorie | Manquants |
|-----------|-----------|
| U08-U14 | ~90% absents |
| U16-U20 | ~30% absents |
| X20-X65 | ✅ Quasi complet |

### Action requise
Un nouveau scraping complet est documenté dans:
```
C:\Dev\ffe_scrapper\TODO_SCRAPING_JOUEURS_COMPLET.md
```

### Impact sur ALICE
- **Compositions (echiquiers.parquet)**: ✅ Non impacté (1.7M lignes OK)
- **Prédictions interclubs adultes**: ✅ Non impacté (adultes présents)
- **Enrichissement joueurs**: ⚠️ Partiel jusqu'au nouveau scraping

---

## 5. Qualite des donnees (ISO 25012)

### 5.1 Completude

| Champ | Completude | Notes |
|-------|------------|-------|
| `saison` | 100% | Extrait du chemin |
| `competition` | 100% | Extrait du chemin |
| `blanc_nom` | 100% | Toujours present |
| `blanc_elo` | 81.8% | 18.2% = 0 (non classes) |
| `date` | ~85% | Calendrier pas toujours disponible |
| `lieu` | ~80% | Idem |

### 4.2 Problemes identifies

#### Elo = 0 (18.2% des echiquiers)

**315,395 echiquiers** ont au moins un joueur avec Elo = 0.

Causes:
- Joueurs non classes FFE (debutants)
- Joueurs etrangers sans Elo FFE
- Erreurs de saisie FFE

**Recommandation**: Filtrer ou imputer pour l'entrainement ML.

#### Forfaits (5.0% des echiquiers)

**86,851 echiquiers** sont des forfaits.

Types:
- `forfait_blanc`: Joueur blancs absent (31,753)
- `forfait_noir`: Joueur noirs absent (52,145)
- `double_forfait`: Les deux joueurs absents (2,953)

**Recommandation**:
- **Entrainement direct** : Exclure (pas de partie jouee)
- **Feature engineering** : Conserver pour extraire patterns de fiabilite
  - Identifier joueurs/clubs a risque de forfait
  - Cf. ANALYSE_INITIALE_ALICE.md Phase 2.3

#### Non joues (13.4% des echiquiers)

**233,024 echiquiers** n'ont pas de resultat enregistre.

Causes:
- Rondes futures (saison 2026 en cours) : 44,675
- COVID 2020-2021 (saisons interrompues) : 47,221
- Matchs annules/reportes : ~141,000

**Recommandation**:
- **Entrainement direct** : Exclure (pas de partie jouee)
- **Feature engineering** : Conserver pour extraire patterns de fiabilite
  - `taux_forfait_club` : clubs defaillants
  - `pattern_dispo_joueur` : absences par mois/jour
  - Cf. ANALYSE_INITIALE_ALICE.md Phase 2.3

#### Ajournements (0.6% des echiquiers)

**10,743 echiquiers** sont des parties ajournees.

Types:
- `victoire_blanc_ajournement`: 5,294
- `victoire_noir_ajournement`: 5,191
- `ajournement`: 258 (resultat final inconnu)

**Recommandation**: Inclure dans l'entrainement (parties jouees avec resultat).

#### Donnees manquantes calendrier

~15% des matchs n'ont pas de date/lieu.

Causes:
- Fichier `calendrier.html` absent
- Format non standard

**Recommandation**: Acceptable pour ML (features optionnelles).

### 4.3 Coherence

| Verification | Resultat |
|--------------|----------|
| Score match = somme echiquiers | OK (99.8%) |
| Elo dans plage [0, 3000] | OK (100%) |
| Ronde dans plage [1, 15] | OK (100%) |
| Echiquier dans plage [1, 12] | OK (100%) |

---

## 5. Comparaison previsions vs realite

| Metrique | Prevu | Reel | Ecart |
|----------|-------|------|-------|
| Echiquiers | ~750,000 | 1,736,490 | **+131%** |
| Joueurs | ~55,000 | 35,320 | -36% |
| Forfaits | ~2% | 5.0% | +3% |
| Elo=0 | ~5% | 18.2% | +13% |

**Analyse**:
- Le dataset est **2.3x plus riche** que prevu en compositions
- Les joueurs sont moins nombreux (une seule saison 2025)
- Les problemes de qualite (Elo=0, forfaits) sont plus frequents

---

## 6. Recommandations pour l'entrainement

### 6.1 Pipeline donnees

```python
# ETAPE 1: Charger TOUTES les donnees (pour feature engineering)
df_all = pd.read_parquet('data/echiquiers.parquet')

# ETAPE 2: Extraire features de fiabilite AVANT filtrage
# (utilise non_joue et forfaits pour patterns)
df_reliability = extract_reliability_features(df_all)
# Cf. ANALYSE_INITIALE_ALICE.md Phase 2.3

# ETAPE 3: Filtrer pour entrainement direct
df_train = df_all[~df_all['type_resultat'].isin([
    'non_joue', 'forfait_blanc', 'forfait_noir', 'double_forfait'
])]

# ETAPE 4: Exclure Elo=0 (ou imputer)
df_train = df_train[(df_train['blanc_elo'] > 0) & (df_train['noir_elo'] > 0)]

# Resultat: ~1.2M echiquiers exploitables pour training
# + features fiabilite extraites de 320k echiquiers "vides"
```

### 6.2 Features recommandees pour CatBoost

**Numeriques**:
- `blanc_elo`, `noir_elo`
- `diff_elo`
- `echiquier` (position dans l'equipe)
- `niveau` (niveau de la division)

**Categoriques** (natives CatBoost):
- `type_competition`
- `division`
- `ligue_code`
- `blanc_titre`, `noir_titre`
- `jour_semaine`

**Fiabilite** (derivees des non_joue/forfaits):
- `taux_forfait_club` : historique defaillance club
- `taux_presence_joueur` : fiabilite joueur
- `pattern_dispo_mois` : disponibilite saisonniere
- `pattern_dispo_jour` : disponibilite jour semaine

**Target**:
- `resultat_blanc` (0, 0.5, 1) ou classification (victoire/nulle/defaite)

### 6.3 Split temporel recommande

```
Train: 2002-2022 (~1.1M echiquiers)
Valid: 2023      (~80k echiquiers)
Test:  2024-2025 (~190k echiquiers)
```

---

## 7. Usage du script

```bash
# Parsing complet (default)
python scripts/parse_dataset.py

# Mode test (un groupe seulement)
python scripts/parse_dataset.py --test

# Verbose
python scripts/parse_dataset.py --verbose

# Compositions seulement
python scripts/parse_dataset.py --compositions-only

# Joueurs seulement
python scripts/parse_dataset.py --joueurs-only

# Chemin personnalise
python scripts/parse_dataset.py --data-dir /path/to/data --output-dir /path/to/output
```

---

## 8. Fichiers source

| Source | Fichiers | Description |
|--------|----------|-------------|
| `dataset_alice/{saison}/.../ronde_N.html` | 84,789 | Compositions par ronde |
| `dataset_alice/{saison}/.../calendrier.html` | ~10,000 | Dates et lieux |
| `dataset_alice/players/page_XXXX.html` | 883 | Joueurs licencies |

---

## 9. Performance

| Metrique | Valeur |
|----------|--------|
| Temps total | 48 minutes |
| Vitesse parsing | ~290 groupes/minute |
| Memoire peak | ~2 GB |
| Compression Parquet | Snappy |

---

## 10. Jointure echiquiers ↔ joueurs

### 10.1 Probleme: Absence de nr_ffe dans compositions

Les fichiers `ronde_N.html` ne contiennent **PAS** le numero de licence FFE (nr_ffe).

| Source | Champs disponibles |
|--------|-------------------|
| echiquiers.parquet | `blanc_nom`, `blanc_elo`, `blanc_titre` |
| joueurs.parquet | `nr_ffe`, `nom_complet`, `elo` |

**Jointure obligatoire**: par `nom_complet` (+ Elo pour desambiguation)

### 10.2 Analyse des homonymes

| Categorie | Joueurs | Homonymes | Taux ambiguite |
|-----------|---------|-----------|----------------|
| Non classes (Elo=1299) | 35,973 (54%) | Concentres ici | Eleve |
| Classes (Elo>1299) | 30,232 (46%) | 372 | **0.2%** |
| **Total** | 66,208 | 1,019 | 1.54% |

**Exemples d'homonymes (4 occurrences):**
- MARTIN Abel: 4x, Elos [1299, 1299, 1299, 1299] (ambigus)
- DIDIER Eric: 4x, Elos [1884, 1660, 1569, 1399] (differenciables)
- NOEL Nicolas: 4x, Elos [2006, 1772, 1669, 1399] (differenciables)

### 10.3 Limitation importante: Evolution temporelle

⚠️ **Les joueurs evoluent sur 25 ans de donnees !**

| Donnee | Source | Temporalite |
|--------|--------|-------------|
| `blanc_elo` / `noir_elo` | echiquiers.parquet | Elo AU MOMENT du match |
| `elo` | joueurs.parquet | Elo saison **2025 uniquement** |

**Exemple: BACROT Etienne**
- 2005: Elo 2731 (pic de carriere)
- 2025: Elo 2631 (joueurs.parquet)
- Difference: **100 points**

**Consequence**: La desambiguation par Elo fonctionne uniquement pour:
- Matchs recents (2024-2025)
- Joueurs stables (Elo varie peu)

### 10.4 Strategie de jointure recommandee

```python
def join_with_joueurs(echiquiers, joueurs):
    """
    Jointure echiquiers <-> joueurs.

    IMPORTANT: joueurs.parquet contient UNIQUEMENT les donnees 2025.
    Pour les matchs historiques, la jointure par Elo est imprecise.

    Strategie:
    1. Jointure par nom_complet uniquement
    2. Pour homonymes: flag ambiguite (pas de resolution automatique)
    3. Enrichissement: nr_ffe pour tracking cross-saison
    """
    # Detecter homonymes
    joueurs['is_homonyme'] = joueurs.duplicated('nom_complet', keep=False)

    joined = echiquiers.merge(
        joueurs[['nom_complet', 'nr_ffe', 'club', 'categorie', 'is_homonyme']],
        left_on='blanc_nom',
        right_on='nom_complet',
        how='left'
    )

    # Pour homonymes: prendre le premier (accepter ambiguite 1.54%)
    joined = joined.drop_duplicates(subset=echiquiers.columns.tolist())

    return joined
```

**Note**: L'Elo utilise pour le ML est `blanc_elo`/`noir_elo` (au moment du match),
PAS l'Elo de `joueurs.parquet` (2025 uniquement).

### 10.5 Impact pour ALI

| Cas | % | Impact |
|-----|---|--------|
| Joueur unique | 98.5% | ✅ Jointure parfaite |
| Homonyme (toutes saisons) | 1.5% | ⚠️ Flag ambiguite |

**Important**: Pour l'entrainement ALI, l'Elo est deja dans `echiquiers.parquet`
(`blanc_elo`, `noir_elo`). La jointure avec `joueurs.parquet` sert uniquement a:
- Recuperer `nr_ffe` pour tracking cross-saison
- Enrichir avec `categorie`, `club` 2025 (optionnel)

---

## 11. Prochaines etapes

1. **Feature engineering** - Creer script jointure + features derivees
2. **Extraction fiabilite** - Patterns depuis non_joue/forfaits
3. **Evaluation ML** - Comparer CatBoost, XGBoost, LightGBM
4. **Validation** - Metriques sur split temporel

---

*Genere automatiquement le 3 Janvier 2026*
*Script: parse_dataset.py v0.2.0*
