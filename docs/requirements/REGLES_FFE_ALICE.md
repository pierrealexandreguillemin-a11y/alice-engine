# Regles FFE pour ALICE

> **Document Type**: System Requirements (SyRS) - ISO 15289
> **Version**: 2.0.0
> **Date**: 4 Janvier 2026
> **MAJ**: Ajout C03, C04, J03, reglements regionaux/departementaux
> **Source**: Reglements FFE 2025-2026 + Ligues PACA/BdR

---

## 1. Vue d'ensemble

Ce document formalise les regles FFE applicables aux competitions par equipes,
extraites des reglements officiels pour implementation dans ALICE.

**Documents sources federaux**:
- `R01_2025_26_Regles_generales.pdf` - Regles generales FFE
- `R02_2025_26_Regles_generales_Annexes.pdf` - Annexes (ZID, procedures)
- `A02_2025_26_Championnat_de_France_des_Clubs.pdf` - CFC Hommes (Top16 a N4)
- `F01_2025_26_Championnat_de_France_des_clubs_Feminin.pdf` - CFC Feminin
- `C01_2025_26_Coupe_de_France.pdf` - Coupe de France
- `C03_Coupe_Jean_Claude_Loubatiere.pdf` - Coupe bas-elo (max 1800)
- `C04_Coupe_de_la_parité.pdf` - Coupe mixte (2H + 2F)
- `J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf` - Interclubs Jeunes
- `J03_Championnat_de_France_scolaire.pdf` - Championnat scolaire

**Documents sources regionaux (PACA)**:
- `règlement_n4_2024_2025.pdf` - N4 regional PACA
- `règlement_régionale_2024_2025.pdf` - Regionale PACA

**Documents sources departementaux (Bouches-du-Rhone)**:
- `Interclubs_DepartementalBdr.pdf` - Departemental adultes (N6)
- `InterclubsJeunes_PACABdr.pdf` - Interclubs jeunes departemental

---

## 2. Regles de composition d'equipe

### 2.1 Ordre Elo obligatoire (A02 Art. 3.6.e)

```
REGLE: Si deux joueurs ont une difference Elo > 100 points,
       le mieux classe DOIT etre place DEVANT le moins bien classe.

SANCTION: Forfait administratif pour le(s) joueur(s) ayant le plus fort Elo.

IMPLEMENTATION:
  - Valider que composition respecte ordre Elo decroissant
  - Tolerance: 100 points
  - Feature: `ordre_elo_valide` (bool)
```

### 2.2 Joueur brule (A02 Art. 3.7.c) - REGLE CRITIQUE

```
REGLE: Un joueur ayant joue 3 FOIS dans une equipe PLUS FORTE
       ne peut plus jouer dans une equipe MOINS FORTE du meme club.

DEFINITION "plus forte": Equipes numerotees par ordre de force decroissant
  - Club X equipe 1 (N1) > Club X equipe 2 (N2) > Club X equipe 3 (N3)

EXEMPLE:
  - Joueur A joue rondes 1, 2, 3 en N1 pour "Mulhouse 1"
  - Joueur A est "brule" pour "Mulhouse 2" (N2) et inferieur
  - Joueur A peut toujours jouer en N1

SANCTION: Forfait administratif sur 1er echiquier concerne + tous suivants

IMPLEMENTATION:
  - Compteur: `nb_matchs_equipe_superieure[joueur][equipe]`
  - Seuil: 3 matchs
  - Feature: `joueur_brule` (bool)
  - Feature: `matchs_restants_avant_brulage` (int, 0-3)
```

### 2.3 Participation meme groupe (A02 Art. 3.7.d)

```
REGLE: Si un club a plusieurs equipes dans le MEME GROUPE,
       un joueur ne peut jouer que pour UNE SEULE de ces equipes.

EXEMPLE:
  - "Lyon 1" et "Lyon 2" dans le meme groupe de N3
  - Joueur B joue pour "Lyon 1" ronde 1
  - Joueur B INTERDIT de jouer pour "Lyon 2" tout le reste de la saison

IMPLEMENTATION:
  - Flag: `joueur_bloque_groupe[joueur][groupe]`
```

### 2.4 Nombre de matchs (A02 Art. 3.7.e)

```
REGLE: Pour disputer le match N, un joueur doit avoir joue
       MOINS DE N matchs dans le championnat.

EXEMPLE:
  - Ronde 5: joueur doit avoir joue max 4 matchs avant
  - Si joueur a deja 5 matchs, interdit de jouer ronde 5

EXCEPTION Top 16: Max 11 rondes totales dans la saison

IMPLEMENTATION:
  - Compteur: `nb_matchs_saison[joueur]`
  - Contrainte: nb_matchs_saison < numero_ronde
  - Feature: `matchs_disponibles` (int)
```

### 2.5 Noyau 50% (A02 Art. 3.7.f)

```
REGLE: En N1, N2, N3, chaque equipe doit aligner au moins 50%
       de joueurs ayant DEJA JOUE pour cette equipe cette saison.

EXCEPTION: Ne s'applique pas a la ronde 1

EXEMPLE (8 joueurs):
  - Ronde 3: min 4 joueurs du "noyau" (deja joue rondes 1 ou 2)
  - Si seulement 3 du noyau: forfait administratif

IMPLEMENTATION:
  - Set: `noyau_equipe[equipe]` = joueurs ayant deja joue
  - Contrainte: len(composition & noyau) >= len(composition) / 2
  - Feature: `pct_noyau` (float, 0-1)
```

### 2.6 Joueurs mutes (A02 Art. 3.7.g)

```
REGLE: Maximum 3 joueurs MUTES par match.

DEFINITION "mute" (R01 Art. 2.2):
  - Joueur licencie dans un AUTRE club la saison precedente
  - ET a joue une competition par equipe pour cet autre club
  - OU joueur etranger (hors UE) nouvellement licencie

IMPLEMENTATION:
  - Flag: `joueur_mute` (bool) dans joueurs.parquet
  - Contrainte: count(mutes dans composition) <= 3
  - Feature: `nb_mutes_alignes` (int, 0-3)
```

### 2.7 Quota etrangers (A02 Art. 3.7.h)

```
REGLE: Au moins 5 joueurs sur 8 doivent etre:
  - Nationalite francaise, OU
  - Ressortissants UE residant en France, OU
  - Extracommunautaires residant en France depuis 5 ans

POUR EQUIPE 6 JOUEURS: Min 4 joueurs

IMPLEMENTATION:
  - Flag: `etranger_hors_quota` (bool)
  - Contrainte: count(etrangers_hors_quota) <= 3
```

### 2.8 Joueur/Joueuse FR obligatoire (A02 Art. 3.7.i)

```
REGLE: En Top 16, N1, N2, chaque equipe doit inscrire:
  - Au moins 1 JOUEUR francais
  - Au moins 1 JOUEUSE francaise

SANCTION: Forfait avec marque -1 par infraction (dernier echiquier)

IMPLEMENTATION:
  - Contraintes: has_joueur_fr AND has_joueuse_fr
  - Feature: `joueuse_fr_alignee` (bool)
```

### 2.9 Elo max en N4 (A02 Art. 3.7.j)

```
REGLE: Joueurs Elo > 2400 INTERDITS en N4 et divisions inferieures.

EXCEPTION: Si le club a moins de 2 equipes en divisions superieures.

IMPLEMENTATION:
  - Contrainte: if division >= N4: max_elo <= 2400
  - Feature: `joueur_surevalue` (bool)
```

---

## 3. Regles de classement et enjeux

### 3.1 Montees / Descentes

| Division | Equipes/groupe | Montee | Descente |
|----------|----------------|--------|----------|
| Top 16 | 16 | - | 13e-16e (4) |
| N1 | 10 | 1er | 9e-10e (2) |
| N2 | 10 | 1er | 9e-10e (2) |
| N3 | 10 | 1er | 8e-10e (3) |
| N4 | 8-10 | 1er (barrages) | Selon ligue |

### 3.2 Zones d'enjeu

```python
def calculer_zone_enjeu(position: int, nb_equipes: int, division: str) -> str:
    """Determine la zone d'enjeu d'une equipe."""

    # Zones de montee
    if position == 1:
        return "montee"

    # Zones de descente
    if division == "Top16" and position >= 13:
        return "descente"
    elif division in ["N1", "N2"] and position >= 9:
        return "descente"
    elif division == "N3" and position >= 8:
        return "descente"

    # Zone intermediaire
    if position <= 3:
        return "course_titre"
    elif division == "N3" and position >= 6:
        return "danger"
    elif position >= nb_equipes - 3:
        return "danger"

    return "mi_tableau"
```

### 3.3 Features d'enjeu pour ALICE

| Feature | Description | Calcul |
|---------|-------------|--------|
| `zone_enjeu` | Zone actuelle | position → zone |
| `ecart_montee` | Points du 1er - points equipe | pts_1er - pts |
| `ecart_descente` | Points equipe - points releguable | pts - pts_8/9 |
| `matchs_restants` | Rondes restantes | total_rondes - ronde_actuelle |
| `points_max_possibles` | Max atteignable | pts + matchs_restants * 3 |
| `maintien_assure` | Bool | pts > seuil_mathematique |
| `montee_possible` | Bool | points_max >= pts_1er |

---

## 4. Effet "vases communiquants"

### 4.1 Definition

Quand un club a plusieurs equipes, le deplacement d'un joueur
d'une equipe a une autre affecte les deux equipes.

### 4.2 Detection

```python
def get_niveau_equipe(equipe: str) -> int:
    """
    Retourne le niveau hierarchique d'une equipe (1=Top16, 8=N4).

    Args:
        equipe: Nom de l'equipe incluant sa division

    Returns:
        int: Niveau (1=plus fort, 8=plus faible)
    """
    equipe_lower = equipe.lower()
    niveaux: dict[str, int] = {
        "top 16": 1, "top16": 1,
        "n1": 2, "nationale 1": 2,
        "n2": 3, "nationale 2": 3,
        "n3": 4, "nationale 3": 4,
        "n4": 5, "nationale 4": 5,
        "regionale": 6, "r1": 6, "r2": 7, "r3": 8,
    }
    for pattern, niveau in niveaux.items():
        if pattern in equipe_lower:
            return niveau
    return 10  # Inconnu = plus faible


def detecter_mouvement_joueur(
    joueur: Joueur,
    equipe_avant: str,
    equipe_apres: str,
    club: str,
) -> MouvementJoueur:
    """
    Detecte et categorise un mouvement de joueur entre equipes.

    Args:
        joueur: Instance Joueur (avec attribut elo)
        equipe_avant: Nom equipe d'origine
        equipe_apres: Nom equipe destination
        club: Nom du club

    Returns:
        MouvementJoueur avec type, equipes impactees et delta elo
    """
    niveau_avant = get_niveau_equipe(equipe_avant)
    niveau_apres = get_niveau_equipe(equipe_apres)

    if niveau_apres < niveau_avant:  # Monte (N2 -> N1)
        return MouvementJoueur(
            type="promotion",
            equipe_renforcee=equipe_apres,
            equipe_affaiblie=equipe_avant,
            impact=joueur.elo,
        )
    elif niveau_apres > niveau_avant:  # Descend (N1 -> N2)
        return MouvementJoueur(
            type="relegation",
            equipe_renforcee=equipe_apres,
            equipe_affaiblie=equipe_avant,
            impact=joueur.elo,
        )
    return MouvementJoueur(
        type="lateral",
        equipe_renforcee=None,
        equipe_affaiblie=None,
        impact=0,
    )
```

### 4.3 Features derivees

| Feature | Description |
|---------|-------------|
| `nb_joueurs_promus` | Joueurs montes d'equipe inferieure |
| `nb_joueurs_relegues` | Joueurs descendus d'equipe superieure |
| `delta_elo_moyen` | Evolution force moyenne depuis R1 |
| `stabilite_effectif` | % joueurs identiques sur 3 rondes |

---

## 5. Implementation dans ALICE

### 5.1 Validation des compositions (CE)

```python
def est_brule(
    joueur: Joueur,
    equipe: Equipe,
    historique: dict[int, dict[str, int]],
) -> bool:
    """
    Verifie si un joueur est brule pour une equipe.

    Args:
        joueur: Le joueur a verifier
        equipe: L'equipe cible
        historique: {joueur_id: {equipe_nom: nb_matchs}}

    Returns:
        True si le joueur a joue 3+ fois dans une equipe superieure
    """
    joueur_hist = historique.get(joueur.id_fide, {})
    niveau_cible = get_niveau_equipe(equipe.division)

    for eq_nom, nb_matchs in joueur_hist.items():
        niveau_eq = get_niveau_equipe(eq_nom)
        if niveau_eq < niveau_cible and nb_matchs >= 3:
            return True
    return False


def get_noyau(
    equipe: Equipe,
    historique: dict[str, set[int]],
) -> set[int]:
    """
    Retourne l'ensemble des joueurs du noyau d'une equipe.

    Args:
        equipe: L'equipe concernee
        historique: {equipe_nom: set(joueur_ids)}

    Returns:
        Set des IDs joueurs ayant deja joue pour cette equipe
    """
    return historique.get(equipe.nom, set())


def valider_composition(
    composition: list[Joueur],
    equipe: Equipe,
    historique_brulage: dict[int, dict[str, int]],
    historique_noyau: dict[str, set[int]],
    regles: ReglesCompetition,
) -> list[str]:
    """
    Valide une composition selon les regles FFE.

    Args:
        composition: Liste des joueurs alignes
        equipe: Equipe concernee
        historique_brulage: {joueur_id: {equipe_nom: nb_matchs}}
        historique_noyau: {equipe_nom: set(joueur_ids)}
        regles: Regles applicables a cette competition

    Returns:
        Liste des erreurs detectees (vide si valide)
    """
    erreurs: list[str] = []

    if not composition:
        return ["Composition vide"]

    # 3.6.e - Ordre Elo (si obligatoire)
    if regles.get("ordre_elo_obligatoire", True):
        for i in range(len(composition) - 1):
            if composition[i].elo < composition[i + 1].elo - 100:
                erreurs.append(f"Ordre Elo invalide: ech {i + 1}")

    # 3.7.c - Joueur brule
    seuil_brulage = regles.get("seuil_brulage")
    if seuil_brulage is not None:
        for j in composition:
            if est_brule(j, equipe, historique_brulage):
                erreurs.append(f"{j.nom} est brule pour {equipe.nom}")

    # 3.7.f - Noyau
    noyau_pct = regles.get("noyau")
    noyau_type = regles.get("noyau_type", "pourcentage")
    if noyau_pct is not None and equipe.ronde > 1:
        noyau_ids = get_noyau(equipe, historique_noyau)
        nb_noyau = sum(1 for j in composition if j.id_fide in noyau_ids)

        if noyau_type == "pourcentage":
            pct = nb_noyau / len(composition)
            if pct < noyau_pct / 100:
                erreurs.append(f"Noyau insuffisant: {pct:.0%} < {noyau_pct}%")
        else:  # absolu
            if nb_noyau < noyau_pct:
                erreurs.append(f"Noyau insuffisant: {nb_noyau} < {noyau_pct}")

    # 3.7.g - Mutes
    max_mutes = regles.get("max_mutes")
    if max_mutes is not None:
        nb_mutes = sum(1 for j in composition if j.mute)
        if nb_mutes > max_mutes:
            erreurs.append(f"Trop de mutes: {nb_mutes} > {max_mutes}")

    return erreurs
```

### 5.2 Features pour ALI (prediction adverse)

```python
FEATURES_REGLEMENTAIRES = [
    # Joueur brule
    "joueur_brule",                    # bool
    "matchs_avant_brulage",            # int 0-3

    # Disponibilite
    "nb_matchs_joues_saison",          # int
    "peut_jouer_ronde_n",              # bool

    # Noyau
    "est_dans_noyau",                  # bool
    "pct_noyau_equipe",                # float

    # Quotas
    "joueur_mute",                     # bool
    "nb_mutes_deja_alignes",           # int

    # Enjeu
    "zone_enjeu_equipe",               # cat: montee/danger/mi_tableau
    "ecart_objectif",                  # int (points)
]
```

---

## 6. Conformite ISO

| Norme | Application |
|-------|-------------|
| ISO 15289 | Structure documentation SyRS |
| ISO 25010 | Exactitude fonctionnelle (regles codees) |
| ISO 25012 | Qualite donnees (flags mute, nationalite) |
| ISO/IEC 5055 | Typage strict, zero any implicite |

---

## 6bis. Types de donnees (ISO/IEC 5055)

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class TypeCompetition(Enum):
    """Types de competition FFE."""

    A02 = "A02"  # CFC Hommes
    F01 = "F01"  # CFC Feminin
    C01 = "C01"  # Coupe de France
    C03 = "C03"  # Coupe Loubatiere
    C04 = "C04"  # Coupe Parite
    J02 = "J02"  # Interclubs Jeunes
    J03 = "J03"  # Scolaire
    REG = "REG"  # Regionale
    DEP = "DEP"  # Departemental


class NiveauCompetition(Enum):
    """Niveaux hierarchiques."""

    NATIONAL = "national"
    N4 = "n4"
    REGIONAL = "regional"
    DEPARTEMENTAL = "departemental"
    COUPE = "coupe"
    INCONNU = "inconnu"


class Sexe(Enum):
    """Sexe du joueur."""

    MASCULIN = "M"
    FEMININ = "F"


@dataclass(frozen=True)
class Joueur:
    """Representation d'un joueur."""

    id_fide: int
    nom: str
    elo: int
    sexe: Sexe
    nationalite: str
    mute: bool
    date_naissance: str | None = None


@dataclass(frozen=True)
class Equipe:
    """Representation d'une equipe."""

    nom: str
    club: str
    division: str
    ronde: int
    groupe: str | None = None


class ReglesCompetition(TypedDict, total=False):
    """Structure typee des regles par competition."""

    taille_equipe: int | dict[str, int]
    seuil_brulage: int | None
    max_parties_saison: int | None
    max_mutes: int | None
    min_fr_eu: int | None
    ordre_elo_obligatoire: bool
    elo_max: int | None
    elo_total_max: int | None
    noyau: int | None
    noyau_type: str  # "pourcentage" | "absolu"
    quota_nationalite: bool
    composition_obligatoire: dict[str, int] | None
    categories_age: dict[int, str] | None


class NiveauInfo(TypedDict):
    """Resultat detection niveau."""

    niveau: str
    regles: str


class MouvementJoueur(TypedDict):
    """Resultat detection mouvement."""

    type: str  # "promotion" | "relegation" | "lateral"
    equipe_renforcee: str | None
    equipe_affaiblie: str | None
    impact: int
```

---

## 7. Variations par competition

### 7.1 Championnat Feminin (F01)

Structure: Top 12F (12 eq), N1F (4x8), N2F (phases ZID)

| Regle | A02 (Hommes) | F01 (Feminin) | Impact ALICE |
|-------|--------------|---------------|--------------|
| Taille equipe | 8 joueurs | **4 joueuses** | Adapter features |
| Brulage Top | 3 matchs | **1 match** | Seuil different |
| Nb parties max | < ronde N | **7/saison** total | Compteur global |
| Mutes | Max 3 | Max **2** | Quota different |
| Nationalite | 5 FR/UE sur 8 | **3 FR/UE sur 4** | Meme ratio |
| FR obligatoire | 1M + 1F | **1F** (joueuse) | Simplifier |

```python
def get_regles_feminin() -> dict:
    return {
        "taille_equipe": 4,
        "seuil_brulage": 1,  # 1 match en Top12F = brule
        "max_parties_saison": 7,
        "max_mutes": 2,
        "min_fr_eu": 3,
    }
```

### 7.2 Coupe de France (C01)

Format: Elimination directe, equipes de 4 joueurs

| Regle | A02 (CFC) | C01 (Coupe) | Impact ALICE |
|-------|-----------|-------------|--------------|
| Taille equipe | 8 | **4** | |
| Ordre Elo | Obligatoire >100pts | **LIBRE** | Pas de validation |
| Nationalite | 5/8 FR/UE | **2/4** (50%) | Meme ratio |
| Mutes | Max 3 | Max **2** | |
| Licence | Avant saison | Avant **15 janvier** | Deadline |
| Departage | Pts match | **Ech1 > Ech2 > Ech3** | Poids echiquiers |

```python
def get_regles_coupe() -> dict:
    return {
        "taille_equipe": 4,
        "ordre_elo_obligatoire": False,  # Capitaine libre!
        "max_mutes": 2,
        "deadline_licence": "15-01",
        "departage": ["ech1", "ech2", "ech3", "elo_faible", "age_jeune"],
    }
```

### 7.3 Interclubs Jeunes (J02)

Structure: Top Jeunes (16), N1 (4x8), N2 (12x8), N3 (ligues)

| Regle | A02 (Hommes) | J02 (Jeunes) | Impact ALICE |
|-------|--------------|--------------|--------------|
| Taille equipe | 8 | 8 (Top/N1/N2), **4** (N3) | |
| Echiquiers 7-8 | 1 partie | **2 parties** | Comptage special |
| Ordre | Elo >100pts | **Age** (sauf Elo >=) | Logique inversee |
| Categories | Aucune | **U16, U14, U12, U10** par ech | Contrainte age |
| Brulage | 3 matchs | **4 matchs** nat sup | Seuil different |
| Nb parties | < ronde N | **11 max**, 7 en N1/N2 | Compteurs multiples |
| Mutes | Max 3 | Max **2** (1 en N3) | |

```python
def get_regles_jeunes() -> dict:
    return {
        "taille_equipe": {"top": 8, "n1": 8, "n2": 8, "n3": 4},
        "categories_age": {
            1: "U16", 2: "U16",
            3: "U14", 4: "U14",
            5: "U12", 6: "U12",
            7: "U10", 8: "U10",
        },
        "ordre": "age",  # Pas Elo!
        "seuil_brulage": 4,
        "max_parties_saison": 11,
        "max_parties_n1_n2": 7,
        "max_mutes": 2,
        "ech_7_8_parties": 2,  # Double partie
    }
```

### 7.4 Coupe Jean-Claude Loubatiere (C03)

Format: Competition pour joueurs bas-elo (max 1800), equipes de 4

| Regle | A02 (CFC) | C03 (Loubatiere) | Impact ALICE |
|-------|-----------|------------------|--------------|
| Taille equipe | 8 | **4** | |
| Elo max | 2400 (N4) | **1800** | Filtre strict |
| Ordre Elo | Obligatoire >100pts | **LIBRE** | Pas de validation |
| Nationalite | 5/8 FR/UE | **3/4** | |
| Etrangers max | 3 hors-quota | **1** (hors UE) | Plus restrictif |
| Phases | Groupes | **KO par phases** | Elimination |
| Bonus femmes | Non | **10% = +1 qualifie** | Promotion parite |

```python
def get_regles_loubatiere() -> dict:
    return {
        "taille_equipe": 4,
        "elo_max": 1800,  # Strict!
        "ordre_elo_obligatoire": False,
        "min_fr_eu": 3,
        "max_etrangers_hors_ue": 1,
        "phases": ["departementale", "phase2", "phase3", "finale"],
        "bonus_femmes": True,  # 10% = qualification bonus
    }
```

### 7.5 Coupe de la Parite (C04)

Format: Competition mixte obligatoire (2 hommes + 2 femmes)

| Regle | A02 (CFC) | C04 (Parite) | Impact ALICE |
|-------|-----------|--------------|--------------|
| Taille equipe | 8 | **4** (2H + 2F) | Parite stricte |
| Elo total | - | **8000 max** | Plafond equipe |
| Elo si 3 joueurs | - | **6000 max** | Ajustement |
| Ordre Elo | >100pts | **Obligatoire** | Validation |
| Nationalite | 5/8 FR/UE | **3/4** | |
| Etrangers max | 3 hors-quota | **1** (hors UE) | |
| Cadence | 2h + 30s | Rapide **25+10** ou **15+5** | Plus court |

```python
def get_regles_parite() -> dict:
    return {
        "taille_equipe": 4,
        "composition_obligatoire": {"hommes": 2, "femmes": 2},
        "elo_total_max": 8000,
        "elo_total_max_3joueurs": 6000,
        "ordre_elo_obligatoire": True,  # >100 pts
        "min_fr_eu": 3,
        "max_etrangers_hors_ue": 1,
        "cadence": "rapide",  # 25+10 ou 15+5
    }

def valider_composition_parite(composition: list[Joueur]) -> list[str]:
    """
    Valide la parite homme/femme pour la Coupe C04.

    Args:
        composition: Liste des joueurs alignes

    Returns:
        Liste des erreurs detectees (vide si valide)
    """
    erreurs: list[str] = []

    if not composition:
        return ["Composition vide"]

    hommes = sum(1 for j in composition if j.sexe == Sexe.MASCULIN)
    femmes = sum(1 for j in composition if j.sexe == Sexe.FEMININ)

    if len(composition) == 4:
        if hommes != 2 or femmes != 2:
            erreurs.append(f"Parite non respectee: {hommes}H/{femmes}F (requis: 2H/2F)")
        elo_total = sum(j.elo for j in composition)
        if elo_total > 8000:
            erreurs.append(f"Elo total {elo_total} > 8000")
    elif len(composition) == 3:
        elo_total = sum(j.elo for j in composition)
        if elo_total > 6000:
            erreurs.append(f"Elo total {elo_total} > 6000 (equipe incomplete)")

    return erreurs
```

### 7.6 Championnat Scolaire (J03)

Format: Competition scolaire, equipes d'etablissement

| Regle | J02 (Jeunes) | J03 (Scolaire) | Impact ALICE |
|-------|--------------|----------------|--------------|
| Taille equipe | 8 | **8** (min 2G + 2F) | Parite min |
| Appartenance | Club | **Etablissement** | Ecole/College/Lycee |
| Cadence | 1h30 | **Rapide** (15 KO ou 12+3s) | Beaucoup plus court |
| Elo defaut | Elo FFE | **799** (primaire), **999** (college) | Estimation |
| Ordre | Age | **Elo FIDE d'abord** | Joueurs classes devant |
| Categories | U16-U10 | **Primaire/College/Lycee** | Par niveau scolaire |
| Brulage | 4 matchs | Non applicable | Pas de multi-equipes |

```python
def get_regles_scolaire() -> dict:
    return {
        "taille_equipe": 8,
        "composition_min": {"garcons": 2, "filles": 2},
        "appartenance": "etablissement",  # Pas club!
        "cadence": "rapide",  # 15 min KO ou 12+3s
        "elo_defaut": {
            "primaire": 799,
            "college": 999,
            "lycee": None,  # Elo reel
        },
        "ordre": "elo_fide_puis_autres",
        "categories": ["primaire", "college", "lycee"],
    }
```

### 7.7 Matrice des regles par competition (complete)

| Regle | A02 | F01 | C01 | C03 | C04 | J02 | J03 |
|-------|-----|-----|-----|-----|-----|-----|-----|
| **Equipe** | 8 | 4 | 4 | 4 | 4 | 8/4 | 8 |
| **Ordre** | Elo>100 | Elo>100 | Libre | Libre | Elo>100 | Age | Elo FIDE |
| **Brulage** | 3 matchs | 1 match | - | - | - | 4 matchs | - |
| **Noyau 50%** | Oui | - | - | - | - | - | - |
| **Mutes max** | 3 | 2 | 2 | - | - | 2 | - |
| **FR/UE min** | 5/8 | 3/4 | 2/4 | 3/4 | 3/4 | - | - |
| **Parties max** | <ronde | 7 | - | - | - | 11 | - |
| **Elo max** | 2400(N4) | - | - | 1800 | 8000 tot | - | - |
| **Parite** | - | - | - | Bonus | **2H+2F** | - | Min 2G+2F |
| **Cadence** | 2h+30s | 2h+30s | 2h+30s | - | 25+10 | 1h30 | 15 KO |

---

## 8. Etat d'implementation

| Tache | Statut | Fichier |
|-------|--------|---------|
| Types et dataclasses stricts | FAIT | `scripts/ffe_rules_features.py` |
| Detection type competition | FAIT | `detecter_type_competition()` |
| Regles par competition | FAIT | `get_regles_competition()` |
| Calcul joueur brule | FAIT | `est_brule()`, `matchs_avant_brulage()` |
| Calcul noyau | FAIT | `get_noyau()`, `calculer_pct_noyau()`, `valide_noyau()` |
| Validation composition | FAIT | `valider_composition()` |
| Features reglementaires ML | FAIT | `feature_engineering.py` |
| Tests unitaires | FAIT | `tests/test_ffe_rules_features.py` (66 tests) |

### Prochaines etapes

1. [ ] Enrichir dataset avec features calculees sur historique complet
2. [ ] Valider detection infractions sur donnees historiques
3. [ ] Integrer validation CE dans API FastAPI
4. [ ] Ajouter metriques de conformite reglementaire

---

## 9. Detection du type de competition (COMPLET)

Pour appliquer les bonnes regles, ALICE doit detecter le type de competition:

```python
def detecter_type_competition(nom_competition: str) -> TypeCompetition:
    """
    Detecte le type de competition pour appliquer les bonnes regles.

    Args:
        nom_competition: Nom de la competition (ex: "Nationale 2", "Coupe Loubatiere")

    Returns:
        TypeCompetition enum correspondant

    Raises:
        None - retourne A02 par defaut si non detecte
    """
    nom = nom_competition.lower()

    # Coupes speciales (prioritaire - detecter avant "coupe france")
    if "loubatiere" in nom or "loubatière" in nom:
        return TypeCompetition.C03
    if "parite" in nom or "parité" in nom:
        return TypeCompetition.C04

    # Feminin
    if "feminin" in nom or "féminin" in nom or "12f" in nom or "feminine" in nom:
        return TypeCompetition.F01

    # Coupe de France (apres coupes speciales)
    if "coupe" in nom and "france" in nom:
        return TypeCompetition.C01

    # Scolaire (avant jeunes general)
    if "scolaire" in nom or "etablissement" in nom or "école" in nom:
        return TypeCompetition.J03

    # Jeunes
    if "jeune" in nom or "junior" in nom or "top jeunes" in nom:
        return TypeCompetition.J02

    # Regional / Departemental
    if any(x in nom for x in ["regionale", "régionale", "r1 ", "r2 ", "r3 "]):
        return TypeCompetition.REG
    if any(x in nom for x in ["departemental", "départemental", "n5", "n6", "criterium"]):
        return TypeCompetition.DEP

    # National (default pour Top16-N4)
    if any(x in nom for x in ["top 16", "n1", "n2", "n3", "n4", "nationale"]):
        return TypeCompetition.A02

    return TypeCompetition.A02  # Default


def get_regles_a02() -> ReglesCompetition:
    """Regles du Championnat de France des Clubs (A02)."""
    return ReglesCompetition(
        taille_equipe=8,
        seuil_brulage=3,
        max_parties_saison=None,  # < numero_ronde
        max_mutes=3,
        min_fr_eu=5,
        ordre_elo_obligatoire=True,
        elo_max=2400,  # N4 uniquement
        noyau=50,
        noyau_type="pourcentage",
        quota_nationalite=True,
    )


def get_regles_competition(type_comp: TypeCompetition) -> ReglesCompetition:
    """
    Retourne les regles applicables selon le type de competition.

    Args:
        type_comp: Type de competition (enum TypeCompetition)

    Returns:
        ReglesCompetition avec toutes les regles applicables
    """
    REGLES: dict[TypeCompetition, ReglesCompetition] = {
        TypeCompetition.A02: get_regles_a02(),
        TypeCompetition.F01: ReglesCompetition(
            taille_equipe=4,
            seuil_brulage=1,
            max_parties_saison=7,
            max_mutes=2,
            min_fr_eu=3,
            ordre_elo_obligatoire=True,
            quota_nationalite=True,
        ),
        TypeCompetition.C01: ReglesCompetition(
            taille_equipe=4,
            seuil_brulage=None,
            max_mutes=2,
            min_fr_eu=2,
            ordre_elo_obligatoire=False,
            quota_nationalite=True,
        ),
        TypeCompetition.C03: ReglesCompetition(
            taille_equipe=4,
            elo_max=1800,
            ordre_elo_obligatoire=False,
            min_fr_eu=3,
            max_mutes=None,
        ),
        TypeCompetition.C04: ReglesCompetition(
            taille_equipe=4,
            elo_total_max=8000,
            ordre_elo_obligatoire=True,
            min_fr_eu=3,
            composition_obligatoire={"hommes": 2, "femmes": 2},
        ),
        TypeCompetition.J02: ReglesCompetition(
            taille_equipe={"top": 8, "n1": 8, "n2": 8, "n3": 4},
            seuil_brulage=4,
            max_parties_saison=11,
            max_mutes=2,
            ordre_elo_obligatoire=False,  # Ordre par age
            categories_age={1: "U16", 2: "U16", 3: "U14", 4: "U14", 5: "U12", 6: "U12", 7: "U10", 8: "U10"},
        ),
        TypeCompetition.J03: ReglesCompetition(
            taille_equipe=8,
            ordre_elo_obligatoire=False,  # Elo FIDE d'abord
            composition_obligatoire={"garcons": 2, "filles": 2},
        ),
        TypeCompetition.REG: ReglesCompetition(
            taille_equipe=5,
            noyau=2,
            noyau_type="absolu",
            quota_nationalite=False,
        ),
        TypeCompetition.DEP: ReglesCompetition(
            taille_equipe=4,
            quota_nationalite=False,
        ),
    }
    return REGLES.get(type_comp, get_regles_a02())
```

---

## 10. Reglements regionaux et departementaux

Les niveaux sous N4 ont des variations significatives selon les ligues.

### 10.1 Hierarchie des niveaux

```
Federal:      Top16 > N1 > N2 > N3 > N4
Regional:     N4 regional > Regionale (R1, R2, R3...)
Departemental: Departemental (N5, N6...) > Critérium
```

### 10.2 N4 Regional (PACA)

| Regle | A02 (Federal) | N4 PACA | Impact |
|-------|---------------|---------|--------|
| Base reglementaire | A02 complet | **A02 par defaut** | Heritage |
| Noyau 50% | Oui N1-N3 | **Oui** (idem A02) | Applique |
| Elo max | 2400 | **2400** | Idem |
| Structure | Variable | **5 groupes x 8 eq** | 5 montees N3 |

```python
def get_regles_n4_regional() -> dict:
    """N4 herite de A02 avec adaptations ligues."""
    base = get_regles_a02()
    base.update({
        "niveau": "N4",
        "noyau_50pct": True,  # Confirme par ligue PACA
        "elo_max": 2400,
        "groupes": 5,
        "equipes_par_groupe": 8,
    })
    return base
```

### 10.3 Regionale (PACA)

Differences majeures avec le niveau national:

| Regle | A02 (National) | Regionale PACA | Impact ALICE |
|-------|----------------|----------------|--------------|
| Taille equipe | 8 | **5** | Moins de joueurs |
| Noyau | 50% (4/8) | **2 joueurs** | Seuil absolu |
| Nb parties max | <ronde | **7** ou **9** | Selon divisions |
| Quota nationalite | Obligatoire | **Non applicable** | Pas de filtre |
| Arbitre | Obligatoire | Facultatif | - |

```python
def get_regles_regionale() -> dict:
    return {
        "taille_equipe": 5,
        "noyau": 2,  # Absolu, pas %
        "noyau_type": "absolu",
        "max_parties": {
            "si_n4_et_reg": 7,
            "si_nat_sup_et_reg": 9,
        },
        "quota_nationalite": False,  # Non applicable!
        "arbitre_obligatoire": False,
    }
```

### 10.4 Departemental adultes (Bouches-du-Rhone - N6)

| Regle | Regionale | Departemental BdR | Impact ALICE |
|-------|-----------|-------------------|--------------|
| Taille equipe | 5 | **4** | Encore moins |
| Noyau | 2 joueurs | Non specifie | Simplification |
| Renfort Elo | - | **1 joueur >1500 apres R3** | Restriction tardive |
| Quota nationalite | Non | **Non** | Pas de filtre |
| Arbitre | Facultatif | **Non requis** | - |

```python
def get_regles_departemental() -> dict:
    return {
        "taille_equipe": 4,
        "noyau": None,  # Pas de regle noyau
        "restriction_elo": {
            "apres_ronde": 3,
            "max_nouveaux_joueurs_elo_sup": 1,
            "seuil_elo": 1500,
        },
        "quota_nationalite": False,
        "arbitre_obligatoire": False,
    }
```

### 10.5 Interclubs Jeunes departemental (PACA BdR)

| Regle | J02 (Federal) | Jeunes BdR | Impact ALICE |
|-------|---------------|------------|--------------|
| Taille equipe | 8 (Top/N1/N2) | **4** | Niveau N3 equiv |
| Base reglement | J02 | **J02** par defaut | Heritage |
| Ech U10 | 1 joueur | **2 joueurs possibles** | Flexibilite |
| Mutes | Max 2 | **Pas de limite** si nouveau club | Ouverture |
| Categories | U16/U14/U12/U10 | **U16/U14/U12/U10** | Idem |

```python
def get_regles_jeunes_departemental() -> dict:
    base = get_regles_jeunes()
    base.update({
        "taille_equipe": 4,  # Niveau N3
        "niveau": "N3_departemental",
        "u10_double_joueur": True,  # 2 joueurs pour 2 parties
        "mutes_nouveaux_clubs": "illimite",
    })
    return base
```

### 10.6 Matrice des regles par niveau

| Niveau | Equipe | Noyau | Brulage | Nat. | Elo max |
|--------|--------|-------|---------|------|---------|
| **Top16-N3** | 8 | 50% | 3 matchs | 5/8 | - |
| **N4 federal** | 8 | 50% | 3 matchs | 5/8 | 2400 |
| **N4 regional** | 8 | 50% | 3 matchs | 5/8 | 2400 |
| **Regionale** | **5** | **2 abs** | - | **Non** | - |
| **Departemental** | **4** | - | - | Non | 1500* |

*Restriction apres ronde 3

### 10.7 Utilisation

Voir Section 9 pour les fonctions de detection typees (`detecter_type_competition`, `get_regles_competition`).

---

## 11. Implementation unifiee

### 11.1 Fonction principale de validation

```python
def get_regles_pour_match(
    competition: str,
    division: str,
    ligue: str | None = None,
) -> ReglesCompetition:
    """
    Retourne les regles applicables pour un match.

    Args:
        competition: Nom de la competition
        division: Division (ex: "N2", "Regionale")
        ligue: Code ligue optionnel (ex: "PACA")

    Returns:
        ReglesCompetition avec toutes les regles applicables
    """
    # Detecter type via fonction Section 9
    type_comp = detecter_type_competition(competition)

    # Charger regles correspondantes
    return get_regles_competition(type_comp)
```

### 11.2 Features a extraire selon niveau

```python
FEATURES_PAR_NIVEAU: dict[str, list[str]] = {
    "national": [
        "joueur_brule",
        "matchs_avant_brulage",
        "pct_noyau",
        "nb_mutes",
        "quota_nationalite_ok",
        "joueuse_fr_presente",
    ],
    "regional": [
        "noyau_2_present",  # Absolu, pas %
        "nb_parties_saison",
    ],
    "departemental": [
        "nouveau_joueur_fort_apres_r3",  # Restriction Elo
    ],
    "coupe_loubatiere": [
        "tous_elo_sous_1800",
    ],
    "coupe_parite": [
        "parite_2h_2f",
        "elo_total_equipe",
    ],
}
```

---

*Document genere le 4 Janvier 2026*
*MAJ le 4 Janvier 2026 - Version 2.0 complete: Federal + Regional + Departemental*
*Source: Reglements FFE 2025-2026 + Ligues PACA + BdR*
