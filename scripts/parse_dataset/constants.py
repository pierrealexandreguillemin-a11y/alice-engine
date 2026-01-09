"""Constantes pour parsing dataset FFE - ISO 5055.

Ce module contient les mappings et constantes:
- Titres FIDE
- Ligues régionales
- Catégories d'âge FFE
- Types de compétitions

Conformité ISO/IEC 5055, 25010.
"""

from __future__ import annotations

from typing import Any

# Mapping titres FIDE (minuscules dans HTML -> standard)
TITRES_FIDE: dict[str, str] = {
    "g": "GM",
    "m": "IM",
    "f": "FM",
    "c": "CM",
    "ff": "WFM",
    "mf": "WIM",
    "gf": "WGM",
    "cf": "WCM",
}

# Mapping ligues regionales (dossier -> code)
LIGUES_REGIONALES: dict[str, str] = {
    "Auvergne-Rhone-Alpes": "ARA",
    "Bourgogne_Franche-Comte": "BFC",
    "Bretagne": "BRE",
    "Corse": "COR",
    "Guyane": "GUY",
    "Reunion": "REU",
    "Ile_de_France": "IDF",
    "Normandie": "NOR",
    "Nouvelle_Aquitaine": "NAQ",
    "Provence_-_Alpes_-_Cote_d'Azur": "PACA",
    "Hauts-De-France": "HDF",
    "Pays_de_la_Loire": "PDL",
    "Occitanie": "OCC",
    "Centre_Val_de_Loire": "CVL",
    "Grand_Est": "GES",
}

# Mapping categories joueurs FFE (format legacy HTML -> ages officiels FFE)
# Source: Reglement FFE 2.4 Categories d'age
# Format donnees: {Cat}{Genre} ex: SenM, PouF
# Genre: M=Masculin, F=Feminin
CATEGORIES_AGE: dict[str, dict[str, Any]] = {
    # U8 - Petits Poussins (moins de 8 ans)
    "PpoM": {"age_max": 7, "genre": "M", "code_ffe": "U8"},
    "PpoF": {"age_max": 7, "genre": "F", "code_ffe": "U8F"},
    # U10 - Poussins (8 ou 9 ans)
    "PouM": {"age_min": 8, "age_max": 9, "genre": "M", "code_ffe": "U10"},
    "PouF": {"age_min": 8, "age_max": 9, "genre": "F", "code_ffe": "U10F"},
    # U12 - Pupilles (10 ou 11 ans)
    "PupM": {"age_min": 10, "age_max": 11, "genre": "M", "code_ffe": "U12"},
    "PupF": {"age_min": 10, "age_max": 11, "genre": "F", "code_ffe": "U12F"},
    # U14 - Benjamins (12 ou 13 ans)
    "BenM": {"age_min": 12, "age_max": 13, "genre": "M", "code_ffe": "U14"},
    "BenF": {"age_min": 12, "age_max": 13, "genre": "F", "code_ffe": "U14F"},
    # U16 - Minimes (14 ou 15 ans)
    "MinM": {"age_min": 14, "age_max": 15, "genre": "M", "code_ffe": "U16"},
    "MinF": {"age_min": 14, "age_max": 15, "genre": "F", "code_ffe": "U16F"},
    # U18 - Cadets (16 ou 17 ans)
    "CadM": {"age_min": 16, "age_max": 17, "genre": "M", "code_ffe": "U18"},
    "CadF": {"age_min": 16, "age_max": 17, "genre": "F", "code_ffe": "U18F"},
    # U20 - Juniors (18 ou 19 ans)
    "JunM": {"age_min": 18, "age_max": 19, "genre": "M", "code_ffe": "U20"},
    "JunF": {"age_min": 18, "age_max": 19, "genre": "F", "code_ffe": "U20F"},
    # Seniors (20 a 49 ans)
    "SenM": {"age_min": 20, "age_max": 49, "genre": "M", "code_ffe": "Sen"},
    "SenF": {"age_min": 20, "age_max": 49, "genre": "F", "code_ffe": "Sen"},
    # S50 - Seniors Plus (50 a 64 ans)
    "SepM": {"age_min": 50, "age_max": 64, "genre": "M", "code_ffe": "S50"},
    "SepF": {"age_min": 50, "age_max": 64, "genre": "F", "code_ffe": "S50"},
    # S65 - Veterans (65 ans et plus)
    "VetM": {"age_min": 65, "genre": "M", "code_ffe": "S65"},
    "VetF": {"age_min": 65, "genre": "F", "code_ffe": "S65"},
}

# Types de competitions
TYPES_COMPETITION: dict[str, str] = {
    "Interclubs": "national",
    "Interclubs_Feminins": "national_feminin",
    "Interclubs_Jeunes": "national_jeunes",
    "Interclubs_Rapide": "national_rapide",
    "Coupe_de_France": "coupe",
    "Coupe_de_la_Parite": "coupe_parite",
    "Coupe_Jean-Claude_Loubatiere": "coupe_jcl",
    "Competitions_Scolaires": "scolaire",
}
