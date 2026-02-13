"""FFE regulatory constants for ALICE Engine training data validation.

Based on FFE (Federation Francaise des Echecs) regulations:
- R01_2025_26: Regles generales
- A02_2025_26: Championnat de France des Clubs
- J02_2025_26: Championnat de France Interclubs Jeunes
- J03_2025_26: Championnat de France scolaire
- Reglement N4 PACA 2024-2025
- Reglement Regionale PACA 2024-2025

ISO Compliance:
- ISO/IEC 5259:2024 - Data Quality for ML (Lineage, Validation)
- ISO/IEC 42001:2023 - AI Management System
"""

# =============================================================================
# FFE REGULATORY CONSTANTS (Source: reglements FFE 2025-26)
# =============================================================================

# Competition hierarchy (niveau) - A02 Art. 1.1
# NOTE: niveau is used for hierarchy (0-13) OR Elo limits for regional cups
NIVEAU_HIERARCHY = {
    "TOP16": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 4,
    "REG": 5,
    "DEP": 6,
    "TOP_JEUNES": 10,
    "N1_JEUNES": 11,
    "N2_JEUNES": 12,
    "N3_JEUNES": 13,
}

# Valid niveau for hierarchy (not Elo-based cups)
NIVEAU_HIERARCHY_MAX = 13

# Elo-based regional cups (Coupe 1500, Coupe 2000, etc.)
NIVEAU_ELO_CUPS = [1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200]

# Competition types (reglements FFE)
# NOTE: "national" removed - not present in actual data
COMPETITION_TYPES = [
    "national_feminin",
    "coupe",
    "coupe_jcl",
    "coupe_parite",
    "national_jeunes",
    "scolaire",
    "regional",
    "autre",
]

# Age categories (R01 Art. 2.4)
AGE_CATEGORIES = [
    "U8",
    "U8F",
    "U10",
    "U10F",
    "U12",
    "U12F",
    "U14",
    "U14F",
    "U16",
    "U16F",
    "U18",
    "U18F",
    "U20",
    "U20F",
    "Sen",
    "S50",
    "S65",
]

# FIDE titles (verified against actual data)
FIDE_TITLES = ["", "CM", "FM", "IM", "GM", "WFM", "WIM", "WGM"]

# Elo constraints (R01 Art. 5, FIDE Ch. 6.1)
ELO_MIN_ESTIME = 799
ELO_MAX_INITIAL = 2200
ELO_FLOOR_FIDE = 1400
ELO_MAX_REASONABLE = 2900

# Game scores by competition type
# Adultes (A02 Art. 4.1): victoire=1, nulle=0.5, defaite=0
# Jeunes ech. 1-6 (J02 Art. 4.1): victoire=2, defaite=0
# Jeunes ech. 7-8 (J02 Art. 4.1): victoire=1, defaite=0
VALID_GAME_SCORES_ADULTES = [0.0, 0.5, 1.0]
VALID_GAME_SCORES_JEUNES_HIGH = [0.0, 2.0]
VALID_GAME_SCORES_JEUNES_LOW = [0.0, 1.0]
VALID_GAME_SCORES_ALL = [0.0, 0.5, 1.0, 2.0]

# Valid result sums (resultat_blanc + resultat_noir)
# 0.0 = ajournement/forfait double
# 1.0 = Adultes standard, Jeunes ech 7-8
# 2.0 = Jeunes ech 1-6
VALID_RESULT_SUMS = [0.0, 1.0, 2.0]

# Match points (A02 Art. 4.2)
VALID_MATCH_POINTS = [0, 1, 2, 3]

# Result types (verified against actual data)
VALID_RESULT_TYPES = [
    "victoire_blanc",
    "victoire_noir",
    "nulle",
    "victoire_blanc_ajournement",
    "victoire_noir_ajournement",
    "ajournement",
]

# Strategic zones (verified against actual data)
VALID_ZONES_ENJEU = ["mi_tableau"]  # Only value in current data

# Board/round constraints
ECHIQUIER_MIN = 1
ECHIQUIER_MAX_ABSOLUTE = 16
ECHIQUIER_JEUNES_HIGH_BOARDS = 6  # J02 Art. 4.1: boards 1-6 have win=2
RONDE_MIN = 1
RONDE_MAX_ABSOLUTE = 18

# FFE Regulatory thresholds (A02 Art. 3.7.j)
NIVEAU_N4 = 4  # N4 = niveau 4
ELO_MAX_N4_PLUS = 2400  # Elo > 2400 interdit en N4+

# ISO 5259 Quality thresholds
QUALITY_VALIDATION_RATE_THRESHOLD = 0.95  # 95% valid required
MAX_SAMPLE_VALUES = 5  # Max sample values to store per error
