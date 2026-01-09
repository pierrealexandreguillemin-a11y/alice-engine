"""Parse dataset FFE - ISO 5055 modular structure.

Ce package parse les fichiers HTML FFE et exporte en Parquet:
- compositions.py: Parsing principal matchs/groupes
- calendrier.py: Parsing calendrier.html
- ronde.py: Parsing ronde_N.html
- players.py: Parsing joueurs licencies
- parsing_utils.py: Fonctions helper parsing
- dataclasses.py: Structures de donnees
- constants.py: Mappings et constantes

Conformite ISO/IEC 5055, 25010, 25012.
"""

# Calendrier
from scripts.parse_dataset.calendrier import parse_calendrier

# Compositions
from scripts.parse_dataset.compositions import (
    extract_metadata_from_path,
    parse_groupe,
)
from scripts.parse_dataset.constants import (
    CATEGORIES_AGE,
    LIGUES_REGIONALES,
    TITRES_FIDE,
    TYPES_COMPETITION,
)

# Dataclasses
from scripts.parse_dataset.dataclasses import (
    Echiquier,
    Joueur,
    JoueurLicencie,
    Match,
    Metadata,
)

# Orchestration
from scripts.parse_dataset.orchestration import (
    find_all_groupes,
    find_player_pages,
    parse_compositions,
    parse_joueurs,
    test_parse_groupe,
)

# Parsing utils
from scripts.parse_dataset.parsing_utils import (
    parse_board_number,
    parse_elo_value,
    parse_player_text,
    parse_result,
)

# Players
from scripts.parse_dataset.players import (
    joueur_to_dict,
    parse_player_page,
)

# Ronde
from scripts.parse_dataset.ronde import parse_ronde

__all__ = [
    # Constants
    "TITRES_FIDE",
    "LIGUES_REGIONALES",
    "CATEGORIES_AGE",
    "TYPES_COMPETITION",
    # Dataclasses
    "Joueur",
    "Echiquier",
    "Match",
    "Metadata",
    "JoueurLicencie",
    # Parsing utils
    "parse_player_text",
    "parse_result",
    "parse_board_number",
    "parse_elo_value",
    # Compositions
    "extract_metadata_from_path",
    "parse_calendrier",
    "parse_ronde",
    "parse_groupe",
    # Players
    "parse_player_page",
    "joueur_to_dict",
    # Orchestration
    "find_all_groupes",
    "find_player_pages",
    "parse_compositions",
    "parse_joueurs",
    "test_parse_groupe",
]
