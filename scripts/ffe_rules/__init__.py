"""Package FFE Rules - ISO 5055.

Ce package contient les regles FFE refactorisees en modules:
- types.py: Types, Enums, dataclasses
- competition.py: Detection type, regles par competition
- brulage.py: Regles brulage et noyau
- zones.py: Zones d'enjeu, mouvements joueurs
- validation.py: Validation composition
- features.py: Calcul features ML
"""

from scripts.ffe_rules.brulage import (
    calculer_pct_noyau,
    est_brule,
    get_noyau,
    matchs_avant_brulage,
    valide_noyau,
)
from scripts.ffe_rules.competition import (
    detecter_type_competition,
    get_niveau_equipe,
    get_regles_a02,
    get_regles_competition,
    get_regles_coupe,
    get_regles_departemental,
    get_regles_feminin,
    get_regles_jeunes,
    get_regles_loubatiere,
    get_regles_parite,
    get_regles_regionale,
    get_regles_scolaire,
)
from scripts.ffe_rules.features import calculer_features_joueur
from scripts.ffe_rules.types import (
    Equipe,
    FeaturesReglementaires,
    HistoriqueJoueur,
    Joueur,
    MouvementJoueur,
    NiveauCompetition,
    ReglesCompetition,
    Sexe,
    TypeCompetition,
)
from scripts.ffe_rules.validation import valider_composition
from scripts.ffe_rules.zones import (
    calculer_ecart_objectif,
    calculer_zone_enjeu,
    detecter_mouvement_joueur,
)

__all__ = [
    # Types
    "TypeCompetition",
    "NiveauCompetition",
    "Sexe",
    "ReglesCompetition",
    "MouvementJoueur",
    "FeaturesReglementaires",
    "Joueur",
    "Equipe",
    "HistoriqueJoueur",
    # Detection
    "detecter_type_competition",
    "get_niveau_equipe",
    # Regles
    "get_regles_competition",
    "get_regles_a02",
    "get_regles_feminin",
    "get_regles_coupe",
    "get_regles_loubatiere",
    "get_regles_parite",
    "get_regles_jeunes",
    "get_regles_scolaire",
    "get_regles_regionale",
    "get_regles_departemental",
    # Brulage
    "est_brule",
    "matchs_avant_brulage",
    # Noyau
    "get_noyau",
    "calculer_pct_noyau",
    "valide_noyau",
    # Enjeu
    "calculer_zone_enjeu",
    "calculer_ecart_objectif",
    # Mouvement
    "detecter_mouvement_joueur",
    # Validation
    "valider_composition",
    # Features
    "calculer_features_joueur",
]
