"""Tests Constants - ISO 29119.

Document ID: ALICE-TEST-PARSE-CONSTANTS
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from scripts.parse_dataset import (
    CATEGORIES_AGE,
    LIGUES_REGIONALES,
    TITRES_FIDE,
    TYPES_COMPETITION,
)

# ==============================================================================
# TESTS CONSTANTS
# ==============================================================================


class TestConstants:
    """Tests pour les constantes du module."""

    def test_titres_fide_lowercase_keys(self) -> None:
        """Test que toutes les clés sont en minuscules."""
        for key in TITRES_FIDE:
            assert key == key.lower()

    def test_titres_fide_uppercase_values(self) -> None:
        """Test que toutes les valeurs sont en majuscules."""
        for value in TITRES_FIDE.values():
            assert value == value.upper()

    def test_titres_fide_standard_titles(self) -> None:
        """Test les titres FIDE standards."""
        assert TITRES_FIDE["g"] == "GM"
        assert TITRES_FIDE["m"] == "IM"
        assert TITRES_FIDE["f"] == "FM"
        assert TITRES_FIDE["c"] == "CM"

    def test_titres_fide_women_titles(self) -> None:
        """Test les titres FIDE féminins."""
        assert TITRES_FIDE["gf"] == "WGM"
        assert TITRES_FIDE["mf"] == "WIM"
        assert TITRES_FIDE["ff"] == "WFM"
        assert TITRES_FIDE["cf"] == "WCM"

    def test_ligues_regionales_all_codes(self) -> None:
        """Test que tous les codes régionaux sont présents."""
        expected_codes = {
            "ARA",
            "BFC",
            "BRE",
            "COR",
            "GUY",
            "REU",
            "IDF",
            "NOR",
            "NAQ",
            "PACA",
            "HDF",
            "PDL",
            "OCC",
            "CVL",
            "GES",
        }
        actual_codes = set(LIGUES_REGIONALES.values())
        assert actual_codes == expected_codes

    def test_ligues_regionales_idf(self) -> None:
        """Test code Île-de-France."""
        assert LIGUES_REGIONALES["Ile_de_France"] == "IDF"

    def test_categories_age_seniors(self) -> None:
        """Test catégorie Seniors."""
        assert "SenM" in CATEGORIES_AGE
        assert CATEGORIES_AGE["SenM"]["age_min"] == 20
        assert CATEGORIES_AGE["SenM"]["age_max"] == 49
        assert CATEGORIES_AGE["SenM"]["genre"] == "M"

    def test_categories_age_all_have_code_ffe(self) -> None:
        """Test que toutes les catégories ont un code FFE."""
        for cat, info in CATEGORIES_AGE.items():
            assert "code_ffe" in info, f"Catégorie {cat} manque code_ffe"
            assert "genre" in info, f"Catégorie {cat} manque genre"

    def test_categories_age_youth_categories(self) -> None:
        """Test les catégories jeunes."""
        # Poussins U10
        assert CATEGORIES_AGE["PouM"]["code_ffe"] == "U10"
        assert CATEGORIES_AGE["PouM"]["age_min"] == 8
        assert CATEGORIES_AGE["PouM"]["age_max"] == 9

        # Minimes U16
        assert CATEGORIES_AGE["MinM"]["code_ffe"] == "U16"
        assert CATEGORIES_AGE["MinM"]["age_min"] == 14
        assert CATEGORIES_AGE["MinM"]["age_max"] == 15

    def test_types_competition_national(self) -> None:
        """Test types de compétition nationaux."""
        assert TYPES_COMPETITION["Interclubs"] == "national"
        assert TYPES_COMPETITION["Interclubs_Feminins"] == "national_feminin"
        assert TYPES_COMPETITION["Coupe_de_France"] == "coupe"


# ==============================================================================
# TESTS DATACLASSES
# ==============================================================================
