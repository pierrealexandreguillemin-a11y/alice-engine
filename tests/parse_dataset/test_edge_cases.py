"""Tests Edge Cases - ISO 29119.

Document ID: ALICE-TEST-PARSE-EDGE-CASES
Version: 1.0.0

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5055:2021 - Code Quality (<300 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-15
"""

from scripts.parse_dataset import (
    CATEGORIES_AGE,
    parse_player_text,
    parse_result,
)


class TestEdgeCases:
    """Tests edge cases."""

    def test_parse_result_nulle_unicode(self) -> None:
        """Test nulle avec caractere unicode."""
        score_b, score_n, type_res = parse_result("½")
        assert score_b == 0.5
        assert score_n == 0.5
        assert type_res == "nulle"

    def test_parse_player_text_composite_name(self) -> None:
        """Test parsing nom compose."""
        result = parse_player_text("g VAN DER WIEL John  2550")
        assert result.titre_fide == "GM"
        assert result.elo == 2550

    def test_parse_result_case_insensitive(self) -> None:
        """Test resultat insensible a la casse."""
        _, _, type_res1 = parse_result("f - 1")
        _, _, type_res2 = parse_result("F - 1")
        assert type_res1 == type_res2 == "forfait_blanc"

    def test_parse_result_zero_zero_is_non_joue(self) -> None:
        """Test 0-0 est classifie comme non_joue, pas nulle."""
        score_b, score_n, type_res = parse_result("0 - 0")
        assert score_b == 0.0
        assert score_n == 0.0
        assert type_res == "non_joue"

    def test_parse_result_zero_zero_compact(self) -> None:
        """Test 0-0 format compact."""
        score_b, score_n, type_res = parse_result("0-0")
        assert type_res == "non_joue"

    def test_categories_paired_genders(self) -> None:
        """Test que chaque categorie a un equivalent masculin et feminin."""
        male_cats = {k for k in CATEGORIES_AGE if k.endswith("M")}
        female_cats = {k for k in CATEGORIES_AGE if k.endswith("F")}
        for m_cat in male_cats:
            f_cat = m_cat[:-1] + "F"
            assert f_cat in female_cats
