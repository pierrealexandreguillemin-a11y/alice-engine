"""Tests FFE Rules Features Calc - ISO 29119/5259.

Document ID: ALICE-TEST-FFE-CALC
Version: 1.0.0
Tests: 9

Classes:
- TestCalculerFeaturesJoueur: Tests calcul features (4 tests)
- TestCalculerMatchsRestants: Tests matchs restants (3 tests)
- TestPeutJouerRonde: Tests disponibilité ronde (2 tests)

ISO Compliance:
- ISO/IEC 29119:2022 - Software Testing
- ISO/IEC 5259:2024 - Data Quality (règles métier)
- ISO/IEC 5055:2021 - Code Quality (<80 lignes)

Author: ALICE Engine Team
Last Updated: 2026-01-14
"""

from scripts.ffe_rules.features import (
    _calculer_matchs_restants,
    _peut_jouer_ronde,
    calculer_features_joueur,
)


class TestCalculerFeaturesJoueur:
    """Tests pour calculer_features_joueur."""

    def test_joueur_non_brule(self):
        """Joueur non brûlé avec peu de matchs."""
        regles = {"seuil_brulage": 3, "max_parties_saison": 10}
        historique_brulage = {123: {"Equipe A": 1}}
        historique_noyau = {"Equipe B": {123, 456}}
        historique_parties = {123: 2}

        result = calculer_features_joueur(
            joueur_id=123,
            equipe_nom="Equipe B",
            ronde=3,
            historique_brulage=historique_brulage,
            historique_noyau=historique_noyau,
            historique_parties=historique_parties,
            regles=regles,
        )

        assert result["joueur_brule"] is False
        assert result["est_dans_noyau"] is True
        assert result["peut_jouer_ronde_n"] is True

    def test_joueur_sans_historique(self):
        """Joueur sans historique."""
        regles = {"seuil_brulage": 3}

        result = calculer_features_joueur(
            joueur_id=999,
            equipe_nom="Equipe X",
            ronde=1,
            historique_brulage={},
            historique_noyau={},
            historique_parties={},
            regles=regles,
        )

        assert result["joueur_brule"] is False
        assert result["nb_matchs_joues_saison"] == 0

    def test_sans_seuil_brulage(self):
        """Pas de brûlage si seuil non défini."""
        regles = {}  # Pas de seuil_brulage

        result = calculer_features_joueur(
            joueur_id=123,
            equipe_nom="Equipe",
            ronde=5,
            historique_brulage={123: {"Top": 10}},
            historique_noyau={},
            historique_parties={},
            regles=regles,
        )

        assert result["joueur_brule"] is False


class TestCalculerMatchsRestants:
    """Tests pour _calculer_matchs_restants."""

    def test_sans_seuil(self):
        """Retourne 0 si pas de seuil."""
        result = _calculer_matchs_restants(123, "Equipe", {}, None)
        assert result == 0

    def test_joueur_sans_historique(self):
        """Joueur inconnu = seuil complet."""
        result = _calculer_matchs_restants(999, "Equipe", {}, 3)
        assert result == 3

    def test_avec_historique_meme_niveau(self):
        """Matchs au même niveau ne comptent pas."""
        historique = {123: {"N2 1": 2}}  # Même niveau
        result = _calculer_matchs_restants(123, "N2 2", historique, 3)
        # N2 1 et N2 2 sont au même niveau, donc pas de brûlage
        assert result == 3


class TestPeutJouerRonde:
    """Tests pour _peut_jouer_ronde."""

    def test_avec_max_parties(self):
        """Respecte max parties saison."""
        assert _peut_jouer_ronde(5, 10, 10) is True
        assert _peut_jouer_ronde(10, 10, 10) is False

    def test_sans_max_parties(self):
        """Utilise ronde comme limite."""
        assert _peut_jouer_ronde(3, 5, None) is True
        assert _peut_jouer_ronde(5, 5, None) is False
