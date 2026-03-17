"""Tests for the NCAAB prediction engine."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from line_tracker.model import predict as mod
from line_tracker.model.features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Probability conversion functions
# ---------------------------------------------------------------------------


class TestMarginToMlProb:
    def test_zero_margin_gives_half(self):
        assert mod.margin_to_ml_prob(0.0) == pytest.approx(0.5)

    def test_positive_margin_above_half(self):
        # margin = sigma → Φ(1) ≈ 0.8413
        assert mod.margin_to_ml_prob(10.5) == pytest.approx(0.8413, abs=1e-3)

    def test_negative_margin_below_half(self):
        prob = mod.margin_to_ml_prob(-10.5)
        assert prob == pytest.approx(1 - 0.8413, abs=1e-3)

    def test_custom_sigma(self):
        # With sigma=5, margin=5 → Φ(1) ≈ 0.8413
        assert mod.margin_to_ml_prob(5.0, sigma=5.0) == pytest.approx(0.8413, abs=1e-3)

    def test_example_from_spec(self):
        # margin +3.0, default sigma → ~0.613
        assert mod.margin_to_ml_prob(3.0) == pytest.approx(0.613, abs=0.005)


class TestMarginToSpreadProb:
    def test_margin_equals_spread(self):
        # predicted exactly = spread → 0.5
        assert mod.margin_to_spread_prob(5.0, 5.0) == pytest.approx(0.5)

    def test_home_covers(self):
        # predicted 5, spread -3 → (5 - (-3)) / 10.5 = 8/10.5
        prob = mod.margin_to_spread_prob(5.0, -3.0)
        assert prob > 0.5
        # Φ(8/10.5) ≈ Φ(0.762) ≈ 0.777
        assert prob == pytest.approx(0.777, abs=0.01)

    def test_spec_example(self):
        # predicted 5, spread -3 → Φ(2/10.5) note: spec says Φ((4.5-2.5)/10.5)
        # Adjusted: predicted 4.5, spread -2.5 → Φ((4.5-(-2.5))/10.5) = Φ(7/10.5)
        # Wait — spec says spread -2.5 means home favoured by 2.5,
        # so (4.5 - (-2.5))/10.5 = 7/10.5 ≈ 0.667 → Φ(0.667) ≈ 0.748
        # Actually re-reading spec: "predicted margin +4.5 and spread is -2.5"
        # → (4.5 - (-2.5))/10.5 = 7/10.5 → Φ(0.667) ≈ 0.748
        # But spec says ≈ 0.576 which matches Φ(2/10.5)=Φ(0.190)≈0.576
        # So they mean market_spread=2.5 (away favoured), margin=4.5
        prob = mod.margin_to_spread_prob(4.5, 2.5)
        assert prob == pytest.approx(0.576, abs=0.01)


class TestMarginToTotalProb:
    def test_predicted_equals_market(self):
        assert mod.margin_to_total_prob(140.0, 140.0) == pytest.approx(0.5)

    def test_predicted_over(self):
        prob = mod.margin_to_total_prob(150.0, 140.0)
        assert prob > 0.5

    def test_predicted_under(self):
        prob = mod.margin_to_total_prob(130.0, 140.0)
        assert prob < 0.5


# ---------------------------------------------------------------------------
# American odds conversion
# ---------------------------------------------------------------------------


class TestAmericanToImplied:
    def test_minus_110(self):
        # -110 → 110/210 ≈ 0.5238
        assert mod._american_to_implied(-110) == pytest.approx(0.5238, abs=1e-3)

    def test_plus_150(self):
        # +150 → 100/250 = 0.40
        assert mod._american_to_implied(150) == pytest.approx(0.40)

    def test_even(self):
        # +100 → 100/200 = 0.50
        assert mod._american_to_implied(100) == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# predict_games — integration with mocked model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Name resolution in predict module (regression tests for team matching bugs)
# ---------------------------------------------------------------------------


def _make_team(name: str, adj_em: float = 10.0) -> dict:
    """Create a minimal ratings row for testing name resolution."""
    return {
        "team": name,
        "adj_em": adj_em, "adj_oe": 100 + adj_em / 2,
        "adj_de": 100 - adj_em / 2, "adj_t": 68.0,
        "efg_o": 50.0, "efg_d": 50.0, "to_o": 18.0, "to_d": 18.0,
        "or_pct": 30.0, "ftr_o": 30.0, "ftr_d": 30.0, "sos": 5.0,
        "rank": 50, "conf": "Conf", "record": "15-10", "barthag": 0.7,
        "wins": 15, "losses": 10,
    }


# Torvik names that must coexist without ambiguity
_AMBIGUOUS_TORVIK_NAMES = [
    "Ohio", "Ohio St.", "Tennessee", "Tennessee St.",
    "Michigan", "Michigan St.", "Iowa", "Iowa St.",
    "Utah", "Utah St.", "North Dakota", "North Dakota St.",
    "TCU", "Duke", "Villanova", "N.C. State",
    "McNeese St.", "Kennesaw St.", "LIU", "Wright St.",
]


class TestResolveTeamAmbiguous:
    """Ensure _resolve_team picks the correct team when ambiguous pairs exist."""

    @pytest.fixture(autouse=True)
    def _index(self):
        self.index = mod._build_name_index(_AMBIGUOUS_TORVIK_NAMES)

    @pytest.mark.parametrize(
        "espn_name, expected_torvik",
        [
            # MISMATCHED cases from bug report
            ("Ohio State Buckeyes", "Ohio St."),
            ("TCU Horned Frogs", "TCU"),
            ("Tennessee St Tigers", "Tennessee St."),
            ("Tennessee State Tigers", "Tennessee St."),
            ("Iowa State Cyclones", "Iowa St."),
            ("North Dakota St Bison", "North Dakota St."),
            ("North Dakota State Bison", "North Dakota St."),
            ("Michigan St Spartans", "Michigan St."),
            ("Michigan State Spartans", "Michigan St."),
            ("Utah State Aggies", "Utah St."),
            # Must NOT pick the "State" variant
            ("Ohio Bobcats", "Ohio"),
            ("Tennessee Volunteers", "Tennessee"),
            ("Michigan Wolverines", "Michigan"),
            ("Iowa Hawkeyes", "Iowa"),
            ("Utah Utes", "Utah"),
            ("North Dakota Fighting Hawks", "North Dakota"),
            # UNRESOLVED cases from bug report
            ("NC State Wolfpack", "N.C. State"),
            ("McNeese Cowboys", "McNeese St."),
            ("Kennesaw State Owls", "Kennesaw St."),
            ("Long Island University Sharks", "LIU"),
            ("Wright State Raiders", "Wright St."),
        ],
    )
    def test_resolves_correctly(self, espn_name, expected_torvik):
        result = mod._resolve_team(espn_name, self.index)
        assert result == expected_torvik, (
            f"Expected {espn_name!r} → {expected_torvik!r}, got {result!r}"
        )


class TestResolveTeamPredictions:
    """End-to-end: predict_games should pull the RIGHT team's ratings."""

    @pytest.fixture()
    def _mock_ambiguous(self):
        """Mock with ambiguous team pairs so we can verify correct resolution."""
        teams = [
            _make_team("Ohio St.", adj_em=20.0),
            _make_team("Ohio", adj_em=-5.0),
            _make_team("TCU", adj_em=5.0),
            _make_team("Tennessee St.", adj_em=-5.0),
            _make_team("Tennessee", adj_em=25.0),
            _make_team("Iowa St.", adj_em=31.0),
            _make_team("Iowa", adj_em=10.0),
            _make_team("North Dakota St.", adj_em=5.0),
            _make_team("North Dakota", adj_em=-10.0),
            _make_team("Michigan St.", adj_em=25.0),
            _make_team("Michigan", adj_em=15.0),
            _make_team("Utah St.", adj_em=15.0),
            _make_team("Utah", adj_em=5.0),
            _make_team("Villanova", adj_em=8.0),
        ]
        ratings_df = pd.DataFrame(teams)
        metadata = {"model_path": "test_model_v1"}

        # Model that returns adj_em_diff + home_court as prediction
        class _EmDiffModel:
            def predict(self, X):  # noqa: N803
                arr = np.asarray(X)
                return arr[:, 0] + arr[:, 10]  # adj_em_diff + home_court

        with (
            patch.object(
                mod.train, "load_model",
                return_value=(_EmDiffModel(), _FakeScaler(), metadata),
            ),
            patch.object(
                mod.data, "fetch_team_ratings",
                return_value=ratings_df,
            ),
        ):
            yield

    def test_ohio_state_not_ohio(self, _mock_ambiguous):
        """Ohio State Buckeyes must use Ohio St. ratings, not Ohio."""
        games = [{"home_team": "Ohio State Buckeyes", "away_team": "TCU Horned Frogs"}]
        preds = mod.predict_games(games)
        assert len(preds) == 1
        assert preds[0]["home_team"] == "Ohio St."
        assert preds[0]["away_team"] == "TCU"
        # Ohio St. AdjEM=20, TCU AdjEM=5 → diff=15, +HCA 3.5 → ~18.5
        assert preds[0]["predicted_margin"] > 10

    def test_tennessee_st_not_tennessee(self, _mock_ambiguous):
        """Tennessee St Tigers must use Tennessee St. ratings, not Tennessee."""
        games = [
            {"home_team": "Iowa State Cyclones", "away_team": "Tennessee St Tigers"},
        ]
        preds = mod.predict_games(games)
        assert len(preds) == 1
        assert preds[0]["home_team"] == "Iowa St."
        assert preds[0]["away_team"] == "Tennessee St."
        # Iowa St. AdjEM=31, Tenn St. AdjEM=-5 → diff=36, +HCA → ~39.5
        assert preds[0]["predicted_margin"] > 25

    def test_north_dakota_st_not_north_dakota(self, _mock_ambiguous):
        """North Dakota St Bison must use North Dakota St. ratings."""
        games = [
            {"home_team": "Michigan St Spartans", "away_team": "North Dakota St Bison"},
        ]
        preds = mod.predict_games(games)
        assert len(preds) == 1
        assert preds[0]["home_team"] == "Michigan St."
        assert preds[0]["away_team"] == "North Dakota St."
        # Mich St. AdjEM=25, ND St. AdjEM=5 → diff=20, +HCA → ~23.5
        assert preds[0]["predicted_margin"] > 15

    def test_utah_state_not_utah(self, _mock_ambiguous):
        """Utah State Aggies must use Utah St. ratings, not Utah."""
        games = [
            {"home_team": "Villanova Wildcats", "away_team": "Utah State Aggies"},
        ]
        preds = mod.predict_games(games)
        assert len(preds) == 1
        assert preds[0]["away_team"] == "Utah St."
        # Villanova AdjEM=8, Utah St. AdjEM=15 → diff=-7, +HCA 3.5 → -3.5
        assert preds[0]["predicted_margin"] < 0  # away favored


# ---------------------------------------------------------------------------
# Existing test fixtures and data
# ---------------------------------------------------------------------------

_DUKE = {
    "team": "Duke",
    "adj_em": 30.0, "adj_oe": 120.0, "adj_de": 90.0, "adj_t": 70.0,
    "efg_o": 55.0, "efg_d": 45.0, "to_o": 15.0, "to_d": 20.0,
    "or_pct": 33.0, "ftr_o": 38.0, "ftr_d": 28.0, "sos": 10.0,
    "rank": 1, "conf": "ACC", "record": "30-2", "barthag": 0.98,
    "wins": 30, "losses": 2,
}

_UNC = {
    "team": "North Carolina",
    "adj_em": 22.0, "adj_oe": 115.0, "adj_de": 93.0, "adj_t": 72.0,
    "efg_o": 53.0, "efg_d": 47.0, "to_o": 17.0, "to_d": 18.0,
    "or_pct": 31.0, "ftr_o": 35.0, "ftr_d": 30.0, "sos": 8.0,
    "rank": 8, "conf": "ACC", "record": "25-7", "barthag": 0.92,
    "wins": 25, "losses": 7,
}


class _FakeModel:
    """Trivial model that returns adj_em_diff as the prediction."""

    def predict(self, X):  # noqa: N803
        return np.array([5.0] * len(X))


class _FakeScaler:
    def transform(self, X):  # noqa: N803
        return X


@pytest.fixture()
def _mock_model():
    """Patch load_model and fetch_team_ratings."""
    ratings_df = pd.DataFrame([_DUKE, _UNC])
    metadata = {"model_path": "test_model_v1"}

    with (
        patch.object(
            mod.train, "load_model",
            return_value=(_FakeModel(), _FakeScaler(), metadata),
        ),
        patch.object(
            mod.data, "fetch_team_ratings",
            return_value=ratings_df,
        ),
    ):
        yield


class TestPredictGames:
    def test_basic_prediction(self, _mock_model):
        games = [
            {"home_team": "Duke", "away_team": "North Carolina"},
        ]
        preds = mod.predict_games(games)
        assert len(preds) == 1
        p = preds[0]
        assert p["home_team"] == "Duke"
        assert p["away_team"] == "North Carolina"
        assert p["predicted_margin"] == 5.0
        assert p["home_ml_prob"] > 0.5
        assert p["away_ml_prob"] < 0.5
        assert p["home_ml_prob"] + p["away_ml_prob"] == pytest.approx(1.0)
        assert "model_version" in p
        assert "prediction_timestamp" in p

    def test_missing_team_skipped(self, _mock_model):
        games = [
            {"home_team": "Duke", "away_team": "Fake University"},
        ]
        preds = mod.predict_games(games)
        assert len(preds) == 0

    def test_multiple_games(self, _mock_model):
        games = [
            {"home_team": "Duke", "away_team": "North Carolina"},
            {"home_team": "North Carolina", "away_team": "Duke"},
        ]
        preds = mod.predict_games(games)
        assert len(preds) == 2


class TestPredictWithMarket:
    def test_adds_spread_and_total(self, _mock_model):
        games = [{"home_team": "Duke", "away_team": "North Carolina"}]
        lines = {
            "Duke vs North Carolina": {
                "spread": -3.5,
                "total": 145.0,
                "home_ml_odds": -200,
            },
        }
        preds = mod.predict_with_market(games, lines)
        assert len(preds) == 1
        p = preds[0]
        assert p["market_spread"] == -3.5
        assert p["home_spread_prob"] is not None
        assert p["home_spread_prob"] > 0.5
        assert p["market_total"] == 145.0
        assert p["model_edge_ml"] is not None

    def test_no_market_data(self, _mock_model):
        games = [{"home_team": "Duke", "away_team": "North Carolina"}]
        preds = mod.predict_with_market(games, {})
        assert len(preds) == 1
        p = preds[0]
        assert p["market_spread"] is None
        assert p["home_spread_prob"] is None
        assert p["model_edge_ml"] is None
