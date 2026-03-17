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
