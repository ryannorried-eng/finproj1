"""Tests for Step 7 – CLV model service."""

from __future__ import annotations

import pytest

from line_tracker.services.clv_model_service import (
    _classify_selection,
    _sigmoid,
    _z_score_safe,
    extract_features,
    predict_clv_positive,
    train_clv_model,
)
from line_tracker.storage import LineStore


class TestClassifySelection:
    def test_total_over(self):
        assert _classify_selection("total", "Over 45.5", 0.50) == "over"

    def test_total_under(self):
        assert _classify_selection("total", "Under 210", 0.50) == "under"

    def test_moneyline_favorite(self):
        assert _classify_selection("moneyline", "Chiefs", 0.60) == "ml_favorite"

    def test_moneyline_underdog(self):
        assert _classify_selection("moneyline", "Jets", 0.35) == "ml_underdog"

    def test_moneyline_tossup(self):
        assert _classify_selection("moneyline", "Eagles", 0.50) == "ml_tossup"

    def test_spread_favorite(self):
        assert _classify_selection("spread", "Chiefs -3", 0.58) == "spread_favorite"

    def test_spread_dog(self):
        assert _classify_selection("spread", "Jets +3", 0.45) == "spread_dog"

    def test_unknown_market(self):
        assert _classify_selection("props", "Player X", 0.50) == "other"


class TestHelpers:
    def test_sigmoid_zero(self):
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_sigmoid_positive(self):
        assert _sigmoid(5) > 0.99

    def test_sigmoid_negative(self):
        assert _sigmoid(-5) < 0.01

    def test_z_score_safe_normal(self):
        assert _z_score_safe(80, 70, 10) == pytest.approx(1.0)

    def test_z_score_safe_zero_scale(self):
        assert _z_score_safe(80, 70, 0) == 0.0


class TestExtractFeatures:
    def test_basic_extraction(self):
        snap = {
            "market": "spread",
            "selection": "Chiefs",
            "consensus_prob": 0.55,
            "edge_z": 2.0,
            "edge_ev_shrunk": 0.03,
            "quality_score": 80,
            "alpha_score": 75,
        }
        feat = extract_features(snap)
        assert feat["market"] == "spread"
        assert feat["selection_type"] == "spread_favorite"
        assert feat["edge_z"] == 2.0
        assert feat["quality_score"] == 80

    def test_missing_fields_default(self):
        snap = {"market": "unknown", "selection": "X"}
        feat = extract_features(snap)
        assert feat["edge_z"] == 0.0
        assert feat["quality_score"] == 0


class TestTrainClvModel:
    def test_no_data(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            result = train_clv_model(store)
        assert result["status"] == "no_data"

    def test_trains_with_closed_snapshots(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Insert and close 10 spread snapshots
            for i in range(10):
                snap = {
                    "created_at": f"2026-01-{15 + i}T12:00:00",
                    "event_id": f"e{i}",
                    "sport": "nfl",
                    "market": "spread",
                    "selection": "Chiefs",
                    "line": -3.5,
                    "book": "FanDuel",
                    "odds_american": -110.0,
                    "odds_decimal": 1.9091,
                    "consensus_prob": 0.58,
                    "breakeven_prob": 0.52,
                    "edge_pct": 3.0,
                    "edge_ev": 0.03,
                    "edge_ev_shrunk": 0.025,
                    "ev_100": 3.0,
                    "edge_z": 2.0,
                    "quality_score": 80,
                    "confidence_label": "High",
                    "tier": "tier1b",
                    "meta": None,
                    "alpha_score": 75,
                    "alpha_label": "Strong",
                }
                store.log_rec_snapshots([snap])
                unclosed = store.get_unclosed_snapshots()
                sid = unclosed[-1]["snapshot_id"]
                clv_delta = 0.02 if i < 7 else -0.01
                store.close_snapshot(
                    sid,
                    close_odds_american=-120.0,
                    close_odds_decimal=1.8333,
                    close_implied_prob=0.5455,
                    open_implied_prob=0.5238,
                    clv_delta_american=-10.0,
                    clv_delta_implied=clv_delta,
                )

            result = train_clv_model(store)
            assert result["status"] == "trained"
            assert result["groups"] >= 1
            assert result["total_samples"] == 10

            # Verify scores were persisted
            scores = store.get_all_clv_model_scores()
            assert len(scores) >= 1


class TestPredictClvPositive:
    def test_no_model_returns_prior(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            entry = {"market": "spread", "selection": "Chiefs", "consensus_prob": 0.55}
            prob = predict_clv_positive(entry, store)
        assert prob == 0.5

    def test_with_trained_model(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            for i in range(10):
                snap = {
                    "created_at": f"2026-01-{15 + i}T12:00:00",
                    "event_id": f"e{i}",
                    "sport": "nfl",
                    "market": "spread",
                    "selection": "Chiefs",
                    "line": -3.5,
                    "book": "FanDuel",
                    "odds_american": -110.0,
                    "odds_decimal": 1.9091,
                    "consensus_prob": 0.58,
                    "breakeven_prob": 0.52,
                    "edge_pct": 3.0,
                    "edge_ev": 0.03,
                    "edge_ev_shrunk": 0.025,
                    "ev_100": 3.0,
                    "edge_z": 2.0,
                    "quality_score": 80,
                    "confidence_label": "High",
                    "tier": "tier1b",
                    "meta": None,
                    "alpha_score": 75,
                    "alpha_label": "Strong",
                }
                store.log_rec_snapshots([snap])
                unclosed = store.get_unclosed_snapshots()
                sid = unclosed[-1]["snapshot_id"]
                store.close_snapshot(
                    sid,
                    close_odds_american=-120.0,
                    close_odds_decimal=1.8333,
                    close_implied_prob=0.5455,
                    open_implied_prob=0.5238,
                    clv_delta_american=-10.0,
                    clv_delta_implied=0.02,
                )

            train_clv_model(store)

            entry = {
                "market": "spread",
                "selection": "Chiefs",
                "consensus_prob": 0.58,
            }
            prob = predict_clv_positive(entry, store)
            assert 0.0 < prob <= 1.0
            # With 100% positive CLV training data, prediction should be high
            assert prob > 0.5
