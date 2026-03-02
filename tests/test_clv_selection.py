"""Tests for Step 1 – CLV-driven selection engine."""

from __future__ import annotations

from line_tracker.services.clv_selection_service import (
    build_clv_filter_profile,
    passes_clv_filter,
)
from line_tracker.storage import LineStore


class TestBuildClvFilterProfile:
    """Unit tests for building CLV filter profiles."""

    def test_empty_store(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            profile = build_clv_filter_profile(store)
        assert profile["total_closed"] == 0
        assert profile["groups"] == {}

    def test_profile_from_closed_snapshots(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Insert and close several snapshots
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
                    "consensus_prob": 0.55,
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
                # 7 out of 10 beat the close
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

            profile = build_clv_filter_profile(store)
            assert profile["total_closed"] == 10
            # The group (spread, tier1b, Strong) should have 10 samples, 70% positive
            key = ("spread", "tier1b", "Strong")
            assert key in profile["groups"]
            group = profile["groups"][key]
            assert group["count"] == 10
            assert group["pct_positive"] == 70.0
            assert group["passes"] is True

    def test_profile_group_fails_below_threshold(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            for i in range(10):
                snap = {
                    "created_at": f"2026-01-{15 + i}T12:00:00",
                    "event_id": f"e{i}",
                    "sport": "nfl",
                    "market": "moneyline",
                    "selection": "Bills",
                    "line": None,
                    "book": "DraftKings",
                    "odds_american": 150.0,
                    "odds_decimal": 2.50,
                    "consensus_prob": 0.40,
                    "breakeven_prob": 0.40,
                    "edge_pct": 1.0,
                    "edge_ev": 0.01,
                    "edge_ev_shrunk": 0.008,
                    "ev_100": 1.0,
                    "edge_z": 0.8,
                    "quality_score": 50,
                    "confidence_label": "Low",
                    "tier": "tier3",
                    "meta": None,
                    "alpha_score": 30,
                    "alpha_label": "Weak",
                }
                store.log_rec_snapshots([snap])
                unclosed = store.get_unclosed_snapshots()
                sid = unclosed[-1]["snapshot_id"]
                # Only 3 out of 10 beat the close
                clv_delta = 0.01 if i < 3 else -0.02
                store.close_snapshot(
                    sid,
                    close_odds_american=140.0,
                    close_odds_decimal=2.40,
                    close_implied_prob=0.4167,
                    open_implied_prob=0.40,
                    clv_delta_american=-10.0,
                    clv_delta_implied=clv_delta,
                )

            profile = build_clv_filter_profile(store)
            key = ("moneyline", "tier3", "Weak")
            assert key in profile["groups"]
            assert profile["groups"][key]["passes"] is False


class TestPassesClvFilter:
    def test_empty_profile_passes(self):
        assert passes_clv_filter({"market": "x", "tier": "y"}, {"groups": {}}) is True

    def test_known_passing_group(self):
        profile = {
            "groups": {
                ("spread", "tier1b", "Strong"): {
                    "pct_positive": 70.0,
                    "count": 20,
                    "passes": True,
                },
            },
        }
        entry = {"market": "spread", "tier": "tier1b", "alpha_label": "Strong"}
        assert passes_clv_filter(entry, profile) is True

    def test_unknown_group_fails(self):
        profile = {
            "groups": {
                ("spread", "tier1b", "Strong"): {"passes": True},
            },
        }
        entry = {"market": "total", "tier": "tier2", "alpha_label": "Neutral"}
        assert passes_clv_filter(entry, profile) is False

    def test_failing_group(self):
        profile = {
            "groups": {
                ("moneyline", "tier3", "Weak"): {
                    "pct_positive": 30.0,
                    "count": 10,
                    "passes": False,
                },
            },
        }
        entry = {"market": "moneyline", "tier": "tier3", "alpha_label": "Weak"}
        assert passes_clv_filter(entry, profile) is False
