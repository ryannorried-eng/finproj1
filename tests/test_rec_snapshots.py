"""Tests for CLV logging pipeline: implied probability, snapshot logging,
CLV computation, and rec_snapshot_clv_summary."""

from __future__ import annotations

import pytest

from line_tracker.core.math import implied_probability
from line_tracker.performance import rec_snapshot_clv_summary
from line_tracker.services.rec_snapshot_service import (
    build_snapshot_rows,
    compute_clv_for_snapshot,
)
from line_tracker.storage import LineStore


# -------------------------------------------------------------------
# 1. Implied probability conversion
# -------------------------------------------------------------------


class TestImpliedProbability:
    """Unit tests for implied_probability (American → [0,1])."""

    def test_negative_odds(self):
        # -110 → 110 / (110 + 100) = 110/210 ≈ 0.5238
        p = implied_probability(-110)
        assert p == pytest.approx(0.5238, abs=0.001)

    def test_large_negative_odds(self):
        # -200 → 200 / (200 + 100) = 200/300 ≈ 0.6667
        p = implied_probability(-200)
        assert p == pytest.approx(0.6667, abs=0.001)

    def test_positive_odds(self):
        # +150 → 100 / (150 + 100) = 100/250 = 0.40
        p = implied_probability(150)
        assert p == pytest.approx(0.40, abs=0.001)

    def test_large_positive_odds(self):
        # +300 → 100 / (300 + 100) = 100/400 = 0.25
        p = implied_probability(300)
        assert p == pytest.approx(0.25, abs=0.001)

    def test_even_money(self):
        # +100 → 100 / (100 + 100) = 0.50
        p = implied_probability(100)
        assert p == pytest.approx(0.50, abs=0.001)

    def test_zero_odds(self):
        # edge case: odds=0 → 0.5
        p = implied_probability(0)
        assert p == 0.5

    def test_minus_100(self):
        # -100 → 100 / (100 + 100) = 0.50
        p = implied_probability(-100)
        assert p == pytest.approx(0.50, abs=0.001)


# -------------------------------------------------------------------
# 2. Snapshot logging writes expected columns
# -------------------------------------------------------------------


class TestSnapshotLogging:
    """Test that build_snapshot_rows produces correct structure and
    that snapshot insertion writes expected columns to DB."""

    def _dummy_slate(self):
        """Minimal slate dict with one tier1b entry."""
        return {
            "tier1a": [],
            "tier1b": [{
                "event_id": "evt_001",
                "market": "moneyline",
                "selection": "Chiefs",
                "line": None,
                "best_sportsbook": "FanDuel",
                "best_odds": -150.0,
                "consensus_prob": 0.60,
                "p_be": 0.55,
                "edge_pct": 3.5,
                "edge_ev": 0.035,
                "edge_ev_shrunk": 0.028,
                "ev_100": 3.5,
                "edge_z": 2.1,
                "quality_score": 85,
                "confidence": "High",
                "tier": "tier1b",
            }],
            "tier2": [],
            "tier3": [],
            "closest_candidates": [],
            "stay_away": [],
        }

    def test_build_snapshot_rows_structure(self):
        rows = build_snapshot_rows(self._dummy_slate(), sport="nfl")
        assert len(rows) == 1
        row = rows[0]
        # Check all required columns are present
        for key in (
            "created_at", "event_id", "sport", "market", "selection",
            "book", "odds_american", "odds_decimal", "consensus_prob",
            "edge_pct", "edge_ev_shrunk", "ev_100", "edge_z",
            "quality_score", "confidence_label", "tier",
        ):
            assert key in row, f"Missing key: {key}"
        assert row["event_id"] == "evt_001"
        assert row["book"] == "FanDuel"
        assert row["odds_american"] == -150.0
        assert row["tier"] == "tier1b"
        assert row["confidence_label"] == "High"

    def test_build_snapshot_rows_deduplicates(self):
        slate = self._dummy_slate()
        # Add same entry to closest_candidates (overlap)
        slate["closest_candidates"] = list(slate["tier1b"])
        rows = build_snapshot_rows(slate, sport="nfl")
        assert len(rows) == 1  # deduped

    def test_stay_away_not_logged(self):
        slate = self._dummy_slate()
        slate["stay_away"] = [{
            "event_id": "evt_avoid",
            "market": "moneyline",
            "selection": "Loser",
            "best_sportsbook": "DK",
            "best_odds": 200.0,
            "consensus_prob": 0.30,
            "tier": "avoid",
        }]
        rows = build_snapshot_rows(slate, sport="nfl")
        event_ids = {r["event_id"] for r in rows}
        assert "evt_avoid" not in event_ids

    def test_snapshot_db_roundtrip(self, tmp_path):
        """Snapshot rows inserted into DB have expected columns."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            rows = build_snapshot_rows(self._dummy_slate(), sport="nfl")
            count = store.log_rec_snapshots(rows)
            assert count == 1

            # Read back
            unclosed = store.get_unclosed_snapshots()
            assert len(unclosed) == 1
            snap = unclosed[0]
            assert snap["event_id"] == "evt_001"
            assert snap["market"] == "moneyline"
            assert snap["odds_american"] == -150.0
            assert snap["tier"] == "tier1b"
            assert snap["closed_at"] is None

    def test_snapshot_idempotent(self, tmp_path):
        """Inserting the same snapshot twice doesn't create duplicates."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            rows = build_snapshot_rows(self._dummy_slate(), sport="nfl")
            c1 = store.log_rec_snapshots(rows)
            c2 = store.log_rec_snapshots(rows)
            assert c1 == 1
            assert c2 == 0  # idempotent
            unclosed = store.get_unclosed_snapshots()
            assert len(unclosed) == 1


# -------------------------------------------------------------------
# 3. CLV computation (implied-prob CLV sign tests)
# -------------------------------------------------------------------


class TestComputeClvForSnapshot:
    """Test CLV sign convention for moneyline odds movements."""

    def test_positive_odds_shortening(self):
        """Odds move from +150 → +130 (market moves toward your pick).
        Implied: 0.40 → 0.4348 ⇒ clv_delta_implied > 0 (beat the close).
        """
        clv = compute_clv_for_snapshot(150.0, 130.0)
        assert clv["open_implied_prob"] == pytest.approx(0.40, abs=0.001)
        assert clv["close_implied_prob"] == pytest.approx(0.4348, abs=0.001)
        assert clv["clv_delta_implied"] > 0  # beat the close

    def test_negative_odds_shortening(self):
        """Odds move from -110 → -125 (market moves toward your pick).
        Implied: 0.5238 → 0.5556 ⇒ clv_delta_implied > 0.
        """
        clv = compute_clv_for_snapshot(-110.0, -125.0)
        assert clv["open_implied_prob"] == pytest.approx(0.5238, abs=0.001)
        assert clv["close_implied_prob"] == pytest.approx(0.5556, abs=0.001)
        assert clv["clv_delta_implied"] > 0

    def test_negative_clv_when_odds_lengthen(self):
        """Odds move from -150 → -120 (market moves away from pick).
        Implied: 0.60 → 0.5455 ⇒ clv_delta_implied < 0 (lost to close).
        """
        clv = compute_clv_for_snapshot(-150.0, -120.0)
        assert clv["open_implied_prob"] == pytest.approx(0.60, abs=0.001)
        assert clv["close_implied_prob"] == pytest.approx(0.5455, abs=0.001)
        assert clv["clv_delta_implied"] < 0

    def test_no_movement(self):
        """No odds change → clv_delta_implied == 0."""
        clv = compute_clv_for_snapshot(-110.0, -110.0)
        assert clv["clv_delta_implied"] == 0.0
        assert clv["clv_delta_american"] == 0.0

    def test_positive_to_negative_crossover(self):
        """Odds cross from +105 → -105 (big move toward pick).
        Implied: 0.4878 → 0.5122 ⇒ positive CLV.
        """
        clv = compute_clv_for_snapshot(105.0, -105.0)
        assert clv["clv_delta_implied"] > 0


# -------------------------------------------------------------------
# 4. CLV summary by tier (rec_snapshot_clv_summary)
# -------------------------------------------------------------------


class TestRecSnapshotClvSummary:
    def test_empty_input(self):
        result = rec_snapshot_clv_summary([])
        assert result == {}

    def test_basic_summary(self):
        rows = [
            {"tier": "tier1b", "cnt": 10, "avg_clv_implied": 0.012,
             "pct_positive": 70.0},
            {"tier": "tier3", "cnt": 25, "avg_clv_implied": -0.003,
             "pct_positive": 44.0},
        ]
        result = rec_snapshot_clv_summary(rows)
        assert "tier1b" in result
        assert "tier3" in result
        assert result["tier1b"]["count"] == 10
        assert result["tier1b"]["avg_clv_implied"] == pytest.approx(0.012)
        assert result["tier1b"]["pct_positive"] == 70.0
        assert result["tier3"]["count"] == 25
        assert result["tier3"]["pct_positive"] == 44.0

    def test_db_roundtrip_summary(self, tmp_path):
        """Close a snapshot in DB and verify get_clv_summary_by_tier."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Insert + close a snapshot
            snap = {
                "created_at": "2026-01-15T12:00:00",
                "event_id": "e1",
                "sport": "nfl",
                "market": "moneyline",
                "selection": "Chiefs",
                "line": None,
                "book": "FanDuel",
                "odds_american": -150.0,
                "odds_decimal": 1.6667,
                "consensus_prob": 0.60,
                "breakeven_prob": 0.55,
                "edge_pct": 3.5,
                "edge_ev": 0.035,
                "edge_ev_shrunk": 0.028,
                "ev_100": 3.5,
                "edge_z": 2.1,
                "quality_score": 85,
                "confidence_label": "High",
                "tier": "tier1b",
                "meta": None,
            }
            store.log_rec_snapshots([snap])
            unclosed = store.get_unclosed_snapshots()
            sid = unclosed[0]["snapshot_id"]

            store.close_snapshot(
                sid,
                close_odds_american=-170.0,
                close_odds_decimal=1.5882,
                close_implied_prob=0.6296,
                open_implied_prob=0.6000,
                clv_delta_american=-20.0,
                clv_delta_implied=0.0296,
            )

            summary = store.get_clv_summary_by_tier()
            assert len(summary) == 1
            assert summary[0]["tier"] == "tier1b"
            assert summary[0]["cnt"] == 1
            assert summary[0]["avg_clv_implied"] == pytest.approx(0.0296, abs=0.001)
