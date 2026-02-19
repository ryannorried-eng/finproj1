"""Tests for alpha_clv_report analytics (read-only, no recommendation logic)."""

from __future__ import annotations

import pytest

from line_tracker.performance import alpha_clv_report
from line_tracker.storage import LineStore

# -------------------------------------------------------------------
# Synthetic fixture data
# -------------------------------------------------------------------

def _make_alpha_stats_rows():
    """Simulate rows returned by RecSnapshotsRepo.get_alpha_clv_stats()."""
    return [
        {
            "alpha_label": "Neutral",
            "cnt": 20,
            "avg_clv_implied": 0.005,
            "avg_clv_american": -3.0,
            "avg_edge_z": 1.8,
            "avg_edge_ev_shrunk": 0.018,
            "beat_close_cnt": 11,
        },
        {
            "alpha_label": "Strong",
            "cnt": 15,
            "avg_clv_implied": 0.025,
            "avg_clv_american": -8.5,
            "avg_edge_z": 3.2,
            "avg_edge_ev_shrunk": 0.035,
            "beat_close_cnt": 12,
        },
        {
            "alpha_label": "Weak",
            "cnt": 10,
            "avg_clv_implied": -0.010,
            "avg_clv_american": 5.0,
            "avg_edge_z": 0.9,
            "avg_edge_ev_shrunk": 0.008,
            "beat_close_cnt": 3,
        },
    ]


# -------------------------------------------------------------------
# 1. Empty dataset returns safe structure
# -------------------------------------------------------------------


class TestAlphaClvReportEmpty:
    def test_empty_list(self):
        result = alpha_clv_report([])
        assert result["summary"]["total_closed"] == 0
        assert result["summary"]["alpha_clv_spread"] is None
        assert result["summary"]["strong_avg_clv"] is None
        assert result["summary"]["neutral_avg_clv"] is None
        assert result["summary"]["weak_avg_clv"] is None
        assert result["rows"] == []

    def test_none_safe(self):
        """Passing an empty list doesn't crash."""
        result = alpha_clv_report([])
        assert isinstance(result, dict)
        assert "summary" in result
        assert "rows" in result


# -------------------------------------------------------------------
# 2. Strong vs Weak spread calculation
# -------------------------------------------------------------------


class TestAlphaClvSpread:
    def test_spread_is_strong_minus_weak(self):
        rows = _make_alpha_stats_rows()
        result = alpha_clv_report(rows)
        summary = result["summary"]
        # strong_avg_clv = 0.025, weak_avg_clv = -0.010
        expected_spread = 0.025 - (-0.010)
        assert summary["alpha_clv_spread"] == pytest.approx(
            expected_spread, abs=1e-6,
        )
        assert summary["strong_avg_clv"] == pytest.approx(0.025, abs=1e-6)
        assert summary["weak_avg_clv"] == pytest.approx(-0.010, abs=1e-6)

    def test_spread_none_when_missing_weak(self):
        rows = [r for r in _make_alpha_stats_rows() if r["alpha_label"] != "Weak"]
        result = alpha_clv_report(rows)
        assert result["summary"]["alpha_clv_spread"] is None

    def test_spread_none_when_missing_strong(self):
        rows = [r for r in _make_alpha_stats_rows() if r["alpha_label"] != "Strong"]
        result = alpha_clv_report(rows)
        assert result["summary"]["alpha_clv_spread"] is None


# -------------------------------------------------------------------
# 3. Beat close percentage computed correctly
# -------------------------------------------------------------------


class TestBeatClosePct:
    def test_beat_close_pct_values(self):
        rows = _make_alpha_stats_rows()
        result = alpha_clv_report(rows)
        by_label = {r["alpha_label"]: r for r in result["rows"]}

        # Strong: 12/15 = 80.0%
        assert by_label["Strong"]["beat_close_pct"] == pytest.approx(80.0, abs=0.1)
        # Neutral: 11/20 = 55.0%
        assert by_label["Neutral"]["beat_close_pct"] == pytest.approx(55.0, abs=0.1)
        # Weak: 3/10 = 30.0%
        assert by_label["Weak"]["beat_close_pct"] == pytest.approx(30.0, abs=0.1)


# -------------------------------------------------------------------
# 4. Rows grouped correctly by alpha label
# -------------------------------------------------------------------


class TestRowGrouping:
    def test_rows_ordered_strong_neutral_weak(self):
        rows = _make_alpha_stats_rows()
        result = alpha_clv_report(rows)
        labels = [r["alpha_label"] for r in result["rows"]]
        assert labels == ["Strong", "Neutral", "Weak"]

    def test_total_closed_sums_all_labels(self):
        rows = _make_alpha_stats_rows()
        result = alpha_clv_report(rows)
        # 15 + 20 + 10 = 45
        assert result["summary"]["total_closed"] == 45

    def test_row_structure(self):
        rows = _make_alpha_stats_rows()
        result = alpha_clv_report(rows)
        for row in result["rows"]:
            assert "alpha_label" in row
            assert "count" in row
            assert "beat_close_pct" in row
            assert "avg_clv_prob" in row
            assert "avg_clv_american" in row
            assert "avg_edge_z" in row
            assert "avg_edge_ev_shrunk" in row

    def test_avg_clv_values(self):
        rows = _make_alpha_stats_rows()
        result = alpha_clv_report(rows)
        by_label = {r["alpha_label"]: r for r in result["rows"]}
        assert by_label["Strong"]["avg_clv_american"] == pytest.approx(-8.5, abs=0.1)
        assert by_label["Neutral"]["avg_clv_american"] == pytest.approx(-3.0, abs=0.1)
        assert by_label["Weak"]["avg_clv_american"] == pytest.approx(5.0, abs=0.1)


# -------------------------------------------------------------------
# 5. Handles missing labels safely
# -------------------------------------------------------------------


class TestMissingLabels:
    def test_ignores_unknown_labels(self):
        rows = [
            {
                "alpha_label": "Unknown",
                "cnt": 5,
                "avg_clv_implied": 0.0,
                "avg_clv_american": 0.0,
                "avg_edge_z": 1.0,
                "avg_edge_ev_shrunk": 0.01,
                "beat_close_cnt": 2,
            },
        ]
        result = alpha_clv_report(rows)
        assert result["rows"] == []
        assert result["summary"]["total_closed"] == 0

    def test_handles_none_edge_values(self):
        rows = [
            {
                "alpha_label": "Strong",
                "cnt": 5,
                "avg_clv_implied": 0.01,
                "avg_clv_american": -2.0,
                "avg_edge_z": None,
                "avg_edge_ev_shrunk": None,
                "beat_close_cnt": 3,
            },
        ]
        result = alpha_clv_report(rows)
        assert len(result["rows"]) == 1
        assert result["rows"][0]["avg_edge_z"] is None
        assert result["rows"][0]["avg_edge_ev_shrunk"] is None

    def test_single_label_only(self):
        rows = [
            {
                "alpha_label": "Strong",
                "cnt": 10,
                "avg_clv_implied": 0.03,
                "avg_clv_american": -10.0,
                "avg_edge_z": 2.5,
                "avg_edge_ev_shrunk": 0.04,
                "beat_close_cnt": 8,
            },
        ]
        result = alpha_clv_report(rows)
        assert len(result["rows"]) == 1
        assert result["summary"]["total_closed"] == 10
        assert result["summary"]["strong_avg_clv"] == pytest.approx(0.03, abs=1e-6)
        assert result["summary"]["weak_avg_clv"] is None
        # spread is None because Weak is missing
        assert result["summary"]["alpha_clv_spread"] is None


# -------------------------------------------------------------------
# 6. Database integration (round-trip with LineStore)
# -------------------------------------------------------------------


class TestAlphaClvDbRoundtrip:
    def _insert_and_close(self, store, alpha_label, open_odds, close_impl_delta):
        """Insert a snapshot with alpha_label, then close it."""
        from line_tracker.core.math import implied_probability

        open_impl = implied_probability(open_odds)
        close_impl = open_impl + close_impl_delta
        snap = {
            "created_at": "2026-02-01T12:00:00",
            "event_id": f"evt_{alpha_label}_{open_odds}",
            "sport": "nfl",
            "market": "moneyline",
            "selection": "Team",
            "line": None,
            "book": "FanDuel",
            "odds_american": open_odds,
            "odds_decimal": 1.0 / open_impl if open_impl else 2.0,
            "consensus_prob": 0.55,
            "breakeven_prob": 0.50,
            "edge_pct": 3.0,
            "edge_ev": 0.03,
            "edge_ev_shrunk": 0.025,
            "ev_100": 3.0,
            "edge_z": 2.0,
            "quality_score": 80,
            "confidence_label": "High",
            "tier": "tier1b",
            "meta": None,
            "alpha_score": 75 if alpha_label == "Strong" else 50,
            "alpha_label": alpha_label,
        }
        store.log_rec_snapshots([snap])
        unclosed = store.get_unclosed_snapshots()
        for s in unclosed:
            if s["event_id"] == snap["event_id"]:
                store.close_snapshot(
                    s["snapshot_id"],
                    close_odds_american=open_odds - 10,
                    close_odds_decimal=snap["odds_decimal"],
                    close_implied_prob=close_impl,
                    open_implied_prob=open_impl,
                    clv_delta_american=-10.0,
                    clv_delta_implied=close_impl_delta,
                )
                break

    def test_roundtrip_alpha_clv(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Insert Strong entries (positive CLV)
            self._insert_and_close(store, "Strong", -150.0, 0.03)
            self._insert_and_close(store, "Strong", -140.0, 0.02)
            # Insert Weak entries (negative CLV)
            self._insert_and_close(store, "Weak", -110.0, -0.01)

            stats = store.get_alpha_clv_stats()
            assert len(stats) == 2  # Strong and Weak

            report = alpha_clv_report(stats)
            assert report["summary"]["total_closed"] == 3
            assert report["summary"]["alpha_clv_spread"] is not None
            assert report["summary"]["alpha_clv_spread"] > 0  # Strong > Weak

            by_label = {r["alpha_label"]: r for r in report["rows"]}
            assert by_label["Strong"]["count"] == 2
            assert by_label["Weak"]["count"] == 1
            assert by_label["Strong"]["avg_clv_prob"] > 0
            assert by_label["Weak"]["avg_clv_prob"] < 0
