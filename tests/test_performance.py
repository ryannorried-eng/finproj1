"""Tests for CLV performance analytics and related schema/snapshot changes."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from line_tracker.bet_history import create_bet, snapshot_pick
from line_tracker.models import BettingLine, BetType
from line_tracker.performance import (
    all_breakdowns,
    apply_filters,
    build_clv_dataframe,
    clv_distribution,
    groupby_breakdown,
    rolling_clv_series,
    summary_kpis,
)
from line_tracker.storage import LineStore

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_NOW = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)


def _ml_line(
    sportsbook: str,
    home_odds: float,
    away_odds: float,
    event: str = "Bills @ Chiefs",
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event=event,
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=_NOW,
    )


def _leg(**overrides):
    base = {
        "sport": "NFL",
        "event_name": "Bills @ Chiefs",
        "sportsbook": "DraftKings",
        "market": "ML",
        "selection": "Home",
        "line": None,
        "odds": -150,
        "fetched_at": "2026-02-01T12:00:00",
    }
    base.update(overrides)
    return base


def _make_bet(**overrides):
    defaults = dict(
        stake=100.0,
        sportsbook="DraftKings",
        legs=[_leg()],
    )
    defaults.update(overrides)
    return create_bet(**defaults)


def _synthetic_clv_rows(n: int = 10) -> list[dict]:
    """Generate synthetic closed CLV rows for testing."""
    rows = []
    for i in range(n):
        prob_pick = 0.50 + i * 0.01
        prob_close = prob_pick + (0.02 if i % 2 == 0 else -0.01)
        dec_pick = 1.80 + i * 0.05
        dec_close = dec_pick + (prob_close - prob_pick) * 2
        rows.append({
            "bet_id": f"bet_{i}",
            "leg_index": 0,
            "event": f"Event {i}",
            "market": "ML" if i % 3 == 0 else ("Spread" if i % 3 == 1 else "Total"),
            "pick_side": "Home",
            "pick_line_value": None,
            "pick_odds_american": -150.0,
            "pick_odds_decimal": dec_pick,
            "consensus_prob_at_pick": prob_pick,
            "consensus_prob_close": prob_close,
            "best_odds_close_american": -170.0,
            "best_odds_close_decimal": dec_close,
            "closed_at": f"2026-01-{15 + i:02d}T12:00:00",
            "pick_sportsbook": "DraftKings" if i % 2 == 0 else "FanDuel",
            "sport": "NFL" if i < 5 else "NBA",
            "confidence_at_pick": "High" if i % 3 == 0 else "Medium",
            "quality_tier_at_pick": "Elite" if i < 3 else "Strong",
            "edge_pct_at_pick": 3.0 + i * 0.5,
            "edge_z_at_pick": 1.5 + i * 0.1,
            "books_used_at_pick": 4 + i % 3,
            "agreement_score_at_pick": 70.0 + i,
            "market_hold_median_at_pick": 4.5,
            "market_volatility_sigma_at_pick": 0.012,
        })
    return rows


# -------------------------------------------------------------------
# Schema migration tests
# -------------------------------------------------------------------


class TestSchemaMigration:
    def test_new_columns_exist(self, tmp_path):
        """bet_clv table has the new metadata columns after init."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            cols = {
                row[1]
                for row in store._conn.execute(
                    "PRAGMA table_info(bet_clv)"
                ).fetchall()
            }
            for expected in [
                "pick_sportsbook",
                "sport",
                "confidence_at_pick",
                "quality_tier_at_pick",
                "edge_pct_at_pick",
                "edge_z_at_pick",
                "books_used_at_pick",
                "agreement_score_at_pick",
            ]:
                assert expected in cols, f"Missing column: {expected}"

    def test_migration_is_idempotent(self, tmp_path):
        """Opening the store twice doesn't fail (migration re-runs safely)."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_clv_pick(
                bet_id="test1",
                leg_index=0,
                event="Test",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-110.0,
                pick_odds_decimal=1.909,
                consensus_prob_at_pick=0.52,
                pick_sportsbook="DK",
            )
        # Re-open — should not fail
        with LineStore(db) as store2:
            rows = store2.get_clv("test1")
            assert len(rows) == 1
            assert rows[0]["pick_sportsbook"] == "DK"

    def test_save_and_retrieve_metadata(self, tmp_path):
        """New metadata columns round-trip through save_clv_pick/get_clv."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_clv_pick(
                bet_id="m1",
                leg_index=0,
                event="Test",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-150.0,
                pick_odds_decimal=1.6667,
                consensus_prob_at_pick=0.58,
                pick_sportsbook="FanDuel",
                sport="NFL",
                confidence_at_pick="High",
                quality_tier_at_pick="Elite",
                edge_pct_at_pick=5.2,
                edge_z_at_pick=2.1,
                books_used_at_pick=6,
                agreement_score_at_pick=85.0,
            )
            rows = store.get_clv("m1")
            r = rows[0]
            assert r["pick_sportsbook"] == "FanDuel"
            assert r["sport"] == "NFL"
            assert r["confidence_at_pick"] == "High"
            assert r["quality_tier_at_pick"] == "Elite"
            assert r["edge_pct_at_pick"] == 5.2
            assert r["edge_z_at_pick"] == 2.1
            assert r["books_used_at_pick"] == 6
            assert r["agreement_score_at_pick"] == 85.0

    def test_get_all_clv(self, tmp_path):
        """get_all_clv returns only closed rows."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Row with close data
            store.save_clv_pick(
                bet_id="a",
                leg_index=0,
                event="E1",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-110.0,
                pick_odds_decimal=1.909,
                consensus_prob_at_pick=0.52,
            )
            store.close_clv(
                bet_id="a",
                leg_index=0,
                consensus_prob_close=0.55,
                best_odds_close_american=-130.0,
                best_odds_close_decimal=1.769,
            )
            # Row without close data
            store.save_clv_pick(
                bet_id="b",
                leg_index=0,
                event="E2",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-110.0,
                pick_odds_decimal=1.909,
                consensus_prob_at_pick=0.52,
            )
            all_rows = store.get_all_clv()
            assert len(all_rows) == 1
            assert all_rows[0]["bet_id"] == "a"


# -------------------------------------------------------------------
# Performance computation tests
# -------------------------------------------------------------------


class TestBuildClvDataframe:
    def test_empty_input(self):
        df = build_clv_dataframe([])
        assert df.empty

    def test_computes_clv_columns(self):
        rows = _synthetic_clv_rows(5)
        df = build_clv_dataframe(rows)
        assert "clv_decimal" in df.columns
        assert "clv_prob" in df.columns
        assert len(df) == 5

    def test_clv_decimal_formula(self):
        rows = [{
            "pick_odds_decimal": 2.0,
            "best_odds_close_decimal": 1.8,
            "consensus_prob_at_pick": 0.50,
            "consensus_prob_close": 0.55,
            "closed_at": "2026-01-20T12:00:00",
        }]
        df = build_clv_dataframe(rows)
        assert df.iloc[0]["clv_decimal"] == pytest.approx(-0.2, abs=0.001)
        assert df.iloc[0]["clv_prob"] == pytest.approx(0.05, abs=0.001)


class TestSummaryKPIs:
    def test_empty_df(self):
        kpis = summary_kpis(pd.DataFrame())
        assert kpis["total_legs"] == 0
        assert kpis["beating_close_pct"] == 0.0

    def test_all_beating(self):
        rows = _synthetic_clv_rows(4)
        # Make all clv_prob positive
        for r in rows:
            r["consensus_prob_close"] = r["consensus_prob_at_pick"] + 0.03
        df = build_clv_dataframe(rows)
        kpis = summary_kpis(df)
        assert kpis["total_legs"] == 4
        assert kpis["beating_close_pct"] == 100.0
        assert kpis["avg_clv_prob"] > 0

    def test_mixed_results(self):
        rows = _synthetic_clv_rows(10)
        df = build_clv_dataframe(rows)
        kpis = summary_kpis(df)
        assert kpis["total_legs"] == 10
        assert 0 <= kpis["beating_close_pct"] <= 100


class TestGroupbyBreakdown:
    def test_by_market(self):
        rows = _synthetic_clv_rows(9)
        df = build_clv_dataframe(rows)
        tbl = groupby_breakdown(df, "market")
        assert not tbl.empty
        assert "market" in tbl.columns
        assert "legs" in tbl.columns
        assert tbl["legs"].sum() == 9

    def test_by_confidence(self):
        rows = _synthetic_clv_rows(6)
        df = build_clv_dataframe(rows)
        tbl = groupby_breakdown(df, "confidence_at_pick")
        assert not tbl.empty

    def test_missing_column(self):
        rows = _synthetic_clv_rows(3)
        df = build_clv_dataframe(rows)
        tbl = groupby_breakdown(df, "nonexistent_column")
        assert tbl.empty

    def test_empty_df(self):
        tbl = groupby_breakdown(pd.DataFrame(), "market")
        assert tbl.empty


class TestAllBreakdowns:
    def test_returns_available_breakdowns(self):
        rows = _synthetic_clv_rows(10)
        df = build_clv_dataframe(rows)
        result = all_breakdowns(df)
        assert isinstance(result, dict)
        # Should include at least market and sport
        assert "market" in result
        assert "sport" in result


class TestRollingClvSeries:
    def test_produces_rolling(self):
        rows = _synthetic_clv_rows(10)
        df = build_clv_dataframe(rows)
        rolling = rolling_clv_series(df)
        assert not rolling.empty
        assert "rolling_clv_prob" in rolling.columns
        assert "cumulative_legs" in rolling.columns

    def test_empty_df(self):
        rolling = rolling_clv_series(pd.DataFrame())
        assert rolling.empty

    def test_cumulative_legs_increase(self):
        rows = _synthetic_clv_rows(10)
        df = build_clv_dataframe(rows)
        rolling = rolling_clv_series(df)
        cum = rolling["cumulative_legs"].tolist()
        assert cum == sorted(cum)
        assert cum[-1] == 10


class TestClvDistribution:
    def test_bins_created(self):
        rows = _synthetic_clv_rows(20)
        df = build_clv_dataframe(rows)
        dist = clv_distribution(df)
        assert not dist.empty
        assert "bin_label" in dist.columns
        assert "count" in dist.columns
        assert "pct" in dist.columns
        assert dist["count"].sum() == 20

    def test_empty_df(self):
        dist = clv_distribution(pd.DataFrame())
        assert dist.empty


class TestApplyFilters:
    def test_filter_by_sport(self):
        rows = _synthetic_clv_rows(10)
        df = build_clv_dataframe(rows)
        filtered = apply_filters(df, sport="NFL")
        assert len(filtered) == 5  # first 5 are NFL

    def test_filter_by_market(self):
        rows = _synthetic_clv_rows(9)
        df = build_clv_dataframe(rows)
        filtered = apply_filters(df, market="ML")
        assert len(filtered) == 3  # indices 0, 3, 6

    def test_filter_by_confidence(self):
        rows = _synthetic_clv_rows(9)
        df = build_clv_dataframe(rows)
        filtered = apply_filters(df, confidence="High")
        assert len(filtered) == 3  # indices 0, 3, 6

    def test_no_filter(self):
        rows = _synthetic_clv_rows(5)
        df = build_clv_dataframe(rows)
        filtered = apply_filters(df)
        assert len(filtered) == 5

    def test_empty_result(self):
        rows = _synthetic_clv_rows(5)
        df = build_clv_dataframe(rows)
        filtered = apply_filters(df, sport="WNBA")
        assert len(filtered) == 0


# -------------------------------------------------------------------
# Snapshot metadata integration tests
# -------------------------------------------------------------------


class TestSnapshotMetadata:
    def test_snapshot_stores_metadata(self, tmp_path):
        """snapshot_pick now stores sportsbook, sport, confidence, etc."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            lines = [
                _ml_line("DraftKings", -150, 130),
                _ml_line("FanDuel", -140, 120),
                _ml_line("BetMGM", -160, 140),
            ]
            store.save_lines(lines)

            bet = _make_bet(
                legs=[_leg(odds=-150, selection="Home")],
            )
            snapshot_pick(bet, store)

            rows = store.get_clv(bet.id)
            assert len(rows) == 1
            r = rows[0]
            # New metadata fields should be populated
            assert r["pick_sportsbook"] == "DraftKings"
            assert r["sport"] == "NFL"
            assert r["confidence_at_pick"] in ("High", "Medium", "Low")
            assert r["quality_tier_at_pick"] in (
                "Elite", "Strong", "Moderate", "Thin",
            )
            assert r["edge_pct_at_pick"] is not None
            assert r["edge_z_at_pick"] is not None
            assert r["books_used_at_pick"] is not None
            assert r["agreement_score_at_pick"] is not None

    def test_snapshot_metadata_absent_gracefully(self, tmp_path):
        """Old-style save_clv_pick without metadata doesn't crash."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_clv_pick(
                bet_id="old_style",
                leg_index=0,
                event="Test",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-110.0,
                pick_odds_decimal=1.909,
                consensus_prob_at_pick=0.52,
            )
            rows = store.get_clv("old_style")
            assert len(rows) == 1
            # Metadata columns should be None
            assert rows[0]["pick_sportsbook"] is None
            assert rows[0]["sport"] is None
