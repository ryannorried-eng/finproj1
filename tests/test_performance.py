"""Tests for the CLV-driven Performance analytics module.

Covers:
  1. KPI computation (means, medians, beat-close %)
  2. Breakdown table correctness (grouping counts and metrics)
  3. Trend function for 14 days of data
  4. Calibration suggestions respect sample-size minimums
  5. Filtering by sport/market/book/confidence/tier
  6. Storage round-trip for CLV rows
  7. Pick-time analytics computation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from line_tracker.best_bets import (
    _classify_confidence,
    _classify_tier,
    _compute_pick_analytics,
    _extract_odds_for_selection,
)
from line_tracker.models import BettingLine, BetType
from line_tracker.performance import (
    calibration_suggestions,
    compute_breakdown_tables,
    compute_kpis,
    compute_trends,
    load_clv_df,
)
from line_tracker.storage import LineStore

T0 = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_store(tmp_path) -> LineStore:
    return LineStore(tmp_path / "test_perf.db")


def _make_clv_row(
    *,
    bet_id: str = "bet001",
    leg_index: int = 0,
    sport: str = "NFL",
    market: str = "ML",
    pick_sportsbook: str = "DraftKings",
    event_name: str = "Bills @ Chiefs",
    selection: str = "Home",
    pick_odds: float = -150,
    exec_clv_prob: float = 0.02,
    market_clv_prob: float = 0.03,
    beat_close_exec: int = 1,
    beat_close_market: int = 1,
    close_estimated: bool = False,
    confidence: str = "High",
    quality_tier: str = "Tier1",
    edge_pct: float = 0.015,
    edge_z: float = 1.8,
    books_used: int = 5,
    market_hold_median: float = 0.035,
    market_volatility_sigma: float = 0.008,
    settled_at: str | None = None,
    outcome: str = "won",
) -> dict:
    if settled_at is None:
        settled_at = T0.isoformat()
    return {
        "bet_id": bet_id,
        "leg_index": leg_index,
        "sport": sport,
        "market": market,
        "pick_sportsbook": pick_sportsbook,
        "event_name": event_name,
        "selection": selection,
        "pick_line": None,
        "pick_odds": pick_odds,
        "pick_timestamp": (T0 - timedelta(hours=5)).isoformat(),
        "commence_time": T0.isoformat(),
        "edge_pct": edge_pct,
        "edge_z": edge_z,
        "books_used": books_used,
        "market_hold_median": market_hold_median,
        "market_volatility_sigma": market_volatility_sigma,
        "confidence": confidence,
        "quality_tier": quality_tier,
        "close_timestamp": None,
        "close_estimated": close_estimated,
        "exec_close_odds": -170,
        "market_close_odds": -160,
        "exec_close_decimal": 1.5882,
        "market_close_decimal": 1.625,
        "exec_clv_prob": exec_clv_prob,
        "market_clv_prob": market_clv_prob,
        "beat_close_exec": beat_close_exec,
        "beat_close_market": beat_close_market,
        "settled_at": settled_at,
        "outcome": outcome,
    }


def _make_n_rows(n: int, **overrides) -> list[dict]:
    """Create n CLV rows with unique bet_ids and sequential dates."""
    rows = []
    for i in range(n):
        kw = {
            "bet_id": f"bet_{i:04d}",
            "settled_at": (
                T0 + timedelta(days=i)
            ).isoformat(),
            **overrides,
        }
        rows.append(_make_clv_row(**kw))
    return rows


# ---------------------------------------------------------------------------
# 1. KPI computation
# ---------------------------------------------------------------------------

class TestComputeKPIs:
    def test_basic_kpis(self):
        rows = [
            _make_clv_row(
                bet_id="a", exec_clv_prob=0.02,
                beat_close_exec=1,
            ),
            _make_clv_row(
                bet_id="b", exec_clv_prob=-0.01,
                beat_close_exec=0,
            ),
            _make_clv_row(
                bet_id="c", exec_clv_prob=0.005,
                beat_close_exec=1,
            ),
        ]
        df = pd.DataFrame(rows)
        kpis = compute_kpis(df)

        assert kpis["sample_size"] == 3
        # Mean: (0.02 + -0.01 + 0.005) / 3 = 0.005
        assert abs(kpis["avg_clv_prob_pts_exec"] - 0.50) < 0.1
        # Beat close: 2/3 ≈ 66.7%
        assert abs(kpis["pct_beat_close_exec"] - 66.7) < 0.1

    def test_empty_df_returns_zeros(self):
        df = pd.DataFrame()
        kpis = compute_kpis(df)
        assert kpis["sample_size"] == 0
        assert kpis["avg_clv_prob_pts_exec"] is None

    def test_all_beat_close(self):
        rows = _make_n_rows(
            5, exec_clv_prob=0.03, beat_close_exec=1,
        )
        kpis = compute_kpis(pd.DataFrame(rows))
        assert kpis["pct_beat_close_exec"] == 100.0

    def test_none_beat_close(self):
        rows = _make_n_rows(
            5, exec_clv_prob=-0.02, beat_close_exec=0,
        )
        kpis = compute_kpis(pd.DataFrame(rows))
        assert kpis["pct_beat_close_exec"] == 0.0

    def test_median_calculation(self):
        rows = [
            _make_clv_row(
                bet_id="a", exec_clv_prob=0.01,
                beat_close_exec=1,
            ),
            _make_clv_row(
                bet_id="b", exec_clv_prob=0.05,
                beat_close_exec=1,
            ),
            _make_clv_row(
                bet_id="c", exec_clv_prob=0.10,
                beat_close_exec=1,
            ),
        ]
        kpis = compute_kpis(pd.DataFrame(rows))
        # Median of [1.0, 5.0, 10.0] prob pts = 5.0
        assert kpis["median_clv_prob_pts_exec"] == 5.0


# ---------------------------------------------------------------------------
# 2. Breakdown table correctness
# ---------------------------------------------------------------------------

class TestBreakdownTables:
    def test_grouping_counts(self):
        rows = (
            _make_n_rows(3, quality_tier="Tier1")
            + _make_n_rows(2, quality_tier="Tier2")
        )
        # Fix bet_ids to be unique
        for i, r in enumerate(rows):
            r["bet_id"] = f"u_{i}"
        df = pd.DataFrame(rows)
        tables = compute_breakdown_tables(df)

        by_tier = tables["by_tier"]
        assert "Tier1" in by_tier.index
        assert "Tier2" in by_tier.index
        assert by_tier.loc["Tier1", "count"] == 3
        assert by_tier.loc["Tier2", "count"] == 2

    def test_market_breakdown(self):
        rows = [
            _make_clv_row(bet_id="a", market="ML"),
            _make_clv_row(bet_id="b", market="ML"),
            _make_clv_row(bet_id="c", market="Spread"),
        ]
        df = pd.DataFrame(rows)
        tables = compute_breakdown_tables(df)

        by_market = tables["by_market"]
        assert by_market.loc["ML", "count"] == 2
        assert by_market.loc["Spread", "count"] == 1

    def test_book_breakdown(self):
        rows = [
            _make_clv_row(
                bet_id="a", pick_sportsbook="DraftKings",
            ),
            _make_clv_row(
                bet_id="b", pick_sportsbook="FanDuel",
            ),
        ]
        df = pd.DataFrame(rows)
        tables = compute_breakdown_tables(df)

        by_book = tables["by_book"]
        assert "DraftKings" in by_book.index
        assert "FanDuel" in by_book.index

    def test_avg_metrics_in_breakdown(self):
        rows = [
            _make_clv_row(
                bet_id="a", quality_tier="Tier1",
                exec_clv_prob=0.02, beat_close_exec=1,
                edge_pct=0.01, edge_z=1.5,
            ),
            _make_clv_row(
                bet_id="b", quality_tier="Tier1",
                exec_clv_prob=0.04, beat_close_exec=1,
                edge_pct=0.03, edge_z=2.0,
            ),
        ]
        df = pd.DataFrame(rows)
        tables = compute_breakdown_tables(df)

        row = tables["by_tier"].loc["Tier1"]
        assert row["pct_beat_close"] == 100.0
        # avg CLV: (2.0 + 4.0) / 2 = 3.0
        assert abs(row["avg_clv_prob_pts"] - 3.0) < 0.01
        # avg edge_z: (1.5 + 2.0) / 2 = 1.75
        assert abs(row["avg_edge_z"] - 1.75) < 0.01

    def test_empty_df_returns_empty_tables(self):
        tables = compute_breakdown_tables(pd.DataFrame())
        for key in (
            "by_tier", "by_confidence", "by_market",
            "by_sport", "by_book",
        ):
            assert key in tables
            assert tables[key].empty


# ---------------------------------------------------------------------------
# 3. Trends for 14 days of data
# ---------------------------------------------------------------------------

class TestComputeTrends:
    def test_14_day_trend(self):
        rows = []
        for day in range(14):
            rows.append(_make_clv_row(
                bet_id=f"d{day}",
                exec_clv_prob=0.01 * (day + 1),
                beat_close_exec=1 if day % 2 == 0 else 0,
                settled_at=(
                    T0 + timedelta(days=day)
                ).isoformat(),
            ))
        df = pd.DataFrame(rows)
        df["settled_dt"] = pd.to_datetime(
            df["settled_at"], utc=True,
        )

        trends = compute_trends(df)
        assert trends["reason"] is None
        assert len(trends["daily"]) == 14
        assert "pct_beat_close_exec" in trends["daily"].columns
        assert "avg_clv_prob_pts_exec" in trends["daily"].columns
        assert len(trends["rolling_7"]) == 14

    def test_empty_df_returns_reason(self):
        trends = compute_trends(pd.DataFrame())
        assert trends["reason"] is not None
        assert trends["daily"].empty

    def test_insufficient_data_returns_reason(self):
        rows = [_make_clv_row(bet_id="x")]
        df = pd.DataFrame(rows)
        df["settled_dt"] = pd.to_datetime(
            df["settled_at"], utc=True,
        )
        trends = compute_trends(df)
        assert "1" in trends["reason"]


# ---------------------------------------------------------------------------
# 4. Calibration suggestions
# ---------------------------------------------------------------------------

class TestCalibrationSuggestions:
    def test_insufficient_data_message(self):
        rows = _make_n_rows(10)
        df = pd.DataFrame(rows)
        tables = compute_breakdown_tables(df)
        msgs = calibration_suggestions(
            df, tables["by_tier"], tables["by_confidence"],
        )
        assert len(msgs) >= 1
        assert "Need more" in msgs[0]

    def test_sufficient_data_no_crash(self):
        # 120 rows, enough for calibration
        rows = _make_n_rows(
            120, quality_tier="Tier1", confidence="High",
        )
        for i, r in enumerate(rows):
            r["bet_id"] = f"cal_{i}"
        df = pd.DataFrame(rows)
        tables = compute_breakdown_tables(df)
        msgs = calibration_suggestions(
            df, tables["by_tier"], tables["by_confidence"],
        )
        assert len(msgs) >= 1
        assert len(msgs) <= 6

    def test_tier_comparison_suggestion(self):
        # 50 Tier1 with lower CLV, 50 Tier2 with higher CLV
        t1 = _make_n_rows(
            50, quality_tier="Tier1", exec_clv_prob=0.005,
        )
        t2 = _make_n_rows(
            50, quality_tier="Tier2", exec_clv_prob=0.02,
        )
        # also 50 more to hit 100+ total
        t3 = _make_n_rows(
            20, quality_tier="Tier3", exec_clv_prob=0.01,
        )
        all_rows = t1 + t2 + t3
        for i, r in enumerate(all_rows):
            r["bet_id"] = f"tc_{i}"
        df = pd.DataFrame(all_rows)
        tables = compute_breakdown_tables(df)
        msgs = calibration_suggestions(
            df, tables["by_tier"], tables["by_confidence"],
        )
        # Should mention Tier 1 not outperforming Tier 2
        assert any("Tier 1" in m for m in msgs)

    def test_low_conf_negative_clv_suggestion(self):
        high = _make_n_rows(
            50, confidence="High", exec_clv_prob=0.03,
        )
        low = _make_n_rows(
            50, confidence="Low", exec_clv_prob=-0.02,
        )
        extra = _make_n_rows(
            20, confidence="Medium", exec_clv_prob=0.01,
        )
        all_rows = high + low + extra
        for i, r in enumerate(all_rows):
            r["bet_id"] = f"lc_{i}"
        df = pd.DataFrame(all_rows)
        tables = compute_breakdown_tables(df)
        msgs = calibration_suggestions(
            df, tables["by_tier"], tables["by_confidence"],
        )
        assert any("Low" in m for m in msgs)

    def test_small_tier_groups_no_strong_claim(self):
        # Only 10 per tier — below MIN_SAMPLE_STRONG
        t1 = _make_n_rows(
            10, quality_tier="Tier1", exec_clv_prob=0.005,
        )
        t2 = _make_n_rows(
            10, quality_tier="Tier2", exec_clv_prob=0.02,
        )
        rest = _make_n_rows(
            90, quality_tier="Tier3", exec_clv_prob=0.01,
        )
        all_rows = t1 + t2 + rest
        for i, r in enumerate(all_rows):
            r["bet_id"] = f"sg_{i}"
        df = pd.DataFrame(all_rows)
        tables = compute_breakdown_tables(df)
        msgs = calibration_suggestions(
            df, tables["by_tier"], tables["by_confidence"],
        )
        # Should not make strong tier claims with n<30
        assert not any(
            "Tier 1 does not outperform" in m for m in msgs
        )


# ---------------------------------------------------------------------------
# 5. Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_filter_by_sport(self, tmp_path):
        store = _tmp_store(tmp_path)
        nfl = _make_clv_row(bet_id="nfl1", sport="NFL")
        nba = _make_clv_row(bet_id="nba1", sport="NBA")
        store.save_clv_rows([nfl, nba])

        df = load_clv_df(store, sport="NFL", include_estimated=True)
        assert len(df) == 1
        assert df.iloc[0]["sport"] == "NFL"

    def test_filter_by_market(self, tmp_path):
        store = _tmp_store(tmp_path)
        ml = _make_clv_row(bet_id="ml1", market="ML")
        sp = _make_clv_row(bet_id="sp1", market="Spread")
        store.save_clv_rows([ml, sp])

        df = load_clv_df(
            store, market="Spread", include_estimated=True,
        )
        assert len(df) == 1
        assert df.iloc[0]["market"] == "Spread"

    def test_filter_by_book(self, tmp_path):
        store = _tmp_store(tmp_path)
        dk = _make_clv_row(
            bet_id="dk1", pick_sportsbook="DraftKings",
        )
        fd = _make_clv_row(
            bet_id="fd1", pick_sportsbook="FanDuel",
        )
        store.save_clv_rows([dk, fd])

        df = load_clv_df(
            store, book="FanDuel", include_estimated=True,
        )
        assert len(df) == 1

    def test_filter_by_confidence(self, tmp_path):
        store = _tmp_store(tmp_path)
        hi = _make_clv_row(bet_id="hi1", confidence="High")
        lo = _make_clv_row(bet_id="lo1", confidence="Low")
        store.save_clv_rows([hi, lo])

        df = load_clv_df(
            store, confidence="High", include_estimated=True,
        )
        assert len(df) == 1

    def test_filter_by_tier(self, tmp_path):
        store = _tmp_store(tmp_path)
        t1 = _make_clv_row(bet_id="t1", quality_tier="Tier1")
        t2 = _make_clv_row(bet_id="t2", quality_tier="Tier2")
        store.save_clv_rows([t1, t2])

        df = load_clv_df(
            store, tier="Tier1", include_estimated=True,
        )
        assert len(df) == 1

    def test_exclude_estimated_by_default(self, tmp_path):
        store = _tmp_store(tmp_path)
        good = _make_clv_row(
            bet_id="g1", close_estimated=False,
        )
        est = _make_clv_row(
            bet_id="e1", close_estimated=True,
        )
        store.save_clv_rows([good, est])

        df = load_clv_df(store, include_estimated=False)
        assert len(df) == 1

    def test_include_estimated(self, tmp_path):
        store = _tmp_store(tmp_path)
        good = _make_clv_row(
            bet_id="g1", close_estimated=False,
        )
        est = _make_clv_row(
            bet_id="e1", close_estimated=True,
        )
        store.save_clv_rows([good, est])

        df = load_clv_df(store, include_estimated=True)
        assert len(df) == 2

    def test_date_range_filter(self, tmp_path):
        store = _tmp_store(tmp_path)
        old = _make_clv_row(
            bet_id="old",
            settled_at=(T0 - timedelta(days=60)).isoformat(),
        )
        recent = _make_clv_row(
            bet_id="new",
            settled_at=T0.isoformat(),
        )
        store.save_clv_rows([old, recent])

        start = (T0 - timedelta(days=30)).isoformat()
        df = load_clv_df(
            store, start=start, include_estimated=True,
        )
        assert len(df) == 1
        assert df.iloc[0]["bet_id"] == "new"


# ---------------------------------------------------------------------------
# 6. Storage round-trip
# ---------------------------------------------------------------------------

class TestStorageRoundTrip:
    def test_save_and_load(self, tmp_path):
        store = _tmp_store(tmp_path)
        row = _make_clv_row()
        store.save_clv_rows([row])

        loaded = store.get_clv_rows(include_estimated=True)
        assert len(loaded) == 1
        assert loaded[0]["bet_id"] == "bet001"
        assert loaded[0]["exec_clv_prob"] == 0.02
        assert loaded[0]["quality_tier"] == "Tier1"
        assert loaded[0]["confidence"] == "High"

    def test_upsert_replaces_existing(self, tmp_path):
        store = _tmp_store(tmp_path)
        row1 = _make_clv_row(exec_clv_prob=0.01)
        store.save_clv_rows([row1])

        row2 = _make_clv_row(exec_clv_prob=0.05)
        store.save_clv_rows([row2])

        loaded = store.get_clv_rows(include_estimated=True)
        assert len(loaded) == 1
        assert loaded[0]["exec_clv_prob"] == 0.05


# ---------------------------------------------------------------------------
# 7. Pick-time analytics
# ---------------------------------------------------------------------------

class TestPickAnalytics:
    def test_extract_odds_moneyline(self):
        line = BettingLine(
            sportsbook="DK", sport="nfl",
            event="A @ B", bet_type=BetType.MONEYLINE,
            home_team="B", away_team="A",
            home_value=-150, away_value=130,
            timestamp=T0,
        )
        assert _extract_odds_for_selection(
            line, BetType.MONEYLINE, "Home",
        ) == -150
        assert _extract_odds_for_selection(
            line, BetType.MONEYLINE, "Away",
        ) == 130

    def test_extract_odds_spread(self):
        line = BettingLine(
            sportsbook="DK", sport="nfl",
            event="A @ B", bet_type=BetType.SPREAD,
            home_team="B", away_team="A",
            home_value=-3.5, away_value=3.5,
            home_price=-110, away_price=-110,
            timestamp=T0,
        )
        assert _extract_odds_for_selection(
            line, BetType.SPREAD, "Home",
        ) == -110

    def test_classify_confidence_high(self):
        assert _classify_confidence(2.5, 0.03, 5) == "High"

    def test_classify_confidence_medium(self):
        assert _classify_confidence(1.5, 0.03, 3) == "Medium"

    def test_classify_confidence_low(self):
        assert _classify_confidence(0.5, 0.03, 2) == "Low"

    def test_classify_tier1(self):
        assert _classify_tier(0.02, 1.8, 0.04, 5) == "Tier1"

    def test_classify_tier2(self):
        assert _classify_tier(0.01, 0.9, 0.06, 3) == "Tier2"

    def test_classify_tier3(self):
        assert _classify_tier(0.005, 0.3, 0.10, 2) == "Tier3"

    def test_classify_stayaway(self):
        assert _classify_tier(-0.01, -0.5, 0.04, 5) == "StayAway"

    def test_compute_pick_analytics_with_data(self, tmp_path):
        store = _tmp_store(tmp_path)
        # Save 4 books' lines
        lines = []
        for book, home_odds, away_odds in [
            ("DK", -150, 130),
            ("FD", -145, 125),
            ("BM", -155, 135),
            ("CZ", -148, 128),
        ]:
            lines.append(BettingLine(
                sportsbook=book, sport="nfl",
                event="Bills @ Chiefs",
                bet_type=BetType.MONEYLINE,
                home_team="Chiefs", away_team="Bills",
                home_value=home_odds, away_value=away_odds,
                timestamp=T0 - timedelta(hours=1),
                commence_time=T0,
            ))
        store.save_lines(lines)

        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
            "market": "ML",
        }
        analytics = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
        )
        assert analytics["books_used"] == 4
        assert analytics["edge_pct"] is not None
        assert analytics["confidence"] in (
            "High", "Medium", "Low",
        )
        assert analytics["quality_tier"] in (
            "Tier1", "Tier2", "Tier3", "StayAway",
        )

    def test_compute_pick_analytics_few_books(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([BettingLine(
            sportsbook="DK", sport="nfl",
            event="Bills @ Chiefs",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-150, away_value=130,
            timestamp=T0 - timedelta(hours=1),
            commence_time=T0,
        )])
        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
        }
        analytics = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
        )
        assert analytics["books_used"] == 1
        assert analytics["edge_pct"] is None
        assert analytics["confidence"] == "Low"


# ---------------------------------------------------------------------------
# 8. Integration: settle_bet persists CLV rows
# ---------------------------------------------------------------------------

class TestSettlePersistsCLV:
    def test_settle_writes_clv_table(self, tmp_path):
        from line_tracker.bet_history import Bet, settle_bet

        store = _tmp_store(tmp_path)
        # Set up close lines
        store.save_lines([BettingLine(
            sportsbook="DraftKings", sport="nfl",
            event="Bills @ Chiefs",
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs", away_team="Bills",
            home_value=-170, away_value=150,
            timestamp=T0 - timedelta(hours=1),
            commence_time=T0,
        )])

        bet = Bet(
            id="settle_test_001",
            sportsbook="DraftKings",
            legs=[{
                "sport": "NFL",
                "event_name": "Bills @ Chiefs",
                "sportsbook": "DraftKings",
                "market": "ML",
                "selection": "Home",
                "line": None,
                "odds": -150,
                "fetched_at": (
                    T0 - timedelta(hours=5)
                ).isoformat(),
            }],
            combined_decimal=1.6667,
            combined_american=-150,
            stake=100.0,
            profit=66.67,
            total_payout=166.67,
            status="active",
            created_at=(
                T0 - timedelta(hours=5)
            ).isoformat(),
            commence_time=T0.isoformat(),
        )
        state = {"active_bets": [bet], "settled_bets": []}
        settle_bet(state, bet.id, "won", store=store)

        # Verify CLV row was persisted
        clv_rows = store.get_clv_rows(include_estimated=True)
        assert len(clv_rows) == 1
        row = clv_rows[0]
        assert row["bet_id"] == "settle_test_001"
        assert row["market"] == "ML"
        assert row["exec_clv_prob"] is not None
        assert row["quality_tier"] is not None
        assert row["confidence"] is not None
