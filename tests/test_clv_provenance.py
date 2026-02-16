"""Tests for pick-analytics provenance guards.

Proves that pick-time analytics (edge_pct, edge_z, hold, sigma,
confidence, tier) are computed ONLY from lines available at or before
pick_timestamp, and that provenance metadata (pick_lines_max_ts,
pick_lines_count) is recorded correctly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from line_tracker.best_bets import (
    _compute_pick_analytics,
)
from line_tracker.bet_history import Bet, settle_bet
from line_tracker.models import BettingLine, BetType
from line_tracker.storage import LineStore

T0 = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

# Timestamps used across tests
BEFORE_PICK = T0 - timedelta(hours=2)
PICK_TIME = T0 - timedelta(hours=1)
AFTER_PICK = T0 - timedelta(minutes=30)
COMMENCE = T0


def _store(tmp_path) -> LineStore:
    return LineStore(tmp_path / "prov.db")


def _ml_line(
    sportsbook: str,
    home_odds: float,
    away_odds: float,
    timestamp: datetime,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="nfl",
        event="Bills @ Chiefs",
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=timestamp,
        commence_time=COMMENCE,
    )


# -----------------------------------------------------------------------
# 1. pick_lines_max_ts <= pick_timestamp
# -----------------------------------------------------------------------

class TestPickProvenanceMaxTs:
    def test_pick_provenance_max_ts_leq_pick_ts(self, tmp_path):
        """Lines fetched AFTER pick_timestamp must NOT be used.

        Insert two snapshots: one BEFORE pick, one AFTER pick.
        Assert pick_lines_max_ts equals the "before" snapshot time,
        not the "after" one.
        """
        store = _store(tmp_path)

        before_line = _ml_line("DK", -150, 130, BEFORE_PICK)
        after_line = _ml_line("FD", -140, 120, AFTER_PICK)
        store.save_lines([before_line, after_line])

        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
            "market": "ML",
            "fetched_at": PICK_TIME.isoformat(),
        }

        result = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )

        max_ts_str = result["pick_lines_max_ts"]
        assert max_ts_str is not None
        max_ts = datetime.fromisoformat(max_ts_str)

        # The max timestamp used must be <= pick time
        assert max_ts <= PICK_TIME
        # Specifically, it should be the BEFORE_PICK line
        assert max_ts == BEFORE_PICK

    def test_multiple_books_before_pick(self, tmp_path):
        """When multiple books have lines before pick, all are used."""
        store = _store(tmp_path)

        t1 = PICK_TIME - timedelta(hours=3)
        t2 = PICK_TIME - timedelta(hours=1)
        store.save_lines([
            _ml_line("DK", -150, 130, t1),
            _ml_line("FD", -145, 125, t2),
            _ml_line("BM", -200, 170, AFTER_PICK),  # after pick
        ])

        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
        }

        result = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )

        # Only DK and FD should be used (not BM which is after pick)
        assert result["pick_lines_count"] == 2
        assert result["books_used"] == 2

        max_ts = datetime.fromisoformat(result["pick_lines_max_ts"])
        assert max_ts <= PICK_TIME

    def test_no_lines_before_pick(self, tmp_path):
        """If all lines are after pick_timestamp, none should be used."""
        store = _store(tmp_path)
        store.save_lines([
            _ml_line("DK", -150, 130, AFTER_PICK),
        ])

        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
        }

        result = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )

        assert result["pick_lines_count"] == 0
        assert result["books_used"] == 0
        assert result["pick_lines_max_ts"] is None


# -----------------------------------------------------------------------
# 2. Pick analytics stable after new fetch
# -----------------------------------------------------------------------

class TestPickAnalyticsStability:
    def test_pick_analytics_stable_after_new_fetch(self, tmp_path):
        """Inserting newer lines must not change persisted analytics."""
        store = _store(tmp_path)

        # Initial lines at pick time
        store.save_lines([
            _ml_line("DK", -150, 130, BEFORE_PICK),
            _ml_line("FD", -145, 125, BEFORE_PICK),
        ])

        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
        }

        # Compute analytics as of pick time
        result1 = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )

        # Simulate a new fetch after pick time (very different odds)
        store.save_lines([
            _ml_line("DK", -300, 250, AFTER_PICK),
            _ml_line("FD", -280, 240, AFTER_PICK),
            _ml_line("BM", -260, 220, AFTER_PICK),
        ])

        # Re-compute: should get same result since we bound by pick time
        result2 = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )

        assert result1["edge_pct"] == result2["edge_pct"]
        assert result1["edge_z"] == result2["edge_z"]
        assert result1["market_hold_median"] == result2["market_hold_median"]
        assert result1["market_volatility_sigma"] == result2["market_volatility_sigma"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["quality_tier"] == result2["quality_tier"]
        assert result1["books_used"] == result2["books_used"]

    def test_settle_persists_provenance_fields(self, tmp_path):
        """settle_bet persists pick_lines_max_ts and pick_lines_count."""
        store = _store(tmp_path)
        store.save_lines([
            _ml_line("DraftKings", -170, 150, BEFORE_PICK),
        ])

        bet = Bet(
            id="prov_test_001",
            sportsbook="DraftKings",
            legs=[{
                "sport": "NFL",
                "event_name": "Bills @ Chiefs",
                "sportsbook": "DraftKings",
                "market": "ML",
                "selection": "Home",
                "line": None,
                "odds": -150,
                "fetched_at": PICK_TIME.isoformat(),
            }],
            combined_decimal=1.6667,
            combined_american=-150,
            stake=100.0,
            profit=66.67,
            total_payout=166.67,
            status="active",
            created_at=(T0 - timedelta(hours=5)).isoformat(),
            commence_time=COMMENCE.isoformat(),
        )
        state = {"active_bets": [bet], "settled_bets": []}
        settle_bet(state, bet.id, "won", store=store)

        rows = store.get_clv_rows(include_estimated=True)
        assert len(rows) == 1
        row = rows[0]

        assert row["pick_lines_max_ts"] is not None
        max_ts = datetime.fromisoformat(row["pick_lines_max_ts"])
        pick_ts = datetime.fromisoformat(row["pick_timestamp"])
        assert max_ts <= pick_ts

        assert row["pick_lines_count"] is not None
        assert row["pick_lines_count"] >= 1


# -----------------------------------------------------------------------
# 3. Provenance count is reasonable
# -----------------------------------------------------------------------

class TestProvenanceCount:
    def test_provenance_count_matches_books(self, tmp_path):
        """pick_lines_count should equal the number of line rows used."""
        store = _store(tmp_path)
        store.save_lines([
            _ml_line("DK", -150, 130, BEFORE_PICK),
            _ml_line("FD", -145, 125, BEFORE_PICK),
            _ml_line("BM", -155, 135, BEFORE_PICK),
        ])

        leg = {
            "event_name": "Bills @ Chiefs",
            "selection": "Home",
            "odds": -150,
        }

        result = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )

        assert result["pick_lines_count"] == 3
        assert result["books_used"] == 3

    def test_provenance_count_zero_when_no_data(self, tmp_path):
        store = _store(tmp_path)
        leg = {
            "event_name": "Nonexistent Game",
            "selection": "Home",
            "odds": -150,
        }
        result = _compute_pick_analytics(
            leg, store, BetType.MONEYLINE, -150,
            pick_timestamp=PICK_TIME,
        )
        assert result["pick_lines_count"] == 0
        assert result["pick_lines_max_ts"] is None


# -----------------------------------------------------------------------
# 4. Integration: end-to-end time leakage detection
# -----------------------------------------------------------------------

class TestEndToEndLeakage:
    def test_settle_does_not_leak_future_lines(self, tmp_path):
        """Full integration: settle a bet where future lines exist.

        Verifies that the stored pick analytics only reflect lines
        available at pick time, not lines fetched later.
        """
        store = _store(tmp_path)

        # Lines available at pick time: -150 / +130 from one book
        store.save_lines([
            _ml_line("DraftKings", -150, 130, BEFORE_PICK),
            _ml_line("FanDuel", -145, 125, BEFORE_PICK),
        ])

        # Lines fetched AFTER pick (dramatically different odds)
        store.save_lines([
            _ml_line("DraftKings", -300, 250, AFTER_PICK),
            _ml_line("FanDuel", -290, 240, AFTER_PICK),
            _ml_line("BetMGM", -280, 230, AFTER_PICK),
        ])

        bet = Bet(
            id="leak_test_001",
            sportsbook="DraftKings",
            legs=[{
                "sport": "NFL",
                "event_name": "Bills @ Chiefs",
                "sportsbook": "DraftKings",
                "market": "ML",
                "selection": "Home",
                "line": None,
                "odds": -150,
                "fetched_at": PICK_TIME.isoformat(),
            }],
            combined_decimal=1.6667,
            combined_american=-150,
            stake=100.0,
            profit=66.67,
            total_payout=166.67,
            status="active",
            created_at=PICK_TIME.isoformat(),
            commence_time=COMMENCE.isoformat(),
        )

        state = {"active_bets": [bet], "settled_bets": []}
        settle_bet(state, bet.id, "won", store=store)

        rows = store.get_clv_rows(include_estimated=True)
        assert len(rows) == 1
        row = rows[0]

        # Provenance check: max_ts must be at or before pick time
        max_ts = datetime.fromisoformat(row["pick_lines_max_ts"])
        assert max_ts <= PICK_TIME

        # books_used should be 2 (DK + FD before pick), not 3
        assert row["books_used"] == 2

        # pick_lines_count should be 2 (the two before-pick snapshots)
        assert row["pick_lines_count"] == 2


# -----------------------------------------------------------------------
# 5. get_lines_asof correctness
# -----------------------------------------------------------------------

class TestGetLinesAsof:
    def test_returns_latest_per_book_before_cutoff(self, tmp_path):
        store = _store(tmp_path)

        # DK has two snapshots: old and recent (both before pick)
        store.save_lines([
            _ml_line("DK", -150, 130, BEFORE_PICK - timedelta(hours=1)),
            _ml_line("DK", -155, 135, BEFORE_PICK),
            # FD only has one snapshot before pick
            _ml_line("FD", -145, 125, BEFORE_PICK),
            # BM has one snapshot after pick
            _ml_line("BM", -140, 120, AFTER_PICK),
        ])

        lines = store.get_lines_asof(
            "Bills @ Chiefs", BetType.MONEYLINE, PICK_TIME,
        )
        books = {ln.sportsbook for ln in lines}
        assert books == {"DK", "FD"}

        # DK should have the *latest* pre-pick snapshot (-155)
        dk = next(ln for ln in lines if ln.sportsbook == "DK")
        assert dk.home_value == -155

    def test_empty_when_all_after_cutoff(self, tmp_path):
        store = _store(tmp_path)
        store.save_lines([
            _ml_line("DK", -150, 130, AFTER_PICK),
        ])
        lines = store.get_lines_asof(
            "Bills @ Chiefs", BetType.MONEYLINE, PICK_TIME,
        )
        assert lines == []
