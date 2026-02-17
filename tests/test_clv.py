"""Tests for Closing Line Value (CLV) tracking."""

from datetime import datetime, timezone

import pytest

from line_tracker.bet_history import (
    close_bet_clv,
    compute_clv,
    create_bet,
    snapshot_pick,
)
from line_tracker.models import BettingLine, BetType
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


def _spread_line(
    sportsbook: str,
    spread: float,
    home_price: float,
    away_price: float,
    event: str = "Bills @ Chiefs",
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event=event,
        bet_type=BetType.SPREAD,
        home_team="Chiefs",
        away_team="Bills",
        home_value=spread,
        away_value=-spread,
        timestamp=_NOW,
        home_price=home_price,
        away_price=away_price,
    )


def _total_line(
    sportsbook: str,
    total: float,
    over_price: float,
    under_price: float,
    event: str = "Bills @ Chiefs",
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event=event,
        bet_type=BetType.TOTAL,
        home_team="Chiefs",
        away_team="Bills",
        home_value=total,
        away_value=total,
        timestamp=_NOW,
        home_price=over_price,
        away_price=under_price,
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


# -------------------------------------------------------------------
# Storage: bet_clv table round-trip
# -------------------------------------------------------------------


class TestCLVStorage:
    def test_save_and_get_pick(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_clv_pick(
                bet_id="abc123",
                leg_index=0,
                event="Bills @ Chiefs",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-150.0,
                pick_odds_decimal=1.6667,
                consensus_prob_at_pick=0.58,
                market_hold_median_at_pick=4.5,
                market_volatility_sigma_at_pick=0.012,
            )
            rows = store.get_clv("abc123")
            assert len(rows) == 1
            r = rows[0]
            assert r["bet_id"] == "abc123"
            assert r["leg_index"] == 0
            assert r["event"] == "Bills @ Chiefs"
            assert r["pick_odds_american"] == -150.0
            assert r["consensus_prob_at_pick"] == 0.58
            assert r["closed_at"] is None

    def test_close_clv_updates_row(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_clv_pick(
                bet_id="abc123",
                leg_index=0,
                event="Bills @ Chiefs",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-150.0,
                pick_odds_decimal=1.6667,
                consensus_prob_at_pick=0.58,
            )
            store.close_clv(
                bet_id="abc123",
                leg_index=0,
                consensus_prob_close=0.62,
                best_odds_close_american=-170.0,
                best_odds_close_decimal=1.5882,
            )
            rows = store.get_clv("abc123")
            r = rows[0]
            assert r["consensus_prob_close"] == 0.62
            assert r["best_odds_close_american"] == -170.0
            assert r["closed_at"] is not None

    def test_multiple_legs(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            for i in range(3):
                store.save_clv_pick(
                    bet_id="parlay1",
                    leg_index=i,
                    event=f"Event {i}",
                    market="ML",
                    pick_side="Home",
                    pick_line_value=None,
                    pick_odds_american=-110.0,
                    pick_odds_decimal=1.9091,
                    consensus_prob_at_pick=0.52,
                )
            rows = store.get_clv("parlay1")
            assert len(rows) == 3
            assert [r["leg_index"] for r in rows] == [0, 1, 2]

    def test_upsert_replaces_existing(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_clv_pick(
                bet_id="bet1",
                leg_index=0,
                event="Bills @ Chiefs",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-150.0,
                pick_odds_decimal=1.6667,
                consensus_prob_at_pick=0.58,
            )
            # Re-insert same key with different prob
            store.save_clv_pick(
                bet_id="bet1",
                leg_index=0,
                event="Bills @ Chiefs",
                market="ML",
                pick_side="Home",
                pick_line_value=None,
                pick_odds_american=-155.0,
                pick_odds_decimal=1.6452,
                consensus_prob_at_pick=0.60,
            )
            rows = store.get_clv("bet1")
            assert len(rows) == 1
            assert rows[0]["pick_odds_american"] == -155.0

    def test_get_clv_unknown_bet_returns_empty(self, tmp_path):
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            assert store.get_clv("nonexistent") == []


# -------------------------------------------------------------------
# compute_clv
# -------------------------------------------------------------------


class TestComputeCLV:
    @pytest.mark.parametrize(
        ("label", "row", "expected_prob_sign", "expected_dec_sign"),
        [
            (
                "favorite_beats_close",
                {
                    "pick_odds_decimal": 1.80,
                    "best_odds_close_decimal": 1.70,
                    "consensus_prob_at_pick": 0.55,
                    "consensus_prob_close": 0.59,
                },
                1,
                1,
            ),
            (
                "favorite_loses_close",
                {
                    "pick_odds_decimal": 1.80,
                    "best_odds_close_decimal": 1.95,
                    "consensus_prob_at_pick": 0.55,
                    "consensus_prob_close": 0.50,
                },
                -1,
                -1,
            ),
            (
                "underdog_beats_close",
                {
                    "pick_odds_decimal": 2.40,
                    "best_odds_close_decimal": 2.20,
                    "consensus_prob_at_pick": 0.42,
                    "consensus_prob_close": 0.46,
                },
                1,
                1,
            ),
            (
                "underdog_loses_close",
                {
                    "pick_odds_decimal": 2.20,
                    "best_odds_close_decimal": 2.45,
                    "consensus_prob_at_pick": 0.45,
                    "consensus_prob_close": 0.40,
                },
                -1,
                -1,
            ),
            (
                "spread_like_beats_close",
                {
                    "pick_odds_decimal": 1.95,
                    "best_odds_close_decimal": 1.87,
                    "consensus_prob_at_pick": 0.52,
                    "consensus_prob_close": 0.54,
                },
                1,
                1,
            ),
            (
                "spread_like_loses_close",
                {
                    "pick_odds_decimal": 1.91,
                    "best_odds_close_decimal": 2.00,
                    "consensus_prob_at_pick": 0.51,
                    "consensus_prob_close": 0.49,
                },
                -1,
                -1,
            ),
        ],
    )
    def test_clv_sign_consistency(self, label, row, expected_prob_sign, expected_dec_sign):
        m = compute_clv(row)
        assert m is not None, label

        prob_sign = 0 if m["clv_prob"] == 0 else (1 if m["clv_prob"] > 0 else -1)
        dec_sign = 0 if m["clv_decimal"] == 0 else (1 if m["clv_decimal"] > 0 else -1)

        assert prob_sign == expected_prob_sign, label
        assert dec_sign == expected_dec_sign, label

    def test_no_close_data_returns_none(self):
        row = {
            "pick_odds_decimal": 1.80,
            "best_odds_close_decimal": None,
            "consensus_prob_at_pick": 0.55,
        }
        assert compute_clv(row) is None

    def test_zero_clv(self):
        row = {
            "pick_odds_decimal": 2.0,
            "best_odds_close_decimal": 2.0,
            "consensus_prob_at_pick": 0.50,
            "consensus_prob_close": 0.50,
        }
        m = compute_clv(row)
        assert m["clv_decimal"] == 0.0
        assert m["clv_prob"] == 0.0


# -------------------------------------------------------------------
# snapshot_pick — integration (uses real consensus)
# -------------------------------------------------------------------


class TestSnapshotPick:
    def test_moneyline_snapshot(self, tmp_path):
        """Snapshot stores consensus + pick odds for a ML bet."""
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
            assert r["market"] == "ML"
            assert r["pick_side"] == "Home"
            assert r["pick_odds_american"] == -150.0
            assert r["consensus_prob_at_pick"] > 0

    def test_spread_snapshot(self, tmp_path):
        """Snapshot works for spread bets."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            lines = [
                _spread_line("DraftKings", -3.5, -110, -110),
                _spread_line("FanDuel", -3.5, -105, -115),
                _spread_line("BetMGM", -3.5, -108, -112),
            ]
            store.save_lines(lines)

            bet = _make_bet(
                legs=[_leg(
                    market="Spread",
                    selection="Home",
                    line=-3.5,
                    odds=-110,
                )],
            )
            snapshot_pick(bet, store)

            rows = store.get_clv(bet.id)
            assert len(rows) == 1
            assert rows[0]["market"] == "Spread"
            assert rows[0]["pick_line_value"] == -3.5

    def test_total_snapshot(self, tmp_path):
        """Snapshot works for total bets."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            lines = [
                _total_line("DraftKings", 48.5, -110, -110),
                _total_line("FanDuel", 48.5, -105, -115),
                _total_line("BetMGM", 48.5, -108, -112),
            ]
            store.save_lines(lines)

            bet = _make_bet(
                legs=[_leg(
                    market="Total",
                    selection="Over",
                    line=48.5,
                    odds=-110,
                )],
            )
            snapshot_pick(bet, store)

            rows = store.get_clv(bet.id)
            assert len(rows) == 1
            assert rows[0]["market"] == "Total"
            assert rows[0]["pick_side"] == "Over"

    def test_no_lines_no_crash(self, tmp_path):
        """If no lines exist for the event, snapshot is a no-op."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            bet = _make_bet(
                legs=[_leg(event_name="Nonexistent Game")],
            )
            snapshot_pick(bet, store)
            assert store.get_clv(bet.id) == []

    def test_multi_leg_snapshot(self, tmp_path):
        """Parlay snapshot stores one row per leg."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            store.save_lines([
                _ml_line("DraftKings", -150, 130),
                _ml_line("FanDuel", -140, 120),
                _ml_line(
                    "DraftKings", -200, 180,
                    event="Knicks @ Heat",
                ),
                _ml_line(
                    "FanDuel", -190, 170,
                    event="Knicks @ Heat",
                ),
            ])

            bet = _make_bet(
                legs=[
                    _leg(
                        odds=-150,
                        selection="Home",
                    ),
                    _leg(
                        event_name="Knicks @ Heat",
                        odds=180,
                        selection="Away",
                    ),
                ],
            )
            snapshot_pick(bet, store)

            rows = store.get_clv(bet.id)
            assert len(rows) == 2
            assert rows[0]["event"] == "Bills @ Chiefs"
            assert rows[1]["event"] == "Knicks @ Heat"


# -------------------------------------------------------------------
# close_bet_clv — integration
# -------------------------------------------------------------------


class TestCloseBetCLV:
    def test_close_updates_consensus_and_odds(self, tmp_path):
        """Closing writes consensus_prob_close and best_odds_close."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # Save initial lines for pick
            lines = [
                _ml_line("DraftKings", -150, 130),
                _ml_line("FanDuel", -140, 120),
            ]
            store.save_lines(lines)

            bet = _make_bet(
                legs=[_leg(odds=-150, selection="Home")],
            )
            snapshot_pick(bet, store)

            # Simulate line movement: odds shift
            moved = [
                _ml_line("DraftKings", -170, 150),
                _ml_line("FanDuel", -160, 140),
            ]
            store.save_lines(moved)

            close_bet_clv(bet.id, store)

            rows = store.get_clv(bet.id)
            r = rows[0]
            assert r["closed_at"] is not None
            assert r["consensus_prob_close"] is not None
            assert r["best_odds_close_american"] is not None

    def test_close_already_closed_is_noop(self, tmp_path):
        """Calling close twice does not overwrite the first close."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            lines = [
                _ml_line("DraftKings", -150, 130),
                _ml_line("FanDuel", -140, 120),
            ]
            store.save_lines(lines)

            bet = _make_bet(
                legs=[_leg(odds=-150, selection="Home")],
            )
            snapshot_pick(bet, store)
            close_bet_clv(bet.id, store)

            first_close = store.get_clv(bet.id)[0]
            first_closed_at = first_close["closed_at"]

            # Shift lines again
            store.save_lines([
                _ml_line("DraftKings", -200, 180),
                _ml_line("FanDuel", -190, 170),
            ])
            close_bet_clv(bet.id, store)

            second_close = store.get_clv(bet.id)[0]
            # closed_at should NOT change
            assert second_close["closed_at"] == first_closed_at

    def test_close_no_snapshot_is_noop(self, tmp_path):
        """If there's no pick snapshot, close is a no-op."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            close_bet_clv("nonexistent", store)
            # Should not raise


# -------------------------------------------------------------------
# End-to-end CLV flow
# -------------------------------------------------------------------


class TestCLVEndToEnd:
    def test_full_flow_pick_to_close_to_compute(self, tmp_path):
        """Full CLV flow: pick snapshot → line movement → close → compute."""
        db = tmp_path / "test.db"
        with LineStore(db) as store:
            # 1. Store initial lines
            store.save_lines([
                _ml_line("DraftKings", -150, 130),
                _ml_line("FanDuel", -140, 120),
                _ml_line("BetMGM", -160, 140),
            ])

            # 2. Place bet and snapshot
            bet = _make_bet(
                legs=[_leg(odds=-150, selection="Home")],
            )
            snapshot_pick(bet, store)

            rows_before = store.get_clv(bet.id)
            assert len(rows_before) == 1
            pick_prob = rows_before[0]["consensus_prob_at_pick"]
            pick_dec = rows_before[0]["pick_odds_decimal"]
            assert pick_prob > 0
            assert pick_dec > 1

            # 3. Simulate line movement (home becomes bigger fav)
            store.save_lines([
                _ml_line("DraftKings", -180, 160),
                _ml_line("FanDuel", -170, 150),
                _ml_line("BetMGM", -190, 170),
            ])

            # 4. Close
            close_bet_clv(bet.id, store)

            rows_after = store.get_clv(bet.id)
            r = rows_after[0]
            assert r["closed_at"] is not None
            close_prob = r["consensus_prob_close"]

            # Home became bigger favorite → consensus_prob_close > pick
            assert close_prob > pick_prob

            # 5. Compute CLV
            m = compute_clv(r)
            assert m is not None
            # Closing decimal odds should be lower (bigger fav)
            assert m["clv_decimal"] > 0
            # Closing prob should be higher
            assert m["clv_prob"] > 0
