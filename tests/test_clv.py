"""Tests for Closing Line Value (CLV) computation.

Covers:
  - Favorite and underdog sign correctness for clv_price_prob
  - Close time selection relative to commence_time
  - Same-book vs best-book close
  - Spread/total line-value matching
  - Tolerance classification
  - No close data scenario
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from line_tracker.best_bets import (
    CLV_TOLERANCE,
    _classify,
    _find_close_odds_for_leg,
    compute_clv,
    enrich_bet_with_clv,
)
from line_tracker.bet_history import Bet, settle_bet
from line_tracker.models import BettingLine, BetType
from line_tracker.storage import LineStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 20, 1, 0, 0, tzinfo=timezone.utc)  # commence_time


def _make_line(
    sportsbook: str,
    bet_type: BetType,
    home_value: float,
    away_value: float,
    timestamp: datetime,
    home_price: float | None = None,
    away_price: float | None = None,
    commence_time: datetime = T0,
) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="americanfootball_nfl",
        event="Bills @ Chiefs",
        bet_type=bet_type,
        home_team="Kansas City Chiefs",
        away_team="Buffalo Bills",
        home_value=home_value,
        away_value=away_value,
        home_price=home_price,
        away_price=away_price,
        timestamp=timestamp,
        commence_time=commence_time,
    )


def _tmp_store(tmp_path) -> LineStore:
    return LineStore(tmp_path / "test.db")


def _make_bet(
    sportsbook: str = "DraftKings",
    legs: list[dict] | None = None,
) -> Bet:
    if legs is None:
        legs = [
            {
                "sport": "NFL",
                "event_name": "Bills @ Chiefs",
                "sportsbook": sportsbook,
                "market": "ML",
                "selection": "Home",
                "line": None,
                "odds": -150,
                "fetched_at": (T0 - timedelta(hours=5)).isoformat(),
            }
        ]
    return Bet(
        id="abc123",
        sportsbook=sportsbook,
        legs=legs,
        combined_decimal=1.6667,
        combined_american=-150,
        stake=100.0,
        profit=66.67,
        total_payout=166.67,
        status="active",
        created_at=(T0 - timedelta(hours=5)).isoformat(),
        commence_time=T0.isoformat(),
    )


# ---------------------------------------------------------------------------
# 1. Favorite sign correctness
# ---------------------------------------------------------------------------

class TestFavoriteSignCorrectness:
    def test_favorite_beat_close(self):
        """Pick home fav at -150, close at -170.  Close implies higher prob
        so bettor got a better (lower-prob) price → positive clv_price_prob."""
        result = compute_clv(pick_odds=-150, close_odds=-170)
        assert result["clv_price_prob"] > 0
        assert result["classification"] == "beat"

    def test_favorite_lost_to_close(self):
        """Pick home fav at -170, close at -150.  Close moved in bettor's
        favor (lower implied prob at close) → negative clv_price_prob."""
        result = compute_clv(pick_odds=-170, close_odds=-150)
        assert result["clv_price_prob"] < 0
        assert result["classification"] == "lost"

    def test_favorite_clv_decimal_direction(self):
        """When you beat the close, your decimal odds are higher than close."""
        result = compute_clv(pick_odds=-150, close_odds=-170)
        # -150 → 1.6667, -170 → 1.5882  ⇒ clv_decimal > 0
        assert result["clv_decimal"] > 0


# ---------------------------------------------------------------------------
# 2. Underdog sign correctness
# ---------------------------------------------------------------------------

class TestUnderdogSignCorrectness:
    def test_underdog_beat_close(self):
        """Pick away dog at +130, close at +120.  Close implies higher prob
        (shorter price) so bettor got better value → positive clv_price_prob."""
        result = compute_clv(pick_odds=130, close_odds=120)
        assert result["clv_price_prob"] > 0
        assert result["classification"] == "beat"

    def test_underdog_lost_to_close(self):
        """Pick away dog at +120, close at +130.  Close drifted out →
        bettor took worse price → negative clv_price_prob."""
        result = compute_clv(pick_odds=120, close_odds=130)
        assert result["clv_price_prob"] < 0
        assert result["classification"] == "lost"

    def test_underdog_clv_decimal_direction(self):
        """Beating the close as underdog: higher decimal at pick time."""
        result = compute_clv(pick_odds=130, close_odds=120)
        # +130 → 2.30, +120 → 2.20  ⇒ clv_decimal > 0
        assert result["clv_decimal"] > 0


# ---------------------------------------------------------------------------
# 3. Close time selection relative to commence_time
# ---------------------------------------------------------------------------

class TestCloseTimeSelection:
    def test_uses_last_line_before_commence(self, tmp_path):
        """Only lines with timestamp <= commence_time should be used."""
        store = _tmp_store(tmp_path)
        lines = [
            _make_line("DK", BetType.MONEYLINE, -150, 130,
                        T0 - timedelta(hours=2)),  # old
            _make_line("DK", BetType.MONEYLINE, -160, 140,
                        T0 - timedelta(hours=1)),  # latest pre-game
            _make_line("DK", BetType.MONEYLINE, -180, 160,
                        T0 + timedelta(minutes=30)),  # LIVE — must be excluded
        ]
        store.save_lines(lines)

        close = store.get_close_lines("Bills @ Chiefs", BetType.MONEYLINE, before=T0)
        assert len(close) == 1
        assert close[0].home_value == -160  # the T-1h line, not the live one

    def test_no_lines_before_commence(self, tmp_path):
        """If all lines are after commence_time, close list is empty."""
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DK", BetType.MONEYLINE, -180, 160,
                        T0 + timedelta(minutes=5)),
        ])
        close = store.get_close_lines("Bills @ Chiefs", BetType.MONEYLINE, before=T0)
        assert close == []


# ---------------------------------------------------------------------------
# 4. Same-book vs best-book close
# ---------------------------------------------------------------------------

class TestSameBookVsBestBook:
    def test_best_book_picks_best_odds(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DraftKings", BetType.MONEYLINE, -150, 130,
                        T0 - timedelta(hours=1)),
            _make_line("FanDuel", BetType.MONEYLINE, -145, 125,
                        T0 - timedelta(hours=1)),
        ])

        # Best-book close (all books)
        close_all = store.get_close_lines(
            "Bills @ Chiefs", BetType.MONEYLINE, before=T0,
        )
        assert len(close_all) == 2

        # Same-book close
        close_dk = store.get_close_lines(
            "Bills @ Chiefs", BetType.MONEYLINE, before=T0, sportsbook="DraftKings",
        )
        assert len(close_dk) == 1
        assert close_dk[0].sportsbook == "DraftKings"

    def test_enrich_computes_both(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DraftKings", BetType.MONEYLINE, -170, 150,
                        T0 - timedelta(hours=1)),
            _make_line("FanDuel", BetType.MONEYLINE, -160, 140,
                        T0 - timedelta(hours=1)),
        ])

        bet = _make_bet(sportsbook="DraftKings")
        clvs = enrich_bet_with_clv(bet, store, commence_time=T0)
        lc = clvs[0]

        # Best close should use the best odds across books for Home selection
        # DK: -170, FD: -160 → best home odds = -160 (less negative = better for bettor)
        assert lc.best_close_odds == -160
        # Book close should use DraftKings
        assert lc.book_close_odds == -170

        # Both should be computed
        assert lc.clv_price_prob_best is not None
        assert lc.clv_price_prob_book is not None


# ---------------------------------------------------------------------------
# 5. Spread line-value matching
# ---------------------------------------------------------------------------

class TestSpreadMatching:
    def test_matches_on_line_value(self, tmp_path):
        store = _tmp_store(tmp_path)
        # Close has spread -2.5 and -3.0 from different books
        store.save_lines([
            _make_line("DraftKings", BetType.SPREAD, -2.5, 2.5,
                        T0 - timedelta(hours=1),
                        home_price=-110, away_price=-110),
            _make_line("FanDuel", BetType.SPREAD, -3.0, 3.0,
                        T0 - timedelta(hours=1),
                        home_price=-105, away_price=-115),
        ])

        # Bet was placed on Home -2.5
        leg = {
            "sport": "NFL", "event_name": "Bills @ Chiefs",
            "sportsbook": "DraftKings", "market": "Spread",
            "selection": "Home", "line": -2.5, "odds": -110,
            "fetched_at": "",
        }
        close_lines = store.get_close_lines(
            "Bills @ Chiefs", BetType.SPREAD, before=T0,
        )
        result = _find_close_odds_for_leg(leg, close_lines, BetType.SPREAD)

        # Should match -2.5 line (DK), not -3.0 line (FD)
        assert result == -110

    def test_no_match_when_line_moved(self, tmp_path):
        store = _tmp_store(tmp_path)
        # Close only has -3.0
        store.save_lines([
            _make_line("DraftKings", BetType.SPREAD, -3.0, 3.0,
                        T0 - timedelta(hours=1),
                        home_price=-110, away_price=-110),
        ])

        # Bet was on Home -2.5 — line moved, so no match
        leg = {
            "sport": "NFL", "event_name": "Bills @ Chiefs",
            "sportsbook": "DraftKings", "market": "Spread",
            "selection": "Home", "line": -2.5, "odds": -110,
            "fetched_at": "",
        }
        close_lines = store.get_close_lines(
            "Bills @ Chiefs", BetType.SPREAD, before=T0,
        )
        result = _find_close_odds_for_leg(leg, close_lines, BetType.SPREAD)
        assert result is None


# ---------------------------------------------------------------------------
# 6. Total line-value matching
# ---------------------------------------------------------------------------

class TestTotalMatching:
    def test_over_matches_on_total_number(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DraftKings", BetType.TOTAL, 47.5, 47.5,
                        T0 - timedelta(hours=1),
                        home_price=-110, away_price=-110),
        ])

        leg = {
            "sport": "NFL", "event_name": "Bills @ Chiefs",
            "sportsbook": "DraftKings", "market": "Total",
            "selection": "Over", "line": 47.5, "odds": -110,
            "fetched_at": "",
        }
        close_lines = store.get_close_lines(
            "Bills @ Chiefs", BetType.TOTAL, before=T0,
        )
        result = _find_close_odds_for_leg(leg, close_lines, BetType.TOTAL)
        assert result == -110

    def test_under_matches_on_total_number(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DraftKings", BetType.TOTAL, 48.0, 48.0,
                        T0 - timedelta(hours=1),
                        home_price=-105, away_price=-115),
        ])

        leg = {
            "sport": "NFL", "event_name": "Bills @ Chiefs",
            "sportsbook": "DraftKings", "market": "Total",
            "selection": "Under", "line": 48.0, "odds": -115,
            "fetched_at": "",
        }
        close_lines = store.get_close_lines(
            "Bills @ Chiefs", BetType.TOTAL, before=T0,
        )
        result = _find_close_odds_for_leg(leg, close_lines, BetType.TOTAL)
        assert result == -115

    def test_no_match_when_total_moved(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DraftKings", BetType.TOTAL, 48.0, 48.0,
                        T0 - timedelta(hours=1),
                        home_price=-110, away_price=-110),
        ])

        leg = {
            "sport": "NFL", "event_name": "Bills @ Chiefs",
            "sportsbook": "DraftKings", "market": "Total",
            "selection": "Over", "line": 47.5, "odds": -110,
            "fetched_at": "",
        }
        close_lines = store.get_close_lines(
            "Bills @ Chiefs", BetType.TOTAL, before=T0,
        )
        result = _find_close_odds_for_leg(leg, close_lines, BetType.TOTAL)
        assert result is None


# ---------------------------------------------------------------------------
# 7. Tolerance classification
# ---------------------------------------------------------------------------

class TestToleranceClassification:
    def test_tiny_positive_is_matched(self):
        assert _classify(0.0005) == "matched"

    def test_above_tolerance_is_beat(self):
        assert _classify(0.002) == "beat"

    def test_below_negative_tolerance_is_lost(self):
        assert _classify(-0.002) == "lost"

    def test_exactly_at_tolerance_is_matched(self):
        assert _classify(CLV_TOLERANCE) == "matched"
        assert _classify(-CLV_TOLERANCE) == "matched"

    def test_just_above_tolerance_is_beat(self):
        assert _classify(CLV_TOLERANCE + 0.0001) == "beat"

    def test_none_is_no_close(self):
        assert _classify(None) == "no_close"

    def test_zero_is_matched(self):
        assert _classify(0.0) == "matched"

    def test_identical_odds_classified_matched(self):
        result = compute_clv(pick_odds=-150, close_odds=-150)
        assert result["classification"] == "matched"
        assert result["clv_price_prob"] == 0.0


# ---------------------------------------------------------------------------
# 8. No close data
# ---------------------------------------------------------------------------

class TestNoCloseData:
    def test_no_commence_time_marks_estimated(self, tmp_path):
        store = _tmp_store(tmp_path)
        bet = _make_bet()
        bet.commence_time = None  # no commence_time

        clvs = enrich_bet_with_clv(bet, store, commence_time=None)
        assert clvs[0].close_estimated is True
        assert clvs[0].classification_best == "no_close"

    def test_no_matching_lines_marks_estimated(self, tmp_path):
        store = _tmp_store(tmp_path)
        # Store has no lines at all
        bet = _make_bet()
        # Even with commence_time, no lines → estimated
        clvs = enrich_bet_with_clv(bet, store, commence_time=T0)
        assert clvs[0].close_estimated is True


# ---------------------------------------------------------------------------
# 9. settle_bet integration
# ---------------------------------------------------------------------------

class TestSettleBetCLV:
    def test_settle_attaches_clv_when_store_provided(self, tmp_path):
        store = _tmp_store(tmp_path)
        store.save_lines([
            _make_line("DraftKings", BetType.MONEYLINE, -170, 150,
                        T0 - timedelta(hours=1)),
        ])

        bet = _make_bet()
        state = {"active_bets": [bet], "settled_bets": []}
        settled = settle_bet(state, bet.id, "won", store=store)

        assert settled.clv is not None
        assert len(settled.clv) == 1
        assert settled.clv[0]["clv_price_prob_best"] is not None

    def test_settle_without_store_has_no_clv(self):
        bet = _make_bet()
        state = {"active_bets": [bet], "settled_bets": []}
        settled = settle_bet(state, bet.id, "won")
        assert settled.clv is None


# ---------------------------------------------------------------------------
# 10. Scraper commence_time pass-through
# ---------------------------------------------------------------------------

class TestScraperCommenceTime:
    def test_parse_events_sets_commence_time(self):
        from line_tracker.scraper import _parse_events

        events = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "commence_time": "2026-01-20T01:00:00Z",
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-01-19T18:30:00Z",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -150},
                        {"name": "Buffalo Bills", "price": 130},
                    ],
                }],
            }],
        }]
        lines = _parse_events(events, "americanfootball_nfl")
        assert len(lines) == 1
        assert lines[0].commence_time is not None
        assert lines[0].commence_time == T0

    def test_parse_events_no_commence_time(self):
        from line_tracker.scraper import _parse_events

        events = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-01-19T18:30:00Z",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -150},
                        {"name": "Buffalo Bills", "price": 130},
                    ],
                }],
            }],
        }]
        lines = _parse_events(events, "americanfootball_nfl")
        assert lines[0].commence_time is None


# ---------------------------------------------------------------------------
# 11. Storage round-trip with commence_time
# ---------------------------------------------------------------------------

class TestStorageCommenceTime:
    def test_save_and_read_commence_time(self, tmp_path):
        store = _tmp_store(tmp_path)
        line = _make_line("DK", BetType.MONEYLINE, -150, 130,
                          T0 - timedelta(hours=1), commence_time=T0)
        store.save_lines([line])

        loaded = store.get_lines(event="Bills @ Chiefs")
        assert len(loaded) == 1
        assert loaded[0].commence_time == T0

    def test_save_and_read_null_commence_time(self, tmp_path):
        store = _tmp_store(tmp_path)
        line = _make_line("DK", BetType.MONEYLINE, -150, 130,
                          T0 - timedelta(hours=1), commence_time=None)
        line.commence_time = None
        store.save_lines([line])

        loaded = store.get_lines(event="Bills @ Chiefs")
        assert loaded[0].commence_time is None
