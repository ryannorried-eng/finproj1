"""Tests for the LineTracker."""

from datetime import datetime

from line_tracker.models import BettingLine, BetType
from line_tracker.tracker import LineTracker


def _make_line(sportsbook: str, home_odds: float, away_odds: float) -> BettingLine:
    return BettingLine(
        sportsbook=sportsbook,
        sport="NFL",
        event="KC vs BUF",
        bet_type=BetType.MONEYLINE,
        home_team="KC",
        away_team="BUF",
        home_value=home_odds,
        away_value=away_odds,
        timestamp=datetime.now(),
    )


def test_add_and_retrieve_lines():
    tracker = LineTracker()
    line = _make_line("DraftKings", -150, 130)
    tracker.add_line(line)

    results = tracker.get_lines_for_event("KC vs BUF")
    assert len(results) == 1
    assert results[0].sportsbook == "DraftKings"


def test_filter_by_bet_type():
    tracker = LineTracker()
    tracker.add_line(_make_line("DraftKings", -150, 130))

    assert len(tracker.get_lines_for_event("KC vs BUF", BetType.MONEYLINE)) == 1
    assert len(tracker.get_lines_for_event("KC vs BUF", BetType.SPREAD)) == 0


def test_best_moneyline():
    tracker = LineTracker()
    tracker.add_line(_make_line("DraftKings", -150, 130))
    tracker.add_line(_make_line("FanDuel", -140, 125))
    tracker.add_line(_make_line("BetMGM", -155, 135))

    best_home = tracker.get_best_moneyline("KC vs BUF", "home")
    assert best_home is not None
    assert best_home.sportsbook == "FanDuel"  # -140 is the best (least negative)

    best_away = tracker.get_best_moneyline("KC vs BUF", "away")
    assert best_away is not None
    assert best_away.sportsbook == "BetMGM"  # +135 is the best
