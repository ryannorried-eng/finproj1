"""Tests for line movement detection."""

from datetime import datetime, timezone

from line_tracker.models import BettingLine, BetType
from line_tracker.movements import detect_moves

T1 = datetime(2026, 1, 19, 17, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 19, 18, 0, tzinfo=timezone.utc)


def _line(home_value, timestamp, **kw):
    defaults = dict(
        sportsbook="DraftKings",
        sport="nfl",
        event="Bills @ Chiefs",
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=home_value,
        away_value=130,
        timestamp=timestamp,
    )
    defaults.update(kw)
    return BettingLine(**defaults)


def test_detect_moneyline_move():
    old = [_line(-150, T1)]
    new = [_line(-170, T2)]
    moves = detect_moves(old, new)
    assert len(moves) == 1
    assert moves[0].change == -20
    assert moves[0].direction == "down"


def test_no_move():
    old = [_line(-150, T1)]
    new = [_line(-150, T2)]
    moves = detect_moves(old, new)
    assert len(moves) == 0


def test_threshold_filters():
    old = [_line(-150, T1)]
    new = [_line(-155, T2)]
    # change = -5, threshold = 10 -> filtered out
    moves = detect_moves(old, new, threshold=10)
    assert len(moves) == 0
    # threshold = 3 -> included
    moves = detect_moves(old, new, threshold=3)
    assert len(moves) == 1


def test_spread_move():
    old = [_line(-3.5, T1, bet_type=BetType.SPREAD)]
    new = [_line(-4.5, T2, bet_type=BetType.SPREAD)]
    moves = detect_moves(old, new)
    assert len(moves) == 1
    assert moves[0].change == -1.0


def test_multiple_books():
    old = [
        _line(-150, T1, sportsbook="DraftKings"),
        _line(-145, T1, sportsbook="FanDuel"),
    ]
    new = [
        _line(-160, T2, sportsbook="DraftKings"),
        _line(-145, T2, sportsbook="FanDuel"),
    ]
    moves = detect_moves(old, new)
    # Only DK moved
    assert len(moves) == 1
    assert moves[0].sportsbook == "DraftKings"


def test_new_line_no_crash():
    old = []
    new = [_line(-150, T2)]
    moves = detect_moves(old, new)
    assert len(moves) == 0
