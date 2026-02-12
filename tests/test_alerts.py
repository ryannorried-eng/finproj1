"""Tests for the alert system."""

from datetime import datetime, timezone

from line_tracker.alerts import AlertLevel, AlertManager
from line_tracker.arbitrage import ArbOpportunity
from line_tracker.models import BettingLine, BetType
from line_tracker.movements import LineMove

TS = datetime(2026, 1, 19, 18, 0, tzinfo=timezone.utc)


def _line(home_value, **kw):
    defaults = dict(
        sportsbook="DraftKings",
        sport="nfl",
        event="Bills @ Chiefs",
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=home_value,
        away_value=130,
        timestamp=TS,
    )
    defaults.update(kw)
    return BettingLine(**defaults)


def test_movement_alert_triggered():
    mgr = AlertManager(move_threshold=10)
    move = LineMove(
        event="Bills @ Chiefs",
        sportsbook="DraftKings",
        bet_type=BetType.MONEYLINE,
        old_line=_line(-150),
        new_line=_line(-170),
        change=-20,
    )
    alerts = mgr.check_movements([move])
    assert len(alerts) == 1
    assert alerts[0].source == "movement"
    assert "DraftKings" in alerts[0].message


def test_movement_below_threshold():
    mgr = AlertManager(move_threshold=10)
    move = LineMove(
        event="Bills @ Chiefs",
        sportsbook="DraftKings",
        bet_type=BetType.MONEYLINE,
        old_line=_line(-150),
        new_line=_line(-155),
        change=-5,
    )
    alerts = mgr.check_movements([move])
    assert len(alerts) == 0


def test_critical_alert_for_large_move():
    mgr = AlertManager(move_threshold=10)
    move = LineMove(
        event="Bills @ Chiefs",
        sportsbook="DraftKings",
        bet_type=BetType.MONEYLINE,
        old_line=_line(-150),
        new_line=_line(-200),
        change=-50,
    )
    alerts = mgr.check_movements([move])
    assert alerts[0].level == AlertLevel.CRITICAL


def test_arb_alert():
    mgr = AlertManager(arb_min_margin=0)
    arb = ArbOpportunity(
        event="Bills @ Chiefs",
        bet_type=BetType.MONEYLINE,
        side_a=_line(150),
        side_b=_line(200, sportsbook="FanDuel"),
        margin=2.5,
    )
    alerts = mgr.check_arbitrage([arb])
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL
    assert "ARB" in alerts[0].message


def test_arb_below_min_margin():
    mgr = AlertManager(arb_min_margin=5)
    arb = ArbOpportunity(
        event="Bills @ Chiefs",
        bet_type=BetType.MONEYLINE,
        side_a=_line(150),
        side_b=_line(200, sportsbook="FanDuel"),
        margin=2.5,
    )
    alerts = mgr.check_arbitrage([arb])
    assert len(alerts) == 0


def test_callback_invoked():
    received = []
    mgr = AlertManager(
        move_threshold=5,
        callbacks=[lambda a: received.append(a)],
    )
    move = LineMove(
        event="Bills @ Chiefs",
        sportsbook="DraftKings",
        bet_type=BetType.MONEYLINE,
        old_line=_line(-150),
        new_line=_line(-170),
        change=-20,
    )
    mgr.check_movements([move])
    assert len(received) == 1
