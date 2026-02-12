"""Tests for SQLite storage layer."""

from datetime import datetime, timezone

from line_tracker.models import BettingLine, BetType
from line_tracker.storage import LineStore


def _make_line(**overrides) -> BettingLine:
    defaults = dict(
        sportsbook="DraftKings",
        sport="americanfootball_nfl",
        event="Bills @ Chiefs",
        bet_type=BetType.MONEYLINE,
        home_team="Chiefs",
        away_team="Bills",
        home_value=-150,
        away_value=130,
        timestamp=datetime(2026, 1, 19, 18, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return BettingLine(**defaults)


def test_save_and_retrieve(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        lines = [_make_line(), _make_line(sportsbook="FanDuel")]
        count = store.save_lines(lines)
        assert count == 2

        result = store.get_lines()
        assert len(result) == 2


def test_filter_by_event(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        store.save_lines([
            _make_line(event="Bills @ Chiefs"),
            _make_line(event="Eagles @ Cowboys"),
        ])
        result = store.get_lines(event="Bills @ Chiefs")
        assert len(result) == 1
        assert result[0].event == "Bills @ Chiefs"


def test_filter_by_bet_type(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        store.save_lines([
            _make_line(bet_type=BetType.MONEYLINE),
            _make_line(bet_type=BetType.SPREAD, home_value=-3.5,
                       away_value=3.5, home_price=-110, away_price=-110),
        ])
        result = store.get_lines(bet_type=BetType.SPREAD)
        assert len(result) == 1
        assert result[0].bet_type == BetType.SPREAD


def test_get_latest_for_event(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        store.save_lines([
            _make_line(home_value=-150,
                       timestamp=datetime(2026, 1, 19, 17, 0,
                                          tzinfo=timezone.utc)),
            _make_line(home_value=-160,
                       timestamp=datetime(2026, 1, 19, 18, 0,
                                          tzinfo=timezone.utc)),
        ])
        latest = store.get_latest_for_event(
            "Bills @ Chiefs", BetType.MONEYLINE
        )
        assert len(latest) == 1
        assert latest[0].home_value == -160


def test_get_events(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        store.save_lines([
            _make_line(event="Bills @ Chiefs"),
            _make_line(event="Eagles @ Cowboys"),
        ])
        events = store.get_events()
        assert events == ["Bills @ Chiefs", "Eagles @ Cowboys"]


def test_empty_db(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        assert store.get_lines() == []
        assert store.get_events() == []


def test_prices_roundtrip(tmp_path):
    db = tmp_path / "test.db"
    line = _make_line(
        bet_type=BetType.SPREAD,
        home_value=-3.5,
        away_value=3.5,
        home_price=-110,
        away_price=-108,
    )
    with LineStore(db) as store:
        store.save_lines([line])
        result = store.get_lines()[0]
        assert result.home_price == -110
        assert result.away_price == -108
