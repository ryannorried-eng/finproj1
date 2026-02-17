"""Tests for SQLite storage layer."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def test_default_path_migrates_legacy_db(tmp_path, monkeypatch):
    legacy_db = tmp_path / "lines.db"
    new_db = tmp_path / ".line_tracker" / "lines.db"

    with LineStore(legacy_db) as legacy_store:
        legacy_store.save_lines([_make_line()])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("line_tracker.storage.DEFAULT_DB_PATH", new_db)
    monkeypatch.setattr("line_tracker.storage.LEGACY_DB_PATH", Path("lines.db"))

    with LineStore() as store:
        assert store.db_path == new_db

    assert new_db.exists()
    with LineStore(new_db) as migrated_store:
        assert len(migrated_store.get_lines()) == 1

    # Idempotent safety: existing new DB is not overwritten on re-open.
    with LineStore(new_db) as explicit_store:
        explicit_store.save_lines([_make_line(event="Eagles @ Cowboys")])
    with LineStore() as store:
        assert store.db_path == new_db
    with LineStore(new_db) as persisted_store:
        assert len(persisted_store.get_lines()) == 2


def test_transaction_rolls_back_on_error(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.insert_bet({
                    "bet_id": "b1",
                    "created_at": "2026-01-01T00:00:00Z",
                    "sportsbook": "DK",
                    "stake": 100.0,
                    "total_odds_american": -110,
                    "total_odds_decimal": 1.9091,
                    "potential_payout": 190.91,
                    "profit": 90.91,
                    "status": "active",
                    "settled_at": None,
                    "outcome": None,
                })
                raise RuntimeError("boom")

        assert store.get_bets() == []


def test_foreign_keys_are_enforced(tmp_path):
    db = tmp_path / "test.db"
    with LineStore(db) as store:
        with pytest.raises(Exception):
            store.insert_legs(
                "missing-bet",
                [{
                    "leg_id": "leg-1",
                    "sport": "NFL",
                    "market": "ML",
                    "event_name": "Bills @ Chiefs",
                    "selection": "Home",
                    "line_value": None,
                    "odds_american": -110,
                    "odds_decimal": 1.9091,
                    "sportsbook": "DK",
                    "pick_timestamp": "2026-01-01T00:00:00Z",
                    "commence_time": None,
                }],
            )
