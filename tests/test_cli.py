"""Tests for the CLI entry point."""

from datetime import datetime, timezone

from line_tracker.__main__ import main
from line_tracker.models import BettingLine, BetType
from line_tracker.storage import LineStore


def test_no_command_shows_help(capsys):
    ret = main([])
    assert ret == 0


def test_events_empty_db(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    ret = main(["events", "--db", db])
    assert ret == 0
    out = capsys.readouterr().out
    assert "No events" in out


def test_lines_empty_db(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    ret = main(["lines", "--db", db])
    assert ret == 0
    out = capsys.readouterr().out
    assert "No lines" in out


def test_moves_needs_data(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    ret = main(["moves", "--db", db])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Need at least" in out


def test_moves_uses_latest_two_snapshots_only(tmp_path, capsys):
    db = tmp_path / "moves.db"
    event = "Bills @ Chiefs"

    def _line(home_value: float, ts_hour: int) -> BettingLine:
        return BettingLine(
            sportsbook="DraftKings",
            sport="americanfootball_nfl",
            event=event,
            bet_type=BetType.MONEYLINE,
            home_team="Chiefs",
            away_team="Bills",
            home_value=home_value,
            away_value=-home_value,
            timestamp=datetime(2026, 1, 1, ts_hour, 0, tzinfo=timezone.utc),
        )

    with LineStore(db) as store:
        store.save_lines([_line(-150, 10)])
        store.save_lines([_line(-170, 11)])
        store.save_lines([_line(-171, 12)])

    ret = main(["moves", "--db", str(db), "--threshold", "5"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "No line movements detected." in out
