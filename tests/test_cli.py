"""Tests for the CLI entry point."""

from line_tracker.__main__ import main


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
