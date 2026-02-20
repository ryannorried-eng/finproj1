"""Regression tests for Streamlit Cloud display fixes.

Covers:
  - config reads secrets over env
  - commence_time parsing produces aware UTC datetimes
  - display time converts correctly to Central (or detected tz)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

# ── test_config_reads_secrets_over_env ────────────────────────────────────

def _reload_config():
    """Force-reload config module to pick up patched environment."""
    import importlib

    import line_tracker.config as cfg

    importlib.reload(cfg)
    return cfg


def test_config_reads_env_when_no_secrets(monkeypatch):
    """get_api_key returns env var when st.secrets is unavailable."""
    monkeypatch.setenv("ODDS_API_KEY", "env-key-123")

    mock_st = MagicMock()
    mock_st.secrets.get.side_effect = Exception("no secrets")

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        assert cfg._read_secret("ODDS_API_KEY") == "env-key-123"


def test_config_reads_secrets_over_env(monkeypatch):
    """st.secrets takes priority over os.environ."""
    monkeypatch.setenv("ODDS_API_KEY", "env-key-123")

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = "secret-key-456"

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        assert cfg._read_secret("ODDS_API_KEY") == "secret-key-456"


def test_config_missing_key_raises(monkeypatch):
    """get_api_key raises ValueError when key is absent everywhere."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = None

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        try:
            cfg.get_api_key()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "ODDS_API_KEY" in str(exc)


# ── test_parse_commence_time_is_aware_utc ─────────────────────────────────

def test_parse_timestamp_z_suffix():
    """ISO 8601 timestamps with 'Z' produce timezone-aware UTC datetimes."""
    from line_tracker.scraper import _parse_timestamp

    dt = _parse_timestamp("2024-02-10T18:00:00Z")
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2024
    assert dt.month == 2
    assert dt.day == 10
    assert dt.hour == 18
    assert dt.minute == 0


def test_parse_timestamp_with_offset():
    """ISO 8601 with explicit offset is also parsed as UTC."""
    from line_tracker.scraper import _parse_timestamp

    dt = _parse_timestamp("2024-02-10T18:00:00+00:00")
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


def test_betting_line_commence_time_is_aware():
    """BettingLine from scraper._parse_events carries aware commence_time."""
    from line_tracker.scraper import _parse_events

    events = [
        {
            "id": "evt1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-12-25T20:00:00Z",
            "bookmakers": [
                {
                    "title": "TestBook",
                    "last_update": "2024-12-25T19:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Team A", "price": -150},
                                {"name": "Team B", "price": +130},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    lines = _parse_events(events, "basketball_nba")
    assert len(lines) >= 1
    ct = lines[0].commence_time
    assert ct is not None
    assert ct.tzinfo is not None
    assert ct.tzinfo == timezone.utc


# ── test_display_time_converts_to_central ─────────────────────────────────

def test_display_time_converts_to_central():
    """UTC 18:00 on a winter day should display as 12:00 PM CST (UTC-6)."""
    utc_dt = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
    central = ZoneInfo("America/Chicago")
    local = utc_dt.astimezone(central)
    assert local.hour == 12  # CST = UTC-6
    assert local.minute == 0


def test_display_time_converts_to_central_summer():
    """UTC 18:00 in summer should display as 1:00 PM CDT (UTC-5)."""
    utc_dt = datetime(2024, 7, 15, 18, 0, 0, tzinfo=timezone.utc)
    central = ZoneInfo("America/Chicago")
    local = utc_dt.astimezone(central)
    assert local.hour == 13  # CDT = UTC-5
    assert local.minute == 0


def test_display_time_converts_to_utc():
    """UTC time displayed in UTC should be unchanged."""
    utc_dt = datetime(2024, 1, 15, 18, 30, 0, tzinfo=timezone.utc)
    utc_tz = ZoneInfo("UTC")
    local = utc_dt.astimezone(utc_tz)
    assert local.hour == 18
    assert local.minute == 30


def test_date_grouping_respects_timezone():
    """A UTC time near midnight should group into the correct local date.

    23:30 UTC on Jan 15 is 5:30 PM CST on Jan 15 (same day).
    01:30 UTC on Jan 16 is 7:30 PM CST on Jan 15 (previous day!).
    """
    central = ZoneInfo("America/Chicago")

    # 23:30 UTC Jan 15 → 5:30 PM CST Jan 15
    dt1 = datetime(2024, 1, 15, 23, 30, tzinfo=timezone.utc)
    assert dt1.astimezone(central).date().day == 15

    # 01:30 UTC Jan 16 → 7:30 PM CST Jan 15
    dt2 = datetime(2024, 1, 16, 1, 30, tzinfo=timezone.utc)
    assert dt2.astimezone(central).date().day == 15


# ── test_get_db_path / has_persistent_db ──────────────────────────────────


def test_get_db_path_reads_env(monkeypatch, tmp_path):
    """get_db_path returns the DB_PATH environment variable when set."""
    db = str(tmp_path / "custom.db")
    monkeypatch.setenv("DB_PATH", db)

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = None

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        assert cfg.get_db_path() == db


def test_get_db_path_none_when_unset(monkeypatch):
    """get_db_path returns None when DB_PATH is not set anywhere."""
    monkeypatch.delenv("DB_PATH", raising=False)

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = None

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        assert cfg.get_db_path() is None


def test_has_persistent_db_true_when_db_path_file_exists(monkeypatch, tmp_path):
    """has_persistent_db returns True when DB_PATH points to an existing file."""
    db = tmp_path / "custom.db"
    db.touch()
    monkeypatch.setenv("DB_PATH", str(db))

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = None

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        assert cfg.has_persistent_db() is True


def test_has_persistent_db_false_when_db_path_file_missing(monkeypatch, tmp_path):
    """has_persistent_db returns False when DB_PATH is set but file doesn't exist yet."""
    db = str(tmp_path / "nonexistent.db")
    monkeypatch.setenv("DB_PATH", db)

    mock_st = MagicMock()
    mock_st.secrets.get.return_value = None

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        cfg = _reload_config()
        assert cfg.has_persistent_db() is False
