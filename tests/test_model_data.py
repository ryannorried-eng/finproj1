"""Tests for line_tracker.model.data — NCAAB data pipeline."""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pandas as pd
import pytest

from line_tracker.model import data as mod

# ---------------------------------------------------------------------------
# Fixtures — synthetic Torvik & ESPN JSON payloads
# ---------------------------------------------------------------------------

_TEAM_RESULTS_ROW_DUKE = [
    1, "Duke", "ACC", "32-2",
    128.16, 4, 90.81, 1, 0.981, 1,
    33.0, 2.0, 17.0, 1.0, "17-1",
    0.74, 0.62, 0.80, 0.73, 0.62, 0.80, 0.62, 0.64,
    116.6, 103.0, 116.3, 103.2, 129.2, 90.8, 128.1, 89.1,
    0.985, 19.0, 0.07, 1457.0, 1123.0, 1175.0, 1.24, 0.96,
    0.0, 0.94, 13.68, 2, 40, 65.80,
]

_TEAM_RESULTS_ROW_UNC = [
    15, "North Carolina", "ACC", "22-11",
    120.50, 12, 98.30, 20, 0.920, 15,
    22.0, 11.0, 12.0, 6.0, "12-6",
    0.60, 0.55, 0.65, 0.60, 0.55, 0.65, 0.50, 0.52,
    110.0, 100.0, 110.0, 100.0, 121.0, 99.0, 120.0, 98.0,
    0.930, 18.0, 0.05, 1300.0, 1100.0, 1100.0, 1.18, 0.90,
    0.0, 0.67, 8.50, 6, 33, 69.50,
]

FAKE_TEAM_RESULTS = [_TEAM_RESULTS_ROW_DUKE, _TEAM_RESULTS_ROW_UNC]

_SLICE_ROW_DUKE = [
    "Duke", 127.0, 90.8, 0.979, "32\u201302", 32, 34,
    56.8, 46.2, 37.8, 23.7, 15.7, 18.1, 38.1, 24.8,
    67.1, 60.1, 46.6, 35.1, 30.4, 10.4, 9.3,
    59.2, 51.4, 44.4, 45.7, 66.8,
    "", "", "", 2026, "", "", "", 13.68, 72.4, 71.7,
]

_SLICE_ROW_UNC = [
    "North Carolina", 120.5, 98.3, 0.920, "22\u201311", 22, 33,
    53.0, 49.0, 33.0, 29.0, 17.0, 16.0, 32.0, 28.0,
    65.0, 55.0, 42.0, 38.0, 28.0, 8.0, 7.0,
    55.0, 50.0, 40.0, 42.0, 64.0,
    "", "", "", 2026, "", "", "", 8.50, 70.0, 69.0,
]

FAKE_SLICE = [_SLICE_ROW_DUKE, _SLICE_ROW_UNC]

# Two perspectives of the same game
FAKE_GAME_STATS = [
    [
        "01/15/26", 1, "Duke", "ACC", "North Carolina", "H",
        "W, 85-70",
        120.0, 90.0, 115.0, 95.0,
        40.0, 45.0, 80.0, 35.0, 42.0, 78.0, 70.0, 68.0,
        0.90, "ACC",
        1, 2026, 70.0, "DukeNorth Carolina1-15",
        # indices 25-30 (extra fields in real data)
        0, 0, 0, 0, 0, 0,
    ],
    [
        "01/15/26", 1, "North Carolina", "ACC", "Duke", "A",
        "L, 85-70",
        95.0, 115.0, 90.0, 120.0,
        35.0, 42.0, 78.0, 40.0, 45.0, 80.0, 68.0, 70.0,
        0.10, "ACC",
        2, 2026, 70.0, "DukeNorth Carolina1-15",
        0, 0, 0, 0, 0, 0,
    ],
    # A neutral-site game
    [
        "03/15/26", 2, "Duke", "ACC", "Michigan", "N",
        "W, 78-72",
        125.0, 92.0, 120.0, 95.0,
        38.0, 44.0, 82.0, 36.0, 43.0, 79.0, 72.0, 70.0,
        0.85, "B10",
        1, 2026, 68.0, "DukeMichigan3-15",
        0, 0, 0, 0, 0, 0,
    ],
    [
        "03/15/26", 2, "Michigan", "B10", "Duke", "N",
        "L, 78-72",
        95.0, 120.0, 92.0, 125.0,
        36.0, 43.0, 79.0, 38.0, 44.0, 82.0, 70.0, 72.0,
        0.15, "ACC",
        2, 2026, 68.0, "DukeMichigan3-15",
        0, 0, 0, 0, 0, 0,
    ],
]

FAKE_ESPN_SCOREBOARD = {
    "events": [
        {
            "id": "401856400",
            "date": "2026-03-20T23:00:00Z",
            "competitions": [
                {
                    "neutralSite": True,
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {"displayName": "Duke Blue Devils"},
                        },
                        {
                            "homeAway": "away",
                            "team": {"displayName": "Michigan Wolverines"},
                        },
                    ],
                }
            ],
        }
    ]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(payload: object, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response with JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("GET", "https://fake"),
    )


class FakeClient:
    """Stand-in for httpx.Client that returns pre-configured responses."""

    def __init__(self, responses: dict[str, object]):
        # url-substring → payload
        self._responses = responses

    def get(self, url: str, **kwargs) -> httpx.Response:
        for key, payload in self._responses.items():
            if key in url:
                return _mock_response(payload)
        return _mock_response({}, status_code=404)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Tests — fetch_team_ratings
# ---------------------------------------------------------------------------


class TestFetchTeamRatings:
    def test_parses_ratings_correctly(self, tmp_path):
        client = FakeClient({
            "team_results.json": FAKE_TEAM_RESULTS,
            "teamslicejson.php": FAKE_SLICE,
        })
        with (
            patch.object(mod, "_get_client", return_value=client),
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df = mod.fetch_team_ratings(2026)

        assert len(df) == 2
        duke = df[df["team"] == "Duke"].iloc[0]
        assert duke["conf"] == "ACC"
        assert duke["adj_oe"] == pytest.approx(128.16, abs=0.01)
        assert duke["adj_de"] == pytest.approx(90.81, abs=0.01)
        assert duke["adj_em"] == pytest.approx(128.16 - 90.81, abs=0.01)
        assert duke["adj_t"] == pytest.approx(65.80, abs=0.01)
        assert duke["barthag"] == pytest.approx(0.981, abs=0.001)
        assert duke["efg_o"] == pytest.approx(56.8)
        assert duke["efg_d"] == pytest.approx(46.2)
        assert duke["to_o"] == pytest.approx(15.7)
        assert duke["to_d"] == pytest.approx(18.1)
        assert duke["or_pct"] == pytest.approx(38.1)
        assert duke["ftr_o"] == pytest.approx(37.8)
        assert duke["ftr_d"] == pytest.approx(23.7)
        assert duke["sos"] == pytest.approx(13.68, abs=0.01)
        assert duke["wins"] == 32
        assert duke["losses"] == 2

    def test_caching_reads_parquet(self, tmp_path):
        """Second call should read from cache, not hit network."""
        client = FakeClient({
            "team_results.json": FAKE_TEAM_RESULTS,
            "teamslicejson.php": FAKE_SLICE,
        })
        with (
            patch.object(mod, "_get_client", return_value=client),
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df1 = mod.fetch_team_ratings(2026)

        # Now fetch again — should come from cache (no client needed)
        with (
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df2 = mod.fetch_team_ratings(2026)

        pd.testing.assert_frame_equal(df1, df2)

    def test_past_season_cache_permanent(self, tmp_path):
        """Past-season caches should not expire."""
        client = FakeClient({
            "team_results.json": FAKE_TEAM_RESULTS,
            "teamslicejson.php": FAKE_SLICE,
        })
        with (
            patch.object(mod, "_get_client", return_value=client),
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
            patch.object(mod, "current_season", return_value=2026),
        ):
            mod.fetch_team_ratings(2025)

        cache_file = tmp_path / "2025_ratings.parquet"
        assert cache_file.exists()

        # Verify cache is considered valid for past season
        with patch.object(mod, "current_season", return_value=2026):
            assert mod._is_cache_valid(cache_file, 2025) is True

    def test_current_season_cache_expires(self, tmp_path):
        """Current-season cache should expire after TTL."""
        cache_file = tmp_path / "2026_ratings.parquet"
        # Create a fake cache file with old mtime
        pd.DataFrame({"team": ["Duke"]}).to_parquet(cache_file)

        # Set mtime to 7 hours ago (beyond 6h TTL)
        old_time = time.time() - 7 * 3600
        import os
        os.utime(cache_file, (old_time, old_time))

        with patch.object(mod, "current_season", return_value=2026):
            assert mod._is_cache_valid(cache_file, 2026) is False

    def test_default_season(self):
        season = mod.current_season()
        assert isinstance(season, int)
        assert 2020 <= season <= 2030


# ---------------------------------------------------------------------------
# Tests — fetch_game_results
# ---------------------------------------------------------------------------


class TestFetchGameResults:
    def test_deduplicates_and_parses(self, tmp_path):
        client = FakeClient({"getgamestats.php": FAKE_GAME_STATS})
        with (
            patch.object(mod, "_get_client", return_value=client),
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df = mod.fetch_game_results(2026)

        # 4 raw rows → 2 unique games
        assert len(df) == 2

        # Check the Duke-UNC game (home game)
        duke_unc = df[df["game_id"] == "DukeNorth Carolina1-15"].iloc[0]
        assert duke_unc["home_team"] == "Duke"
        assert duke_unc["away_team"] == "North Carolina"
        assert duke_unc["home_score"] == 85
        assert duke_unc["away_score"] == 70
        assert duke_unc["margin"] == 15
        assert duke_unc["neutral_site"] == False  # noqa: E712

    def test_neutral_site_flag(self, tmp_path):
        client = FakeClient({"getgamestats.php": FAKE_GAME_STATS})
        with (
            patch.object(mod, "_get_client", return_value=client),
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df = mod.fetch_game_results(2026)

        tourney = df[df["game_id"] == "DukeMichigan3-15"].iloc[0]
        assert tourney["neutral_site"] == True  # noqa: E712
        assert tourney["home_team"] == "Duke"
        assert tourney["home_score"] == 78
        assert tourney["away_score"] == 72
        assert tourney["margin"] == 6

    def test_game_results_caching(self, tmp_path):
        client = FakeClient({"getgamestats.php": FAKE_GAME_STATS})
        with (
            patch.object(mod, "_get_client", return_value=client),
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df1 = mod.fetch_game_results(2026)

        with (
            patch.object(mod, "_CACHE_DIR", tmp_path),
            patch.object(mod, "_cache_dir", return_value=tmp_path),
        ):
            df2 = mod.fetch_game_results(2026)

        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Tests — fetch_schedule
# ---------------------------------------------------------------------------


class TestFetchSchedule:
    def test_parses_espn_response(self):
        client = FakeClient({"scoreboard": FAKE_ESPN_SCOREBOARD})
        with patch.object(mod, "_get_client", return_value=client):
            df = mod.fetch_schedule()

        assert len(df) >= 1
        row = df.iloc[0]
        assert row["home_team"] == "Duke Blue Devils"
        assert row["away_team"] == "Michigan Wolverines"
        assert row["neutral_site"] == True  # noqa: E712
        assert row["espn_id"] == "401856400"

    def test_empty_schedule_returns_empty_df(self):
        client = FakeClient({"scoreboard": {"events": []}})
        with patch.object(mod, "_get_client", return_value=client):
            df = mod.fetch_schedule()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_handles_http_error_gracefully(self):
        """If ESPN returns errors, schedule should return empty rather than crash."""
        class FailClient:
            def get(self, *args, **kwargs):
                raise httpx.HTTPError("connection failed")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch.object(mod, "_get_client", return_value=FailClient()):
            df = mod.fetch_schedule()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
