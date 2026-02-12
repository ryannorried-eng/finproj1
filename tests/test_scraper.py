"""Tests for the odds scraper (uses mock API data, no network calls)."""

import pytest

from line_tracker.models import BetType
from line_tracker.scraper import OddsClient, _parse_events

# Realistic API response fixture
MOCK_EVENT = {
    "id": "abc123",
    "sport_key": "americanfootball_nfl",
    "home_team": "Kansas City Chiefs",
    "away_team": "Buffalo Bills",
    "commence_time": "2026-01-20T01:00:00Z",
    "bookmakers": [
        {
            "key": "draftkings",
            "title": "DraftKings",
            "last_update": "2026-01-19T18:30:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -150},
                        {"name": "Buffalo Bills", "price": 130},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -110, "point": -2.5},
                        {"name": "Buffalo Bills", "price": -110, "point": 2.5},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": -105, "point": 47.5},
                        {"name": "Under", "price": -115, "point": 47.5},
                    ],
                },
            ],
        },
        {
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": "2026-01-19T18:25:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -145},
                        {"name": "Buffalo Bills", "price": 125},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "price": -108, "point": -2.5},
                        {"name": "Buffalo Bills", "price": -112, "point": 2.5},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": -110, "point": 48.0},
                        {"name": "Under", "price": -110, "point": 48.0},
                    ],
                },
            ],
        },
    ],
}


def test_parse_moneylines():
    lines = _parse_events([MOCK_EVENT], "americanfootball_nfl")
    ml_lines = [ln for ln in lines if ln.bet_type == BetType.MONEYLINE]

    assert len(ml_lines) == 2
    dk = next(ln for ln in ml_lines if ln.sportsbook == "DraftKings")
    assert dk.home_value == -150
    assert dk.away_value == 130
    assert dk.event == "Buffalo Bills @ Kansas City Chiefs"


def test_parse_spreads():
    lines = _parse_events([MOCK_EVENT], "americanfootball_nfl")
    spread_lines = [ln for ln in lines if ln.bet_type == BetType.SPREAD]

    assert len(spread_lines) == 2
    dk = next(ln for ln in spread_lines if ln.sportsbook == "DraftKings")
    assert dk.home_value == -2.5
    assert dk.away_value == 2.5
    assert dk.home_price == -110
    assert dk.away_price == -110


def test_parse_totals():
    lines = _parse_events([MOCK_EVENT], "americanfootball_nfl")
    total_lines = [ln for ln in lines if ln.bet_type == BetType.TOTAL]

    assert len(total_lines) == 2
    fd = next(ln for ln in total_lines if ln.sportsbook == "FanDuel")
    assert fd.home_value == 48.0  # over number
    assert fd.away_value == 48.0  # under number
    assert fd.home_price == -110
    assert fd.away_price == -110


def test_all_markets_parsed():
    lines = _parse_events([MOCK_EVENT], "americanfootball_nfl")
    # 2 bookmakers x 3 markets = 6 lines
    assert len(lines) == 6


def test_client_requires_api_key():
    with pytest.raises(ValueError, match="API key required"):
        OddsClient(api_key="")


def test_empty_event_list():
    lines = _parse_events([], "americanfootball_nfl")
    assert lines == []
