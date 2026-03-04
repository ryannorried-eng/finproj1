"""Client for The Odds API (https://the-odds-api.com)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

from line_tracker.models import BettingLine, BetType

BASE_URL = "https://api.the-odds-api.com/v4"

MARKET_TO_BET_TYPE = {
    "h2h": BetType.MONEYLINE,
    "spreads": BetType.SPREAD,
    "totals": BetType.TOTAL,
}


class OddsClient:
    """Fetches live odds from The Odds API and returns BettingLine objects."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key= or set ODDS_API_KEY env var. "
                "Get a free key at https://the-odds-api.com"
            )
        self._client = httpx.Client(timeout=30)

    def get_sports(self) -> list[dict]:
        """List available sports (does not count against quota)."""
        try:
            resp = self._client.get(
                f"{BASE_URL}/sports/", params={"apiKey": self.api_key}
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise ValueError("Invalid API key or access denied.") from exc
            raise
        return resp.json()

    def get_odds(
        self,
        sport: str = "americanfootball_nfl",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        bookmakers: str | None = None,
    ) -> list[BettingLine]:
        """Fetch live odds for a sport and return as BettingLine objects."""
        try:
            params: dict[str, str] = {
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            }
            if bookmakers:
                params["bookmakers"] = bookmakers
            resp = self._client.get(
                f"{BASE_URL}/sports/{sport}/odds/",
                params=params,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise ValueError("Invalid API key or access denied.") from exc
            if exc.response.status_code == 422:
                raise ValueError(
                    f"Invalid sport key: {sport!r}"
                ) from exc
            raise
        return _parse_events(resp.json(), sport)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _parse_events(events: list[dict], sport: str) -> list[BettingLine]:
    """Convert raw API response into BettingLine objects."""
    lines: list[BettingLine] = []
    for event in events:
        home_team = event["home_team"]
        away_team = event["away_team"]
        event_name = f"{away_team} @ {home_team}"
        api_event_id = event.get("id")
        ct_raw = event.get("commence_time")
        commence_time = _parse_timestamp(ct_raw) if ct_raw else None

        for bookmaker in event.get("bookmakers", []):
            sportsbook = bookmaker["title"]
            updated = _parse_timestamp(bookmaker["last_update"])

            for market in bookmaker.get("markets", []):
                bet_type = MARKET_TO_BET_TYPE.get(market["key"])
                if bet_type is None:
                    continue

                line = _parse_market(
                    outcomes=market["outcomes"],
                    bet_type=bet_type,
                    sportsbook=sportsbook,
                    sport=sport,
                    event_name=event_name,
                    home_team=home_team,
                    away_team=away_team,
                    timestamp=updated,
                )
                if line is not None:
                    line.commence_time = commence_time
                    line.api_event_id = api_event_id
                    lines.append(line)
    return lines


def _parse_market(
    outcomes: list[dict],
    bet_type: BetType,
    sportsbook: str,
    sport: str,
    event_name: str,
    home_team: str,
    away_team: str,
    timestamp: datetime,
) -> BettingLine | None:
    """Parse a single market's outcomes into a BettingLine."""
    if bet_type == BetType.MONEYLINE:
        return _parse_moneyline(
            outcomes, sportsbook, sport, event_name, home_team, away_team, timestamp
        )
    elif bet_type == BetType.SPREAD:
        return _parse_spread(
            outcomes, sportsbook, sport, event_name, home_team, away_team, timestamp
        )
    elif bet_type == BetType.TOTAL:
        return _parse_total(
            outcomes, sportsbook, sport, event_name, home_team, away_team, timestamp
        )
    return None


def _parse_moneyline(
    outcomes, sportsbook, sport, event_name, home_team, away_team, timestamp
) -> BettingLine | None:
    home_odds = _find_outcome(outcomes, home_team)
    away_odds = _find_outcome(outcomes, away_team)
    if home_odds is None or away_odds is None:
        return None
    return BettingLine(
        sportsbook=sportsbook,
        sport=sport,
        event=event_name,
        bet_type=BetType.MONEYLINE,
        home_team=home_team,
        away_team=away_team,
        home_value=home_odds["price"],
        away_value=away_odds["price"],
        timestamp=timestamp,
    )


def _parse_spread(
    outcomes, sportsbook, sport, event_name, home_team, away_team, timestamp
) -> BettingLine | None:
    home = _find_outcome(outcomes, home_team)
    away = _find_outcome(outcomes, away_team)
    if home is None or away is None:
        return None
    return BettingLine(
        sportsbook=sportsbook,
        sport=sport,
        event=event_name,
        bet_type=BetType.SPREAD,
        home_team=home_team,
        away_team=away_team,
        home_value=home.get("point", 0),
        away_value=away.get("point", 0),
        home_price=home["price"],
        away_price=away["price"],
        timestamp=timestamp,
    )


def _parse_total(
    outcomes, sportsbook, sport, event_name, home_team, away_team, timestamp
) -> BettingLine | None:
    over = _find_outcome(outcomes, "Over")
    under = _find_outcome(outcomes, "Under")
    if over is None or under is None:
        return None
    return BettingLine(
        sportsbook=sportsbook,
        sport=sport,
        event=event_name,
        bet_type=BetType.TOTAL,
        home_team=home_team,
        away_team=away_team,
        home_value=over.get("point", 0),
        away_value=under.get("point", 0),
        home_price=over["price"],
        away_price=under["price"],
        timestamp=timestamp,
    )


def _find_outcome(outcomes: list[dict], name: str) -> dict | None:
    """Find an outcome by team/side name."""
    for o in outcomes:
        if o["name"] == name:
            return o
    return None


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO 8601 timestamp from the API."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc
    )
