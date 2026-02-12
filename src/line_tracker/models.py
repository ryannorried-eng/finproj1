"""Data models for betting lines."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class BetType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


@dataclass
class BettingLine:
    """A single betting line from a sportsbook."""

    sportsbook: str
    sport: str
    event: str
    bet_type: BetType
    home_team: str
    away_team: str
    home_value: float
    away_value: float
    timestamp: datetime
    home_price: float | None = None
    away_price: float | None = None

    # For moneylines: home_value/away_value are the odds (e.g., -150, +130)
    # For spreads: values are the spread, prices are the juice
    # For totals: home_value is over number, away_value is under number,
    #             home_price/away_price are the juice for over/under
