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

    # For spreads: home_value/away_value are the spread numbers
    # For totals: home_value is the over/under number, away_value is unused
    # For moneylines: home_value/away_value are the odds (e.g., -150, +130)
