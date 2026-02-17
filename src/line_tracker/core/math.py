"""Shared betting math helpers (odds, payout, EV, parlay, Kelly)."""

from __future__ import annotations

import hashlib

_KELLY_CAP = 0.25
_CONFIDENCE_MULTIPLIER: dict[str, float] = {
    "High": 1.0,
    "Medium": 0.5,
    "Low": 0.25,
}


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 0:
        return odds / 100 + 1
    return 100 / abs(odds) + 1


def decimal_to_american(dec: float) -> float:
    """Convert decimal odds to American odds."""
    if dec >= 2.0:
        return round((dec - 1) * 100, 2)
    if dec <= 1.0:
        return 0.0
    return round(-100 / (dec - 1), 2)


def implied_probability(odds: float) -> float:
    """Convert American odds to implied probability in [0, 1]."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return 0.5


def american_profit(stake: float, odds: float) -> float:
    """Profit from a winning straight bet at American odds."""
    dec = american_to_decimal(odds)
    return round(stake * (dec - 1), 2)


def american_total_return(stake: float, odds: float) -> float:
    """Total return (stake + profit) from a winning straight bet."""
    dec = american_to_decimal(odds)
    return round(stake * dec, 2)


def roi(stake: float, profit: float) -> float:
    """Return on investment as a decimal ratio (profit/stake)."""
    if stake == 0:
        return 0.0
    return profit / stake


def ev_per_dollar(prob: float, odds: float) -> float:
    """Expected value per $1 staked given win probability and American odds."""
    dec = american_to_decimal(odds)
    return prob * (dec - 1) - (1 - prob)


def parlay_payout(stake: float, legs_american: list[float]) -> dict:
    """Compute parlay payout from American-odds legs via decimal multiplication."""
    if not legs_american:
        raise ValueError("Parlay requires at least one leg.")

    combined_dec = 1.0
    for odds in legs_american:
        combined_dec *= american_to_decimal(odds)

    total_return = round(stake * combined_dec, 2)
    return {
        "combined_decimal": round(combined_dec, 4),
        "combined_american": decimal_to_american(combined_dec),
        "profit": round(total_return - stake, 2),
        "total_return": total_return,
    }


def kelly_fraction(p: float, decimal_odds: float, cap: float = _KELLY_CAP) -> float:
    """Compute clipped Kelly fraction for win probability and decimal odds."""
    if decimal_odds <= 1.0 or p <= 0.0:
        return 0.0
    raw = (p * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return min(max(raw, 0.0), cap)


def kelly_suggested(
    p: float,
    decimal_odds: float,
    confidence: str,
    cap: float = _KELLY_CAP,
) -> float:
    """Kelly fraction scaled by confidence multiplier."""
    base = kelly_fraction(p, decimal_odds, cap)
    mult = _CONFIDENCE_MULTIPLIER.get(confidence, 0.25)
    return round(base * mult, 6)


def build_recommendation_id(rec) -> str:
    """Build a deterministic recommendation ID from a recommendation.

    Accepts a ``BetRecommendation`` dataclass, a dict, or any object
    exposing the required attributes: *market*, *selection*, *line*,
    *best_sportsbook*, *best_odds*.

    Returns a 16-character hex digest (SHA-256 prefix).
    """

    def _get(obj, key, default=""):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    parts = [
        str(_get(rec, "market")),
        str(_get(rec, "selection")),
        str(_get(rec, "line")),
        str(_get(rec, "best_sportsbook")),
        str(_get(rec, "best_odds")),
    ]
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
