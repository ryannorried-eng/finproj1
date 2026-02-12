"""Bet slip helpers: odds conversion and parlay math."""

from __future__ import annotations


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds.

    Examples: +150 → 2.50, -200 → 1.50, +100 → 2.00
    """
    if odds >= 0:
        return odds / 100 + 1
    return 100 / abs(odds) + 1


def decimal_to_american(dec: float) -> float:
    """Convert decimal odds to American odds.

    Examples: 2.50 → +150, 1.50 → -200
    """
    if dec >= 2.0:
        return round((dec - 1) * 100, 2)
    if dec <= 1.0:
        return 0.0
    return round(-100 / (dec - 1), 2)


def american_profit(stake: float, odds: float) -> float:
    """Calculate profit from a winning straight bet at American odds."""
    dec = american_to_decimal(odds)
    return round(stake * (dec - 1), 2)


def american_total_return(stake: float, odds: float) -> float:
    """Calculate total return (stake + profit) from a winning straight bet."""
    dec = american_to_decimal(odds)
    return round(stake * dec, 2)


def implied_prob_from_american(odds: float) -> float:
    """Convert American odds to implied probability (0–1).

    Examples: -200 → 0.6667, +150 → 0.4000
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    if odds > 0:
        return 100 / (odds + 100)
    return 0.5  # EVEN


def parlay_payout(stake: float, legs_american: list[float]) -> dict:
    """Compute parlay payout from a list of American odds.

    Returns dict with combined_decimal, combined_american, profit, total_return.
    """
    combined_dec = 1.0
    for odds in legs_american:
        combined_dec *= american_to_decimal(odds)

    combined_american = decimal_to_american(combined_dec)
    total_return = round(stake * combined_dec, 2)
    profit = round(total_return - stake, 2)

    return {
        "combined_decimal": round(combined_dec, 4),
        "combined_american": combined_american,
        "profit": profit,
        "total_return": total_return,
    }


def format_american(odds: float) -> str:
    """Format American odds with +/- prefix."""
    if odds > 0:
        return f"+{odds:.0f}"
    if odds < 0:
        return f"{odds:.0f}"
    return "EVEN"


def is_duplicate_leg(slip: list[dict], leg: dict) -> bool:
    """Check if an exact duplicate leg already exists in the slip."""
    for existing in slip:
        if (
            existing["event_name"] == leg["event_name"]
            and existing["sportsbook"] == leg["sportsbook"]
            and existing["market"] == leg["market"]
            and existing["selection"] == leg["selection"]
            and existing["line"] == leg["line"]
            and existing["odds"] == leg["odds"]
        ):
            return True
    return False


def has_conflicting_leg(slip: list[dict], leg: dict) -> bool:
    """Check if a conflicting leg exists (same event+market, different selection)."""
    for existing in slip:
        if (
            existing["event_name"] == leg["event_name"]
            and existing["market"] == leg["market"]
            and existing["selection"] != leg["selection"]
        ):
            return True
    return False
