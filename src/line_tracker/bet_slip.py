"""Bet slip helpers: odds conversion and parlay math."""

from __future__ import annotations

from statistics import median as _median


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
        # Decimal odds of 1.0 means zero profit; not a valid line.
        # Return 0.0 as a sentinel (EVEN display).
        return 0.0
    # 1.0 < dec < 2.0 → negative American favourite
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


def breakeven_prob_from_american(odds: float) -> float:
    """Return the breakeven probability for a bet at given American odds.

    This is the minimum win probability needed for the bet to be +EV.
    Numerically identical to implied_prob_from_american, but semantically
    represents the breakeven threshold rather than market-implied probability.
    """
    return implied_prob_from_american(odds)


def ev_per_dollar(prob: float, odds: float) -> float:
    """Expected value per $1 staked given a win probability and American odds.

    EV = p * (decimal - 1) - (1 - p)
    """
    d = american_to_decimal(odds)
    return prob * (d - 1) - (1 - prob)


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


def compute_standouts(entries: list[dict], stake: float = 100.0) -> list[dict]:
    """Compute shopping-value standouts from odds entries.

    Each entry dict must have keys:
        event, market (ML/Spread/Total), selection, sportsbook, odds, line, sport

    Groups by (event, market, selection).  For each group with >= 2 books:
      - consensus_prob  = median implied probability across books
      - edge            = consensus_prob - book_prob
                          (positive ⇒ book offers better-than-consensus odds)
      - dollar_impact   = stake * (book_decimal - median_decimal)
                          (positive ⇒ extra payout vs median book)

    Returns list of dicts sorted by edge descending (best standouts first).
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for e in entries:
        key = (e["event"], e["market"], e["selection"])
        groups.setdefault(key, []).append(e)

    results: list[dict] = []
    for (event, market, selection), items in groups.items():
        if len(items) < 2:
            continue

        probs = [implied_prob_from_american(it["odds"]) for it in items]
        consensus = _median(probs)

        decs = [american_to_decimal(it["odds"]) for it in items]
        median_dec = _median(decs)
        median_american = decimal_to_american(median_dec)

        # d_best / d_ref for execution advantage
        sorted_decs = sorted(decs, reverse=True)
        d_best = sorted_decs[0]
        d_ref = sorted_decs[1] if len(sorted_decs) >= 2 else median_dec

        for it in items:
            bp = implied_prob_from_american(it["odds"])
            bd = american_to_decimal(it["odds"])
            edge = consensus - bp
            impact = round(stake * (bd - median_dec), 2)
            exec_adv = round(
                100.0 * consensus * (d_best - d_ref), 2,
            )

            results.append({
                "event": event,
                "market": market,
                "selection": selection,
                "sportsbook": it["sportsbook"],
                "odds": it["odds"],
                "line": it.get("line"),
                "book_prob": round(bp, 4),
                "consensus_prob": round(consensus, 4),
                "edge": round(edge, 4),
                "dollar_impact": impact,
                "median_odds": round(median_american, 1),
                "sport": it.get("sport", ""),
                "d_best": round(d_best, 4),
                "d_ref": round(d_ref, 4),
                "exec_adv_100": exec_adv,
            })

    results.sort(key=lambda x: x["edge"], reverse=True)
    return results
