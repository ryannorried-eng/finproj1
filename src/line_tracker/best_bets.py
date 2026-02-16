"""Closing Line Value (CLV) computation.

Primary metric:  clv_price_prob = p_close - p_pick
  where p = implied_prob(decimal) = 1 / decimal_odds.
  Positive means the bettor beat the closing line.

Secondary metric: clv_decimal = decimal_pick - decimal_close
  Positive means the bettor got higher decimal odds than close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from line_tracker.bet_slip import american_to_decimal, implied_prob_from_american
from line_tracker.models import BetType
from line_tracker.storage import LineStore

# Minimum implied-probability difference to classify as "beat" or "lost".
# Differences within this band are "matched".
CLV_TOLERANCE = 0.001


@dataclass
class LegCLV:
    """CLV result for one leg of a bet."""

    best_close_odds: float | None = None
    book_close_odds: float | None = None
    clv_price_prob_best: float | None = None
    clv_price_prob_book: float | None = None
    clv_decimal_best: float | None = None
    clv_decimal_book: float | None = None
    classification_best: str = "no_close"
    classification_book: str = "no_close"
    close_estimated: bool = False


def _classify(clv_prob: float | None) -> str:
    """Classify a CLV prob value as beat / matched / lost / no_close."""
    if clv_prob is None:
        return "no_close"
    if clv_prob > CLV_TOLERANCE:
        return "beat"
    if clv_prob < -CLV_TOLERANCE:
        return "lost"
    return "matched"


def compute_clv(pick_odds: float, close_odds: float) -> dict:
    """Compute CLV between pick American odds and close American odds.

    Returns dict with clv_price_prob, clv_decimal, classification.
    """
    p_pick = implied_prob_from_american(pick_odds)
    p_close = implied_prob_from_american(close_odds)

    d_pick = american_to_decimal(pick_odds)
    d_close = american_to_decimal(close_odds)

    clv_prob = p_close - p_pick  # positive = beat
    clv_dec = d_pick - d_close   # positive = got better decimal

    return {
        "clv_price_prob": round(clv_prob, 6),
        "clv_decimal": round(clv_dec, 6),
        "classification": _classify(clv_prob),
    }


def _pick_odds_for_leg(leg: dict) -> float | None:
    """Extract the American odds from a bet leg dict."""
    return leg.get("odds")


def _find_close_odds_for_leg(
    leg: dict,
    close_lines: list,
    bet_type: BetType,
) -> float | None:
    """Find the closing odds from *close_lines* that match this leg.

    For moneylines: match on selection side (Home/Away).
    For spreads: match on selection side AND line value.
    For totals: match on selection side (Over/Under) AND line value.
    """
    selection = leg.get("selection")
    pick_line = leg.get("line")

    for cl in close_lines:
        if bet_type == BetType.MONEYLINE:
            if selection == "Home":
                return cl.home_value
            elif selection == "Away":
                return cl.away_value

        elif bet_type == BetType.SPREAD:
            if selection == "Home" and cl.home_value == pick_line:
                return cl.home_price
            elif selection == "Away" and cl.away_value == pick_line:
                return cl.away_price

        elif bet_type == BetType.TOTAL:
            if selection == "Over" and cl.home_value == pick_line:
                return cl.home_price
            elif selection == "Under" and cl.away_value == pick_line:
                return cl.away_price

    return None


def _bet_type_from_market(market: str) -> BetType | None:
    """Map leg market label to BetType."""
    mapping = {
        "ML": BetType.MONEYLINE,
        "Spread": BetType.SPREAD,
        "Total": BetType.TOTAL,
    }
    return mapping.get(market)


def enrich_bet_with_clv(
    bet,
    store: LineStore,
    commence_time: datetime | None = None,
) -> list[LegCLV]:
    """Compute CLV for every leg of *bet*.

    Parameters
    ----------
    bet : Bet
        A placed bet with ``.legs`` list and ``.sportsbook``.
    store : LineStore
        Database handle for querying closing lines.
    commence_time : datetime | None
        The game start time.  If ``None``, the function tries to read it
        from stored lines.  If still unavailable, returns ``close_estimated``
        flags on every leg.

    Returns
    -------
    list[LegCLV]
        One CLV result per leg, in order.
    """
    results: list[LegCLV] = []

    for leg in bet.legs:
        lclv = LegCLV()
        bt = _bet_type_from_market(leg.get("market", ""))
        if bt is None:
            results.append(lclv)
            continue

        event_name = leg.get("event_name", "")
        pick_odds = _pick_odds_for_leg(leg)
        if pick_odds is None:
            results.append(lclv)
            continue

        # Resolve commence_time from stored lines if not provided
        ct = commence_time
        if ct is None:
            stored = store.get_lines(event=event_name, bet_type=bt, limit=1)
            if stored and stored[0].commence_time:
                ct = stored[0].commence_time
        if ct is None:
            lclv.close_estimated = True
            results.append(lclv)
            continue

        # Best-market close (all books)
        best_closes = store.get_close_lines(event_name, bt, before=ct)
        best_close_odds = _find_close_odds_for_leg(leg, best_closes, bt)

        # Same-book close
        book_closes = store.get_close_lines(
            event_name, bt, before=ct, sportsbook=bet.sportsbook,
        )
        book_close_odds = _find_close_odds_for_leg(leg, book_closes, bt)

        # For best-market close, pick the *best* odds among all books
        if bt == BetType.MONEYLINE and best_closes:
            candidates = []
            for cl in best_closes:
                o = _find_close_odds_for_leg(leg, [cl], bt)
                if o is not None:
                    candidates.append(o)
            if candidates:
                # "Best" = highest American odds (most favorable to bettor)
                best_close_odds = max(candidates)

        if best_close_odds is not None:
            lclv.best_close_odds = best_close_odds
            cv = compute_clv(pick_odds, best_close_odds)
            lclv.clv_price_prob_best = cv["clv_price_prob"]
            lclv.clv_decimal_best = cv["clv_decimal"]
            lclv.classification_best = cv["classification"]

        if book_close_odds is not None:
            lclv.book_close_odds = book_close_odds
            cv = compute_clv(pick_odds, book_close_odds)
            lclv.clv_price_prob_book = cv["clv_price_prob"]
            lclv.clv_decimal_book = cv["clv_decimal"]
            lclv.classification_book = cv["classification"]

        if best_close_odds is None and book_close_odds is None:
            lclv.close_estimated = True

        results.append(lclv)

    return results
