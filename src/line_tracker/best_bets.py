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
from statistics import median as _median
from statistics import stdev as _stdev

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


# ---------------------------------------------------------------------------
# Pick-time analytics helpers
# ---------------------------------------------------------------------------

def _compute_pick_analytics(
    leg: dict,
    store: LineStore,
    bet_type: BetType,
    pick_odds: float,
    pick_timestamp: datetime | None = None,
) -> dict:
    """Compute pick-time edge/hold/volatility analytics for a leg.

    Uses stored lines **at or before** *pick_timestamp* to compute how
    this pick compared to the broader market at placement time.  If
    *pick_timestamp* is ``None``, falls back to the latest snapshot
    (legacy behaviour).

    The returned dict includes provenance fields:
      - pick_lines_max_ts: ISO string of the newest line used
      - pick_lines_count: number of sportsbook lines used
    """
    event_name = leg.get("event_name", "")
    selection = leg.get("selection", "")

    # Fetch lines with strict time bound when possible
    if pick_timestamp is not None:
        all_lines = store.get_lines_asof(
            event_name, bet_type, pick_timestamp,
        )
    else:
        all_lines = store.get_latest_for_event(event_name, bet_type)

    # Provenance: track the max timestamp among lines used
    pick_lines_max_ts: str | None = None
    if all_lines:
        max_ts = max(ln.timestamp for ln in all_lines)
        pick_lines_max_ts = max_ts.isoformat()
    pick_lines_count = len(all_lines)

    # Extract relevant odds from each book
    odds_list: list[float] = []
    for ln in all_lines:
        o = _extract_odds_for_selection(ln, bet_type, selection)
        if o is not None:
            odds_list.append(o)

    books_used = len(odds_list)
    if books_used < 2:
        return {
            "edge_pct": None,
            "edge_z": None,
            "books_used": books_used,
            "market_hold_median": None,
            "market_volatility_sigma": None,
            "confidence": "Low",
            "quality_tier": "Tier3",
            "pick_lines_max_ts": pick_lines_max_ts,
            "pick_lines_count": pick_lines_count,
        }

    # Edge: consensus_prob - pick_prob (positive = good)
    probs = [implied_prob_from_american(o) for o in odds_list]
    consensus = _median(probs)
    pick_prob = implied_prob_from_american(pick_odds)
    edge_pct = consensus - pick_prob

    # Edge z-score
    if len(probs) >= 3:
        try:
            sigma = _stdev(probs)
            edge_z = edge_pct / sigma if sigma > 0 else 0.0
        except Exception:
            sigma = 0.0
            edge_z = 0.0
    else:
        sigma = 0.0
        edge_z = edge_pct / 0.01 if edge_pct else 0.0

    # Market hold (overround): sum of implied probs for both sides
    holds: list[float] = []
    for ln in all_lines:
        h, a = _extract_both_probs(ln, bet_type)
        if h is not None and a is not None:
            holds.append(h + a - 1.0)
    hold_median = _median(holds) if holds else None

    # Market volatility
    volatility = sigma

    # Confidence classification
    confidence = _classify_confidence(edge_z, hold_median, books_used)
    tier = _classify_tier(
        edge_pct, edge_z, hold_median, books_used,
    )

    return {
        "edge_pct": round(edge_pct, 6) if edge_pct is not None else None,
        "edge_z": round(edge_z, 4) if edge_z is not None else None,
        "books_used": books_used,
        "market_hold_median": (
            round(hold_median, 6) if hold_median is not None else None
        ),
        "market_volatility_sigma": round(volatility, 6),
        "confidence": confidence,
        "quality_tier": tier,
        "pick_lines_max_ts": pick_lines_max_ts,
        "pick_lines_count": pick_lines_count,
    }


def _extract_odds_for_selection(
    line, bet_type: BetType, selection: str,
) -> float | None:
    """Extract American odds for a specific selection from a line."""
    if bet_type == BetType.MONEYLINE:
        if selection == "Home":
            return line.home_value
        if selection == "Away":
            return line.away_value
    elif bet_type == BetType.SPREAD:
        if selection == "Home":
            return line.home_price
        if selection == "Away":
            return line.away_price
    elif bet_type == BetType.TOTAL:
        if selection == "Over":
            return line.home_price
        if selection == "Under":
            return line.away_price
    return None


def _extract_both_probs(
    line, bet_type: BetType,
) -> tuple[float | None, float | None]:
    """Extract implied probs for both sides of a line."""
    if bet_type == BetType.MONEYLINE:
        h = implied_prob_from_american(line.home_value)
        a = implied_prob_from_american(line.away_value)
        return h, a
    elif bet_type in (BetType.SPREAD, BetType.TOTAL):
        if line.home_price is not None and line.away_price is not None:
            h = implied_prob_from_american(line.home_price)
            a = implied_prob_from_american(line.away_price)
            return h, a
    return None, None


def _classify_confidence(
    edge_z: float,
    hold_median: float | None,
    books_used: int,
) -> str:
    """Classify confidence as High/Medium/Low."""
    if edge_z >= 2.0 and books_used >= 4:
        if hold_median is not None and hold_median < 0.05:
            return "High"
        return "Medium"
    if edge_z >= 1.0:
        return "Medium"
    return "Low"


def _classify_tier(
    edge_pct: float | None,
    edge_z: float,
    hold_median: float | None,
    books_used: int,
) -> str:
    """Classify quality tier: Tier1/Tier2/Tier3/StayAway."""
    if edge_pct is not None and edge_pct < 0:
        return "StayAway"
    hold_ok = hold_median is not None and hold_median < 0.05
    if edge_z >= 1.5 and hold_ok and books_used >= 4:
        return "Tier1"
    hold_ok2 = hold_median is not None and hold_median < 0.08
    if edge_z >= 0.75 and (hold_ok2 or hold_median is None):
        return "Tier2"
    return "Tier3"
