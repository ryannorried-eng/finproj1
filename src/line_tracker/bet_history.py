"""Bet history data layer: Bet model and session-state helpers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from line_tracker.bet_slip import (
    american_profit,
    american_to_decimal,
    american_total_return,
    decimal_to_american,
    parlay_payout,
)


@dataclass
class Bet:
    """A placed bet (straight or parlay) with full payout info."""

    id: str
    sportsbook: str
    legs: list[dict]
    combined_decimal: float
    combined_american: int
    stake: float
    profit: float
    total_payout: float
    status: Literal["active", "won", "lost", "push"]
    created_at: str
    settled_at: str | None = None
    commence_time: str | None = None
    clv: list[dict] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_bet(
    stake: float,
    sportsbook: str,
    legs: list[dict],
) -> Bet:
    """Build a Bet from a list of leg dicts and a stake.

    Legs follow the existing slip schema (keys: sport, event_name,
    sportsbook, market, selection, line, odds, fetched_at).
    """
    if not legs:
        raise ValueError("A bet must have at least one leg.")
    if stake <= 0:
        raise ValueError("Stake must be positive.")

    odds_list = [leg["odds"] for leg in legs]

    if len(legs) == 1:
        odds = odds_list[0]
        dec = american_to_decimal(odds)
        combined_american = int(decimal_to_american(dec))
        profit = american_profit(stake, odds)
        total_payout = american_total_return(stake, odds)
    else:
        info = parlay_payout(stake, odds_list)
        dec = info["combined_decimal"]
        combined_american = int(info["combined_american"])
        profit = info["profit"]
        total_payout = info["total_return"]

    return Bet(
        id=uuid.uuid4().hex,
        sportsbook=sportsbook,
        legs=list(legs),
        combined_decimal=round(dec, 4),
        combined_american=combined_american,
        stake=stake,
        profit=profit,
        total_payout=total_payout,
        status="active",
        created_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def init_bet_state(state: dict) -> None:
    """Ensure active_bets and settled_bets lists exist in *state*."""
    state.setdefault("active_bets", [])
    state.setdefault("settled_bets", [])


def submit_bet(state: dict, stake: float) -> Bet:
    """Create a Bet from the current slip and move it to active_bets.

    Reads ``state["bet_slip"]`` and ``state["slip_book"]``, clears
    them after submission, and returns the newly created Bet.
    """
    init_bet_state(state)

    slip: list[dict] = state.get("bet_slip", [])
    book: str | None = state.get("slip_book")

    if not slip:
        raise ValueError("Bet slip is empty.")
    if not book:
        raise ValueError("No sportsbook locked on the slip.")

    bet = create_bet(stake=stake, sportsbook=book, legs=slip)
    state["active_bets"].append(bet)

    # Clear the slip
    state["bet_slip"] = []
    state["slip_book"] = None
    state["slip_stake"] = 100.0

    return bet


def settle_bet(
    state: dict,
    bet_id: str,
    outcome: Literal["won", "lost", "push"],
    store=None,
) -> Bet:
    """Move a bet from active_bets to settled_bets with the given outcome.

    If *store* (a :class:`LineStore`) is provided, closing-line value is
    computed and attached to the bet.
    """
    init_bet_state(state)

    if outcome not in ("won", "lost", "push"):
        raise ValueError(f"Invalid outcome: {outcome!r}")

    active: list[Bet] = state["active_bets"]
    for i, bet in enumerate(active):
        if bet.id == bet_id:
            bet.status = outcome
            bet.settled_at = _now_iso()
            if outcome == "lost":
                bet.profit = float(-bet.stake)
                bet.total_payout = 0.0
            elif outcome == "push":
                bet.profit = 0.0
                bet.total_payout = float(bet.stake)
            else:  # won
                bet.profit = float(bet.profit)
                bet.total_payout = float(bet.total_payout)

            # Compute CLV if store available
            if store is not None:
                _attach_clv(bet, store)

            state["settled_bets"].append(active.pop(i))
            return bet

    raise KeyError(f"No active bet with id {bet_id!r}")


def _attach_clv(bet: Bet, store) -> None:
    """Compute and attach CLV data to a settled bet.

    Also persists CLV analytics rows to the ``bet_clv`` table for the
    Performance page.
    """
    from line_tracker.best_bets import (
        CLV_TOLERANCE,
        LegCLV,
        _bet_type_from_market,
        _compute_pick_analytics,
        enrich_bet_with_clv,
    )
    from line_tracker.bet_slip import american_to_decimal

    ct = None
    if bet.commence_time:
        ct = datetime.fromisoformat(bet.commence_time)

    leg_clvs: list[LegCLV] = enrich_bet_with_clv(bet, store, commence_time=ct)
    bet.clv = [
        {
            "best_close_odds": lc.best_close_odds,
            "book_close_odds": lc.book_close_odds,
            "clv_price_prob_best": lc.clv_price_prob_best,
            "clv_price_prob_book": lc.clv_price_prob_book,
            "clv_decimal_best": lc.clv_decimal_best,
            "clv_decimal_book": lc.clv_decimal_book,
            "classification_best": lc.classification_best,
            "classification_book": lc.classification_book,
            "close_estimated": lc.close_estimated,
        }
        for lc in leg_clvs
    ]

    # Persist analytics rows for Performance page
    clv_rows: list[dict] = []
    for i, (leg, lc) in enumerate(zip(bet.legs, leg_clvs)):
        bt = _bet_type_from_market(leg.get("market", ""))
        pick_odds = leg.get("odds")

        # Compute pick-time analytics
        analytics: dict = {}
        if bt is not None and pick_odds is not None:
            analytics = _compute_pick_analytics(leg, store, bt, pick_odds)

        # beat_close flags (1=beat, 0=lost/matched, None=no close)
        def _beat_flag(clv_val):
            if clv_val is None:
                return None
            return 1 if clv_val > CLV_TOLERANCE else 0

        exec_dec = (
            american_to_decimal(lc.book_close_odds)
            if lc.book_close_odds is not None
            else None
        )
        mkt_dec = (
            american_to_decimal(lc.best_close_odds)
            if lc.best_close_odds is not None
            else None
        )

        row = {
            "bet_id": bet.id,
            "leg_index": i,
            "sport": leg.get("sport"),
            "market": leg.get("market"),
            "pick_sportsbook": leg.get("sportsbook", bet.sportsbook),
            "event_name": leg.get("event_name"),
            "selection": leg.get("selection"),
            "pick_line": leg.get("line"),
            "pick_odds": pick_odds,
            "pick_timestamp": leg.get("fetched_at"),
            "commence_time": bet.commence_time,
            "edge_pct": analytics.get("edge_pct"),
            "edge_z": analytics.get("edge_z"),
            "books_used": analytics.get("books_used"),
            "market_hold_median": analytics.get("market_hold_median"),
            "market_volatility_sigma": analytics.get(
                "market_volatility_sigma",
            ),
            "confidence": analytics.get("confidence", "Low"),
            "quality_tier": analytics.get("quality_tier", "Tier3"),
            "close_timestamp": None,
            "close_estimated": lc.close_estimated,
            "exec_close_odds": lc.book_close_odds,
            "market_close_odds": lc.best_close_odds,
            "exec_close_decimal": exec_dec,
            "market_close_decimal": mkt_dec,
            "exec_clv_prob": lc.clv_price_prob_book,
            "market_clv_prob": lc.clv_price_prob_best,
            "beat_close_exec": _beat_flag(lc.clv_price_prob_book),
            "beat_close_market": _beat_flag(lc.clv_price_prob_best),
            "settled_at": bet.settled_at,
            "outcome": bet.status,
        }
        clv_rows.append(row)

    if clv_rows:
        store.save_clv_rows(clv_rows)


def delete_bet(state: dict, bet_id: str) -> Bet:
    """Remove a bet from active_bets or settled_bets and return it."""
    init_bet_state(state)

    for lst in (state["active_bets"], state["settled_bets"]):
        for i, bet in enumerate(lst):
            if bet.id == bet_id:
                return lst.pop(i)

    raise KeyError(f"No bet with id {bet_id!r}")
