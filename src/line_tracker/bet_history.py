"""Bet history data layer: Bet model and session-state helpers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
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
) -> Bet:
    """Move a bet from active_bets to settled_bets with the given outcome."""
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
            state["settled_bets"].append(active.pop(i))
            return bet

    raise KeyError(f"No active bet with id {bet_id!r}")


def delete_bet(state: dict, bet_id: str) -> Bet:
    """Remove a bet from active_bets or settled_bets and return it."""
    init_bet_state(state)

    for lst in (state["active_bets"], state["settled_bets"]):
        for i, bet in enumerate(lst):
            if bet.id == bet_id:
                return lst.pop(i)

    raise KeyError(f"No bet with id {bet_id!r}")
