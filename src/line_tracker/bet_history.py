"""Bet history data layer: Bet model, session-state helpers, and DB persistence."""

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
from line_tracker.models import BetType


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


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------

def persist_bet(bet: Bet, store) -> str:
    """Write a Bet and its legs to SQLite in a single transaction.

    *store* is a ``LineStore`` instance.  Returns the bet_id.
    """
    bet_row = {
        "bet_id": bet.id,
        "created_at": bet.created_at,
        "sportsbook": bet.sportsbook,
        "stake": bet.stake,
        "total_odds_american": bet.combined_american,
        "total_odds_decimal": bet.combined_decimal,
        "potential_payout": bet.total_payout,
        "profit": bet.profit,
        "status": bet.status,
        "settled_at": bet.settled_at,
        "outcome": None,
    }
    leg_rows = []
    for lg in bet.legs:
        odds = lg["odds"]
        leg_rows.append({
            "leg_id": uuid.uuid4().hex,
            "sport": lg.get("sport"),
            "market": lg.get("market"),
            "event_name": lg.get("event_name"),
            "selection": lg.get("selection"),
            "line_value": lg.get("line"),
            "odds_american": int(odds),
            "odds_decimal": round(american_to_decimal(odds), 4),
            "sportsbook": lg.get("sportsbook"),
            "pick_timestamp": lg.get("fetched_at"),
            "commence_time": lg.get("commence_time"),
        })
    return store.insert_bet_with_legs(bet_row, leg_rows)


def load_bets_from_db(store, status: str | None = None) -> list[Bet]:
    """Read bets (and their legs) from SQLite and return Bet objects.

    *store* is a ``LineStore`` instance.
    """
    bet_rows = store.get_bets(status=status)
    bets: list[Bet] = []
    for br in bet_rows:
        leg_rows = store.get_bet_legs(br["bet_id"])
        legs = [
            {
                "sport": lr.get("sport", ""),
                "event_name": lr.get("event_name", ""),
                "sportsbook": lr.get("sportsbook", ""),
                "market": lr.get("market", ""),
                "selection": lr.get("selection", ""),
                "line": lr.get("line_value"),
                "odds": lr["odds_american"],
                "fetched_at": lr.get("pick_timestamp", ""),
            }
            for lr in leg_rows
        ]
        # Map DB status to Bet status literal
        db_status = br["status"]
        if db_status in ("won", "lost", "push"):
            bet_status = db_status
        else:
            bet_status = "active"
        bets.append(Bet(
            id=br["bet_id"],
            sportsbook=br["sportsbook"],
            legs=legs,
            combined_decimal=br["total_odds_decimal"],
            combined_american=int(br["total_odds_american"]),
            stake=br["stake"],
            profit=br["profit"],
            total_payout=br["potential_payout"],
            status=bet_status,
            created_at=br["created_at"],
            settled_at=br.get("settled_at"),
        ))
    return bets


def settle_bet_persistent(
    bet_id: str,
    outcome: Literal["won", "lost", "push"],
    store,
) -> None:
    """Write settlement to DB (call *after* session-state settle)."""
    # Fetch the bet to compute final P&L
    rows = store.get_bets()
    bet_row = None
    for r in rows:
        if r["bet_id"] == bet_id:
            bet_row = r
            break
    if bet_row is None:
        return
    stake = bet_row["stake"]
    settled_at = _now_iso()
    if outcome == "won":
        profit = bet_row["profit"]
        payout = bet_row["potential_payout"]
    elif outcome == "lost":
        profit = -stake
        payout = 0.0
    else:  # push
        profit = 0.0
        payout = stake
    store.settle_bet_db(bet_id, outcome, settled_at, profit, payout)


def delete_bet_persistent(bet_id: str, store) -> None:
    """Remove a bet and its legs from the database."""
    store.delete_bet_db(bet_id)


# ---------------------------------------------------------------------------
# CLV tracking
# ---------------------------------------------------------------------------

_MARKET_LABEL_TO_BET_TYPE = {
    "ML": BetType.MONEYLINE,
    "Spread": BetType.SPREAD,
    "Total": BetType.TOTAL,
}

_SIDE_MAP = {
    ("ML", "Home"): "home",
    ("ML", "Away"): "away",
    ("Spread", "Home"): "home",
    ("Spread", "Away"): "away",
    ("Total", "Over"): "over",
    ("Total", "Under"): "under",
}


def snapshot_pick(bet: Bet, store) -> None:
    """Persist pick-time consensus snapshot for every leg.

    *store* is a ``LineStore`` instance.  For each leg we look up the
    latest lines, run consensus, and find the matching recommendation.
    """
    from line_tracker.best_bets import recommend_best_bets

    for idx, leg in enumerate(bet.legs):
        event = leg["event_name"]
        market_label = leg["market"]
        bet_type = _MARKET_LABEL_TO_BET_TYPE.get(market_label)
        if bet_type is None:
            continue

        lines = store.get_latest_for_event(event, bet_type)
        if not lines:
            continue

        recs = recommend_best_bets(lines, top_n=10)
        side_key = _SIDE_MAP.get(
            (market_label, leg["selection"]),
        )

        # Find rec matching this side
        rec = None
        for r in recs:
            if r.side == side_key:
                rec = r
                break

        if rec is None:
            continue

        pick_odds = leg["odds"]
        store.save_clv_pick(
            bet_id=bet.id,
            leg_index=idx,
            event=event,
            market=market_label,
            pick_side=leg["selection"],
            pick_line_value=leg.get("line"),
            pick_odds_american=pick_odds,
            pick_odds_decimal=round(
                american_to_decimal(pick_odds), 4,
            ),
            consensus_prob_at_pick=rec.consensus_prob,
            market_hold_median_at_pick=rec.market_hold_median,
            market_volatility_sigma_at_pick=(
                rec.market_volatility_sigma
            ),
            pick_sportsbook=leg.get("sportsbook"),
            sport=leg.get("sport"),
            confidence_at_pick=rec.confidence,
            quality_tier_at_pick=rec.quality_tier,
            edge_pct_at_pick=rec.edge_pct,
            edge_z_at_pick=rec.edge_z,
            books_used_at_pick=rec.books_used_count,
            agreement_score_at_pick=rec.agreement_score,
        )


def close_bet_clv(bet_id: str, store) -> None:
    """Fetch closing lines and write CLV close snapshot.

    For each open CLV row (no closed_at), re-fetch latest lines,
    compute consensus, and store closing odds/prob.
    """
    from line_tracker.best_bets import recommend_best_bets

    rows = store.get_clv(bet_id)
    for row in rows:
        if row.get("closed_at"):
            continue

        event = row["event"]
        market_label = row["market"]
        bet_type = _MARKET_LABEL_TO_BET_TYPE.get(market_label)
        if bet_type is None:
            continue

        lines = store.get_latest_for_event(event, bet_type)
        if not lines:
            continue

        recs = recommend_best_bets(lines, top_n=10)
        side_key = _SIDE_MAP.get(
            (market_label, row["pick_side"]),
        )

        rec = None
        for r in recs:
            if r.side == side_key:
                rec = r
                break

        if rec is None:
            continue

        store.close_clv(
            bet_id=bet_id,
            leg_index=row["leg_index"],
            consensus_prob_close=rec.consensus_prob,
            best_odds_close_american=rec.best_odds,
            best_odds_close_decimal=round(
                american_to_decimal(rec.best_odds), 4,
            ),
        )


def compute_clv(row: dict) -> dict | None:
    """Compute CLV metrics from a single bet_clv row.

    Returns dict with clv_decimal and clv_prob, or None
    if the row has no closing data.
    """
    if row.get("best_odds_close_decimal") is None:
        return None
    pick_dec = row["pick_odds_decimal"]
    close_dec = row["best_odds_close_decimal"]
    pick_prob = row["consensus_prob_at_pick"]
    close_prob = row.get("consensus_prob_close", pick_prob)
    return {
        "clv_decimal": round(close_dec - pick_dec, 4),
        "clv_prob": round(close_prob - pick_prob, 4),
    }
